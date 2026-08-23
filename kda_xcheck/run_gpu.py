# Copyright 2026. Apache-2.0.
"""Run MoonshotAI/FlashKDA on the canonical inputs.

Two implementations:
  --impl kernel     flash_kda.fwd, the CUDA kernel
  --impl torch_ref  FlashKDA's tests/torch_ref.py

Note on `torch_ref`: it is NOT a clean fp32 reference. It deliberately
emulates the kernel bit-for-bit -- inline-PTX `tanh.approx.f32` sigmoid,
cuBLAS GEMMs with CUBLAS_COMPUTE_16F (fp16 accumulation), `exp2` with
flush-to-zero, and an l2-norm that reproduces the warp-shuffle tree
reduction. It therefore also requires CUDA, and it will NOT agree with the
fp32 arbiter to fp32 precision. Its value is as an independent statement of
what the kernel *intends*, so a kernel-vs-torch_ref gap localizes a bug to
the CUDA code rather than to the algorithm.

Unlike the FlashQLA/GDN harness, no reimplementation of a surrounding
pipeline is needed here: FlashKDA's entrypoint covers the same span as
tokamax's op (gate activation, l2norm, and the recurrence are all inside).

Usage (on a CUDA box):
  python run_gpu.py --case all --impl kernel
  python run_gpu.py --case all --impl torch_ref
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import kda_case


def run_case(name: str, impl: str, dtype: str, indir: str, outdir: str,
             flashkda_root: str | None) -> None:
  import torch

  case = kda_case.CASES[name]
  inp = kda_case.load_inputs(os.path.join(indir, f"in_{name}.npz"))
  dev = torch.device("cuda")
  tdt = getattr(torch, dtype)

  t = lambda x, d: torch.as_tensor(np.ascontiguousarray(x), dtype=d, device=dev)

  # Canonical layout [B,T,H,D] is already FlashKDA's layout -- no transpose.
  q = t(inp["q"], tdt)
  k = t(inp["k"], tdt)
  v = t(inp["v"], tdt)
  g = t(inp["g"], tdt)
  # FlashKDA takes beta LOGITS and applies sigmoid internally
  # (__init__.py:13), the opposite of tokamax. Pass the raw logits.
  beta = t(inp["beta_logits"], tdt)
  a_log = t(inp["a_log"], torch.float32)
  dt_bias = t(inp["dt_bias"], torch.float32)
  out = torch.zeros_like(v)

  # FlashKDA's state is V-major [N,H,V,K]; the canonical layout is K-major
  # [N,H,K,V]. Transpose in, and transpose back out.
  init_kv = inp["initial_state"]
  initial_state = (
      t(init_kv.transpose(0, 1, 3, 2), torch.float32)
      if bool(inp["has_initial_state"]) else None)
  final_state = torch.zeros(
      (case.num_states, case.heads, case.head_dim, case.head_dim),
      dtype=torch.float32, device=dev)

  cu = (torch.as_tensor(inp["cu_seqlens"], dtype=torch.int64, device=dev)
        if case.is_varlen else None)

  call = dict(
      q=q, k=k, v=v, g=g, beta=beta, scale=float(case.head_dim ** -0.5),
      out=out, A_log=a_log, dt_bias=dt_bias,
      lower_bound=float(case.lower_bound),
      initial_state=initial_state, final_state=final_state, cu_seqlens=cu,
  )

  if impl == "kernel":
    import flash_kda
    flash_kda.fwd(**call)
  else:
    if flashkda_root:
      sys.path.insert(0, os.path.join(flashkda_root, "tests"))
    import torch_ref as tr
    tr.torch_ref(**call)
  torch.cuda.synchronize()

  # [N,H,V,K] -> canonical [N,H,K,V]
  st = final_state.transpose(-1, -2).contiguous()
  np.savez(
      os.path.join(outdir, f"{'gpu' if impl == 'kernel' else 'gpuref'}_{name}.npz"),
      output=out.float().cpu().numpy(),
      final_state=st.float().cpu().numpy())
  print(f"{impl} {name}: out{tuple(out.shape)} state{tuple(st.shape)} "
        f"dtype={dtype}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--impl", default="kernel", choices=["kernel", "torch_ref"])
  # FlashKDA is bf16-only; the flag exists so a mismatch fails loudly.
  p.add_argument("--dtype", default="bfloat16", choices=["bfloat16"])
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  p.add_argument("--flashkda-root", default=None,
                 help="path to a FlashKDA checkout, for --impl torch_ref")
  args = p.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  names = list(kda_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    run_case(name, args.impl, args.dtype, args.indir, args.outdir,
             args.flashkda_root)


if __name__ == "__main__":
  main()
