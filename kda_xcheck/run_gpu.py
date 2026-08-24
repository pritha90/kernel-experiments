# Copyright 2026. Apache-2.0.
"""Run FLA's KDA on the canonical inputs.

  --impl chunk   fla.ops.kda.chunk_kda -- the Triton kernel (chunk_fwd.py /
                 chunk_bwd.py). Requires CUDA.
  --impl naive   fla.ops.kda.naive.naive_recurrent_kda -- pure PyTorch, so it
                 runs on CPU. This is FLA's own statement of KDA semantics.

The `naive` path is what makes the GPU side checkable without a GPU, which
the previous FlashKDA target could not offer (its torch reference is a
bit-exact CUDA emulation and needs a device). Agreement between `naive` and
the arbiter validates the conversions below before the Triton kernel is ever
run -- the same role `--impl xla` plays on the TPU side.

`naive_recurrent_kda` takes *pre-activated* inputs: `g` already in log space,
`q`/`k` already l2-normalized, `beta` already post-sigmoid, and it has no
varlen support. So this file spells out l2norm + gate activation explicitly
and loops over segments. Those torch ops are differentiable, so the naive
path yields gradients too, by ordinary autograd.

Usage:
  python run_gpu.py --case all --impl naive --backward             # CPU
  python run_gpu.py --case all --impl chunk --dtype bfloat16 --backward
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import kda_case

# Activations cast to the run dtype. `a_log`, `dt_bias`, and the state stay
# fp32 -- FLA asserts `initial_state.dtype == torch.float32` (chunk.py:385).
_CAST = ("q", "k", "v", "g", "beta")


def _l2norm(x, eps: float = 1e-6):
  # 1/sqrt(sum(x*x) + eps), eps inside the sqrt -- matching FLA's
  # l2norm_fwd_kernel (modules/l2norm.py:43) and tokamax's `_l2_normalize`.
  return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _activate_gate(g, a_log, dt_bias, lower_bound):
  """`lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`.

  This is the branch FLA's gate kernel takes whenever `lower_bound is not
  None` -- `USE_LOWER_BOUND` in gate.py:100/163/395 keys off that alone. The
  chunk_kda docstring claims the branch also requires `safe_gate=True`; the
  code disagrees, and `lower_bound` is forwarded unconditionally in both
  chunk_fwd.py:55 and chunk_bwd.py:474. It is the same formula tokamax uses.
  """
  h, dk = g.shape[2], g.shape[3]
  a = torch.exp(a_log)[None, None, :, None]
  return lower_bound * torch.sigmoid(a * (g + dt_bias.view(h, dk)))


def to_torch(case, inp, dtype, device) -> dict:
  """Canonical [B,T,H,D] -> FLA tensors. No layout change is needed.

  FLA's convention *is* the canonical one: token-first `[B,T,H,K]`, flat
  `[H*K]` dt_bias, K-major `[N,H,K,V]` state. Only tokamax transposes.
  """
  dt = getattr(torch, dtype)
  mk = lambda name: torch.as_tensor(
      inp[name], dtype=dt if name in _CAST else torch.float32, device=device
  ).requires_grad_(True)

  out = {n: mk(n) for n in ("q", "k", "v", "g", "beta", "a_log", "dt_bias")}
  out["initial_state"] = (
      torch.as_tensor(inp["initial_state"], dtype=torch.float32,
                      device=device).requires_grad_(True)
      if bool(inp["has_initial_state"]) else None)
  out["cu_seqlens"] = (
      torch.as_tensor(case.cu_seqlens, dtype=torch.int64, device=device)
      if case.is_varlen else None)
  return out


def call_chunk(case, t: dict):
  """The Triton kernel."""
  from fla.ops.kda import chunk_kda

  return chunk_kda(
      t["q"], t["k"], t["v"], t["g"], t["beta"],
      scale=float(case.head_dim ** -0.5),
      initial_state=t["initial_state"],
      output_final_state=True,
      use_qk_l2norm_in_kernel=True,
      use_gate_in_kernel=True,
      # Feed post-sigmoid beta so `db` is d/d(beta_post) on both sides.
      # With True, FLA would additionally chain through sigmoid' and the
      # gradient would not be comparable to tokamax's.
      use_beta_sigmoid_in_kernel=False,
      lower_bound=case.lower_bound,
      # False (the default): safe_gate is an independent TensorCore/clamping
      # option with no tokamax counterpart, so enabling it would confound the
      # comparison. It does NOT select the gate branch -- see _activate_gate.
      safe_gate=False,
      # K-major, like tokamax. `state_v_first=True` would transpose it.
      state_v_first=False,
      cu_seqlens=t["cu_seqlens"],
      A_log=t["a_log"],
      dt_bias=t["dt_bias"],
      # Both backends chunk at 64, so the chunk boundaries coincide and a
      # disagreement cannot be blamed on differing blocking.
      chunk_size=64,
  )


def _naive_recurrent_kda():
  """Import FLA's pure-torch reference, Triton or no Triton.

  `import fla.ops.kda.naive` runs `fla/ops/__init__.py`, which eagerly
  imports every op in the library, and all of them `import triton` -- which
  has no macOS or CPU-only wheel. `naive.py` itself needs only torch and
  einops, so on those hosts load it straight from the source tree, bypassing
  the package __init__ chain. Same file either way.
  """
  try:
    from fla.ops.kda.naive import naive_recurrent_kda
    return naive_recurrent_kda
  except ImportError as first:
    pass

  import importlib.util
  import sys

  roots = ([os.environ["FLA_ROOT"]] if "FLA_ROOT" in os.environ else []) + \
      list(sys.path)
  for root in roots:
    path = os.path.join(root, "fla", "ops", "kda", "naive.py")
    if os.path.exists(path):
      spec = importlib.util.spec_from_file_location("_fla_kda_naive", path)
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      return mod.naive_recurrent_kda
  raise SystemExit(
      f"could not import FLA ({first}), and fla/ops/kda/naive.py was not "
      f"found. Set FLA_ROOT to a checkout of "
      f"fla-org/flash-linear-attention.")


def call_naive(case, t: dict):
  """Pure-torch reference. Runs on CPU."""
  naive_recurrent_kda = _naive_recurrent_kda()

  qn, kn = _l2norm(t["q"].float()), _l2norm(t["k"].float())
  gg = _activate_gate(t["g"].float(), t["a_log"], t["dt_bias"],
                      case.lower_bound)
  h0 = t["initial_state"]
  scale = float(case.head_dim ** -0.5)

  cu = case.cu_seqlens
  outs, finals = [], []
  for n in range(len(cu) - 1):
    lo, hi = int(cu[n]), int(cu[n + 1])
    # naive_recurrent_kda has no varlen path and indexes its state by batch,
    # so each sequence is run as its own B=1 call.
    bi = 0 if case.is_varlen else n
    t0, t1 = (lo, hi) if case.is_varlen else (0, hi - lo)
    sl = lambda x: x[bi:bi + 1, t0:t1]
    o, s = naive_recurrent_kda(
        sl(qn), sl(kn), sl(t["v"].float()), sl(gg), sl(t["beta"].float()),
        scale=scale,
        initial_state=None if h0 is None else h0[n:n + 1],
        output_final_state=True)
    outs.append(o)
    finals.append(s)

  out = (torch.cat(outs, 1) if case.is_varlen else torch.cat(outs, 0))
  return out, torch.cat(finals, 0)


def run(case, t: dict, inp: dict, impl: str, backward: bool) -> dict:
  o, ht = (call_chunk if impl == "chunk" else call_naive)(case, t)
  res = {"output": o, "final_state": ht}
  if backward:
    names = [n for n in kda_case.DIFFERENTIABLE if t[n] is not None]
    do = torch.as_tensor(inp["do"], dtype=o.dtype, device=o.device)
    dht = torch.as_tensor(inp["dht"], dtype=ht.dtype, device=ht.device)
    # allow_unused + zero-fill, mirroring tokamax's VJP, which returns
    # `jnp.zeros_like(a_log)` rather than nothing when a grad is absent.
    # Without this, any configuration that leaves an input out of the graph
    # -- FLA permits `A_log=None` whenever `lower_bound` is set -- raises
    # "does not have been used in the graph" instead of reporting a zero.
    grads = torch.autograd.grad([o, ht], [t[n] for n in names], [do, dht],
                                allow_unused=True)
    res.update({f"d{n}": (g if g is not None else torch.zeros_like(t[n]))
                for n, g in zip(names, grads)})
  return {k: v.detach().to(torch.float32).cpu().numpy()
          for k, v in res.items()}


def run_case(name, impl, dtype, device, indir, outdir, backward) -> None:
  case = kda_case.CASES[name]
  inp = kda_case.load_inputs(os.path.join(indir, f"in_{name}.npz"))
  res = run(case, to_torch(case, inp, dtype, device), inp, impl, backward)
  tag = "gpu" if impl == "chunk" else "gpuref"
  np.savez(os.path.join(outdir, f"{tag}_{name}.npz"), **res)
  print(f"{tag}_{name}: {' '.join(sorted(res))}  dtype={dtype} impl={impl}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--impl", default="naive", choices=["chunk", "naive"])
  p.add_argument("--dtype", default="float32",
                 choices=["bfloat16", "float16", "float32"])
  p.add_argument("--device", default=None,
                 help="default: cuda for --impl chunk, cpu for --impl naive")
  p.add_argument("--backward", action="store_true")
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  args = p.parse_args()

  device = args.device or ("cuda" if args.impl == "chunk" else "cpu")
  if args.impl == "chunk" and not (
      str(device).startswith("cuda") and torch.cuda.is_available()):
    # FLA's KDA kernels are Triton and there is no CPU fallback. Say so here
    # rather than surfacing "Torch not compiled with CUDA enabled" from
    # somewhere inside tensor construction.
    raise SystemExit(
        "--impl chunk requires CUDA (fla.ops.kda is Triton-only). "
        "Use --impl naive to run FLA's semantics on CPU.")

  os.makedirs(args.outdir, exist_ok=True)
  names = list(kda_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    run_case(name, args.impl, args.dtype, device, args.indir, args.outdir,
             args.backward)


if __name__ == "__main__":
  main()
