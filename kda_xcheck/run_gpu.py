# Copyright 2026. Apache-2.0.
"""Run FLA's KDA on the canonical inputs.

  --impl naive        naive_recurrent_kda -- the token-by-token recurrence.
                      FLA's own statement of KDA semantics. Pure torch.
  --impl chunk_torch  naive_chunk_kda -- the chunked algorithm (gate cumsum,
                      UT transform / WY representation, Neumann inverse of
                      (I - tril(A))). Pure torch: the same math the Triton
                      kernel implements, without Triton.
  --impl chunk        chunk_kda -- the Triton kernel (chunk_fwd.py /
                      chunk_bwd.py). The only path that requires CUDA.

Only `chunk` needs an NVIDIA GPU. The other two run on cpu, mps, or cuda, and
`--device` defaults to the best available. That is what makes both halves of
this harness checkable on a laptop, which the previous FlashKDA target could
not offer -- its torch reference is a bit-exact CUDA emulation and needs a
device.

The two torch paths play different roles. `naive` validates the conversions
below against the arbiter before any kernel runs -- the same role `--impl
xla` plays on the TPU side. `chunk_torch` is a stage-2 stand-in: it measures
the *chunked algorithm's* error rather than the recurrence's, so `tpu vs
gpuchunk` is a real cross-implementation check with no CUDA in it, and where
the Triton kernel is available `gpu vs gpuchunk` separates implementation
error from algorithm error.

Neither torch entry point applies the activations or handles varlen: both
take `g` already in log space, `q`/`k` already l2-normalized, and `beta`
already post-sigmoid. So this file spells out l2norm + gate activation
explicitly and loops over segments. Those torch ops are differentiable, so
both paths yield gradients by ordinary autograd.

Usage:
  python run_gpu.py --case all --impl naive       --backward   # any device
  python run_gpu.py --case all --impl chunk_torch --backward   # any device
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


def _fla_naive():
  """Import FLA's pure-torch module, Triton or no Triton.

  `import fla.ops.kda.naive` runs `fla/ops/__init__.py`, which eagerly
  imports every op in the library, and all of them `import triton` -- which
  has no macOS or CPU-only wheel. `naive.py` itself needs only torch and
  einops, so on those hosts load it straight from the source tree, bypassing
  the package __init__ chain. Same file either way.
  """
  try:
    import fla.ops.kda.naive as mod
    return mod
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
      return mod
  raise SystemExit(
      f"could not import FLA ({first}), and fla/ops/kda/naive.py was not "
      f"found. Set FLA_ROOT to a checkout of "
      f"fla-org/flash-linear-attention.")


def _per_sequence(case, t: dict, fn):
  """Apply a torch KDA function one sequence at a time, in canonical layout.

  Neither `naive_recurrent_kda` nor `naive_chunk_kda` has a varlen path, and
  both index their state by batch, so each sequence runs as its own B=1 call
  and the results are reassembled. This is also where the pre-activation FLA
  expects is applied: both take `g` already in log space, `q`/`k` already
  l2-normalized, and `beta` already post-sigmoid.
  """
  qn, kn = _l2norm(t["q"].float()), _l2norm(t["k"].float())
  gg = _activate_gate(t["g"].float(), t["a_log"], t["dt_bias"],
                      case.lower_bound)
  h0 = t["initial_state"]
  scale = float(case.head_dim ** -0.5)

  cu = case.cu_seqlens
  outs, finals = [], []
  for n in range(len(cu) - 1):
    lo, hi = int(cu[n]), int(cu[n + 1])
    bi = 0 if case.is_varlen else n
    t0, t1 = (lo, hi) if case.is_varlen else (0, hi - lo)
    sl = lambda x: x[bi:bi + 1, t0:t1]
    o, s = fn(sl(qn), sl(kn), sl(t["v"].float()), sl(gg),
              sl(t["beta"].float()), scale=scale,
              initial_state=None if h0 is None else h0[n:n + 1],
              output_final_state=True)
    outs.append(o)
    finals.append(s)

  out = (torch.cat(outs, 1) if case.is_varlen else torch.cat(outs, 0))
  return out, torch.cat(finals, 0)


def call_naive(case, t: dict):
  """FLA's token-by-token recurrence. Pure torch, any device."""
  return _per_sequence(case, t, _fla_naive().naive_recurrent_kda)


# exp() overflows fp32 above this. numpy, not torch, so it is a plain float.
_FP32_EXP_MAX = float(np.log(np.finfo(np.float32).max))  # 88.72


def _check_gate_overflow(g, chunk_size: int) -> None:
  """Reject inputs on which `naive_chunk_kda`'s backward returns NaN.

  `naive_chunk_kda` forms its attention matrix as `(g - g_i).exp()` over the
  *full* BTxBT block (naive.py, the `for i in range(BT)` loop), where `g` is
  the within-chunk gate cumsum, and only masks the upper triangle
  *afterwards*. For c > i that exponent is positive and as large as the
  within-chunk cumsum span, so it overflows fp32 to +inf. The forward
  survives, because `masked_fill` discards exactly those entries; the
  backward does not, because masked_fill's gradient is 0 there and
  `0 * inf = NaN`. The NaN then floods dq, dk, dg, dbeta, da_log and ddt_bias
  while dv and dinitial_state -- which do not flow through that exp -- stay
  clean. That signature is what this harness first observed.

  The span is bounded by `chunk_size * |lower_bound|`, so at lower_bound=-2
  overflow is structurally impossible at BT=32 (64 < 88.7) and reachable at
  BT=64 (128 > 88.7). Rather than trust that bound, measure the real span:
  it is exact, and it stays correct for any other lower_bound.

  This is a defect in FLA's torch reference, not in its Triton kernel, which
  computes the masked region differently. Reporting it beats writing NaN
  artifacts and calling the case FAIL.
  """
  b, tt, h, k = g.shape
  with torch.no_grad():  # a diagnostic, not part of the computation
    cum = g.detach().reshape(b, tt // chunk_size, chunk_size, h, k).cumsum(2)
    span = float((cum.amax(2) - cum.amin(2)).max())
  if span > _FP32_EXP_MAX:
    raise NotImplementedError(
        f"naive_chunk_kda's backward is NaN here: the within-chunk gate "
        f"cumsum span reaches {span:.1f} at chunk_size={chunk_size}, and "
        f"exp() overflows fp32 above {_FP32_EXP_MAX:.1f}. The forward masks "
        f"the overflowing entries away; the backward multiplies them by a "
        f"zero cotangent and gets 0*inf. Halve --chunk-size: the span scales "
        f"with it, so {chunk_size // 2} gives ~{span / 2:.0f}.")


def call_chunk_torch(case, t: dict, chunk_size: int = 32):
  """FLA's chunked algorithm in pure torch. Any device -- no Triton, no CUDA.

  This is the same algorithm `chunk_kda` implements -- gate cumsum, the UT
  transform / WY representation, the Neumann inverse of (I - tril(A)) -- just
  written in torch instead of Triton. So `gpuchunk vs arbiter` measures the
  *chunked algorithm's* own error, and `gpu vs gpuchunk` isolates what the
  Triton implementation adds on top of it. That separation is what FlashKDA's
  torch_ref used to provide, except this one runs without a GPU.

  Defaults to chunk_size=32 rather than the 64 the two kernels use. Chunk size
  is internal blocking -- the chunked form is mathematically equivalent to the
  recurrence at any of them, and this is a torch stand-in, not the kernel
  whose tiling is under test -- so matching 64 buys nothing here, while 32 is
  the largest value at which `_check_gate_overflow` cannot fire.

  Two honest limits. `naive_chunk_kda` asserts `T % chunk_size == 0`, so a
  segment of unaligned length cannot run (see `varlen_unaligned`); and it
  casts every input `.to(torch.float)` on entry, so it cannot reproduce bf16
  rounding -- it measures algorithm error, not precision error, whatever
  `--dtype` says.
  """
  for n, ln in enumerate(case.seq_lens if case.is_varlen
                         else (case.seq_len,) * case.batch):
    if ln % chunk_size:
      raise NotImplementedError(
          f"naive_chunk_kda requires T % chunk_size == 0; sequence {n} has "
          f"length {ln}, chunk_size={chunk_size}. Use --impl naive for this "
          f"case.")

  def fn(q, k, v, g, beta, **kw):
    # g arrives here already sliced to one sequence and activated into log
    # space, which is exactly what naive_chunk_kda will cumsum.
    _check_gate_overflow(g, chunk_size)
    return _fla_naive().naive_chunk_kda(q, k, v, g, beta,
                                        chunk_size=chunk_size, **kw)

  return _per_sequence(case, t, fn)


_IMPLS = {"chunk": call_chunk, "naive": call_naive,
          "chunk_torch": call_chunk_torch}
# Implementations whose blocking --chunk-size may override.
_CHUNKED = ("chunk_torch",)
# Artifact prefix per implementation.
_TAGS = {"chunk": "gpu", "naive": "gpuref", "chunk_torch": "gpuchunk"}


def run(case, t: dict, inp: dict, impl: str, backward: bool,
        chunk_size: int | None = None) -> dict:
  kw = {} if chunk_size is None or impl not in _CHUNKED else {
      "chunk_size": chunk_size}
  o, ht = _IMPLS[impl](case, t, **kw)
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


def run_case(name, impl, dtype, device, indir, outdir, backward,
             chunk_size=None) -> None:
  case = kda_case.CASES[name]
  inp = kda_case.load_inputs(os.path.join(indir, f"in_{name}.npz"))
  res = run(case, to_torch(case, inp, dtype, device), inp, impl, backward,
            chunk_size)
  tag = _TAGS[impl]
  np.savez(os.path.join(outdir, f"{tag}_{name}.npz"), **res)
  bt = f" chunk_size={chunk_size}" if impl in _CHUNKED else ""
  print(f"{tag}_{name}: {' '.join(sorted(res))}  "
        f"dtype={dtype} impl={impl} device={device}{bt}")


def _pick_device(impl: str, requested: str | None) -> str:
  """Resolve --device. Only the Triton kernel is pinned to CUDA."""
  if requested:
    return requested
  if impl == "chunk":
    return "cuda"
  # The torch implementations run anywhere. Prefer a real accelerator when
  # one exists -- on this laptop that is MPS -- and fall back to CPU.
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--impl", default="naive", choices=list(_IMPLS))
  p.add_argument("--dtype", default="float32",
                 choices=["bfloat16", "float16", "float32"])
  p.add_argument("--device", default=None,
                 help="cpu | mps | cuda. Default: cuda for --impl chunk; "
                      "best available accelerator otherwise.")
  p.add_argument("--backward", action="store_true")
  p.add_argument("--chunk-size", type=int, default=32, choices=[16, 32, 64],
                 help="Blocking for --impl chunk_torch only. Default 32: at "
                      "64 the gate cumsum span overflows fp32 inside "
                      "naive_chunk_kda and its backward returns NaN.")
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  args = p.parse_args()

  device = _pick_device(args.impl, args.device)
  if args.impl == "chunk" and not (
      str(device).startswith("cuda") and torch.cuda.is_available()):
    # FLA's KDA kernels are Triton and there is no CPU or MPS fallback. Say
    # so here rather than surfacing "Torch not compiled with CUDA enabled"
    # from somewhere inside tensor construction.
    raise SystemExit(
        "--impl chunk requires CUDA (fla.ops.kda is Triton-only).\n"
        "  --impl chunk_torch  same chunked algorithm, pure torch, any "
        "device\n"
        "  --impl naive        token-by-token recurrence, pure torch")

  os.makedirs(args.outdir, exist_ok=True)
  names = list(kda_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    try:
      run_case(name, args.impl, args.dtype, device, args.indir, args.outdir,
               args.backward, args.chunk_size)
    except NotImplementedError as e:
      # Something the implementation cannot express -- an unaligned length, or
      # a gate span that overflows its exp -- is a real result, not a harness
      # failure. Report it and keep going, as run_tpu.py does for Mosaic's
      # tiling limits. compare.py then reports the case as SKIP rather than
      # silently passing it.
      print(f"SKIP {name}: {e}")


if __name__ == "__main__":
  main()
