# Copyright 2026. Apache-2.0.
"""Compare the KDA artifacts produced by run_tpu.py and run_gpu.py.

Runs anywhere -- it only reads .npz files.

Unlike the GDN harness, which had one fp32 reference per side, KDA has a
single shared arbiter: the float64 recurrence in kda_case.reference(). Every
backend is scored against it, so a disagreement is attributable rather than
merely observed.

  ref_      arbiter, float64 token-by-token recurrence (kda_case.py)
  tpuref_   tokamax `implementation="xla"`, fp32
  tpu_      tokamax `implementation="mosaic"`, the Pallas TPU kernel
  gpuref_   FlashKDA tests/torch_ref.py -- a bit-exact bf16/fp16 emulation
            of the CUDA kernel, NOT an fp32 reference
  gpu_      FlashKDA CUDA kernel

Stages:
  --stage semantics   tpuref vs ref. Pure fp32-vs-fp64, no kernels. If this
                      fails, the conversions in run_tpu.py are wrong and
                      nothing downstream means anything. Runs on CPU.
  --stage kernels     everything against the arbiter, plus tpu vs gpu.

Note the asymmetry in `gpuref vs ref`: torch_ref accumulates its chunk
inverse in fp16 (CUBLAS_COMPUTE_16F) by design, so it is expected to sit at
bf16-ish error against the arbiter, not fp32 error. `gpu vs gpuref` is the
tight one on the GPU side -- those two should agree closely.

Usage:
  python compare.py --stage semantics --dir artifacts
  python compare.py --stage kernels   --dir artifacts
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import kda_case

TENSORS = ("output", "final_state")

# bf16 has ~3 decimal digits; the recurrence is sequential so error grows with
# T. These are the thresholds the GDN harness used and are appropriate for a
# chunked bf16 linear-attention kernel.
RTOL, ATOL = 2e-2, 2e-2
# fp32 vs fp64 on identical semantics.
SEMANTIC_TOL = 1e-4


def stats(a: np.ndarray, b: np.ndarray) -> dict:
  a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
  d = np.abs(a - b)
  denom = np.maximum(np.abs(b), 1e-6)
  na, nb = np.linalg.norm(a), np.linalg.norm(b)
  return dict(
      max_abs=float(d.max()),
      mean_abs=float(d.mean()),
      max_rel=float((d / denom).max()),
      cos=float(a @ b / (na * nb)) if na and nb else float("nan"),
      # Every predicate here is `<=`, never `not >`: a NaN anywhere makes the
      # comparison False and the case FAIL, which is what we want. A kernel
      # that produces NaN must never be reported as passing.
      passes=bool(np.all(d <= ATOL + RTOL * np.abs(b))),
      n_nonfinite=int((~np.isfinite(a)).sum()),
  )


def _load(d: str, tag: str, name: str) -> dict | None:
  p = os.path.join(d, f"{tag}_{name}.npz")
  return dict(np.load(p)) if os.path.exists(p) else None


def _row(label: str, s: dict | None, tol: float | None = None) -> str:
  if s is None:
    return f"    {label:<16} (missing)"
  ok = s["passes"] if tol is None else s["max_abs"] <= tol
  nf = f"  NONFINITE={s['n_nonfinite']}" if s["n_nonfinite"] else ""
  return (f"    {label:<16} max|d|={s['max_abs']:.3e}  "
          f"mean|d|={s['mean_abs']:.3e}  maxrel={s['max_rel']:.3e}  "
          f"cos={s['cos']:.9f}  {'PASS' if ok else 'FAIL'}{nf}")


def _header(name: str) -> None:
  c = kda_case.CASES[name]
  shape = (f"seqs={list(c.seq_lens)}" if c.is_varlen
           else f"B={c.batch} T={c.seq_len}")
  print(f"\n=== {name}  {shape} H={c.heads} D={c.head_dim} "
        f"state={c.with_initial_state} lb={c.lower_bound}")


def compare_semantics(name: str, d: str) -> bool | None:
  """Stage 1: tokamax's own fp32 reference against the arbiter."""
  ref, tpuref = _load(d, "ref", name), _load(d, "tpuref", name)
  if ref is None or tpuref is None:
    print(f"\n=== {name}: SKIP (need ref_ and tpuref_)")
    return None
  _header(name)
  ok = True
  for t in TENSORS:
    s = stats(tpuref[t], ref[t])
    good = s["max_abs"] <= SEMANTIC_TOL
    ok &= good
    nf = f"  NONFINITE={s['n_nonfinite']}" if s["n_nonfinite"] else ""
    print(f"  {t:<14} max|d|={s['max_abs']:.3e}  cos={s['cos']:.9f}  "
          f"{'OK' if good else 'MISMATCH'}{nf}")
  return ok


def compare_kernels(name: str, d: str) -> bool | None:
  ref = _load(d, "ref", name)
  tpu, gpu = _load(d, "tpu", name), _load(d, "gpu", name)
  gpuref = _load(d, "gpuref", name)
  if tpu is None or gpu is None:
    missing = " and ".join(
        n for n, a in (("tpu_", tpu), ("gpu_", gpu)) if a is None)
    print(f"\n=== {name}: SKIP (no {missing} artifact)")
    return None
  _header(name)

  ok = True
  for t in TENSORS:
    print(f"  {t}")
    if ref is not None:
      print(_row("tpu vs arbiter", stats(tpu[t], ref[t]) if tpu else None))
      print(_row("gpu vs arbiter", stats(gpu[t], ref[t]) if gpu else None))
      if gpuref is not None:
        # Expected to be loose: torch_ref accumulates in fp16 on purpose.
        print(_row("torchref vs arb", stats(gpuref[t], ref[t])))
    if gpu is not None and gpuref is not None:
      print(_row("gpu vs torchref", stats(gpu[t], gpuref[t])))
    if tpu is not None and gpu is not None:
      s = stats(tpu[t], gpu[t])
      print(_row("tpu vs gpu", s))
      ok &= s["passes"]
  return ok


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--dir", default="artifacts")
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--stage", default="kernels",
                 choices=["kernels", "semantics"])
  args = p.parse_args()

  fn = compare_kernels if args.stage == "kernels" else compare_semantics
  names = list(kda_case.CASES) if args.case == "all" else [args.case]
  results = {n: fn(n, args.dir) for n in names}

  tol = (f"(rtol={RTOL}, atol={ATOL}, on tpu vs gpu)"
         if args.stage == "kernels" else f"(threshold {SEMANTIC_TOL})")
  print(f"\n{'=' * 64}\nsummary {tol}")
  for n, ok in results.items():
    print(f"  {n:<20} {'SKIP' if ok is None else 'PASS' if ok else 'FAIL'}")

  ran = [v for v in results.values() if v is not None]
  skipped = len(results) - len(ran)
  if skipped:
    # A run with no artifacts must not read as success. This is the whole
    # point of the harness: never report a comparison that did not happen.
    print(f"\n{skipped}/{len(results)} case(s) skipped -- "
          f"artifacts missing, nothing was compared for them")
  if not ran:
    print("NO COMPARISONS RUN")
    raise SystemExit(2)
  raise SystemExit(0 if all(ran) else 1)


if __name__ == "__main__":
  main()
