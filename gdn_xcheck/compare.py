# Copyright 2026. Apache-2.0.
"""Compare the TPU (tokamax) and GPU (FlashQLA) artifacts.

Runs on either machine — it only reads .npz files.

Three comparisons per tensor, which together attribute any disagreement:

  ref  vs ref   both fp32 scans. Should be ~1e-6. If this fails, the *torch
                preprocessing in run_gpu.py* does not match tokamax semantics
                and nothing downstream is meaningful.
  tpu  vs ref   tokamax Pallas kernel error against its own fp32 scan.
  gpu  vs ref   FlashQLA error against its own fp32 scan.
  tpu  vs gpu   the actual cross-backend number.

Usage:
  python compare.py --dir artifacts
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import gdn_case

TENSORS = ("output", "recurrent_state", "conv_state")
RTOL, ATOL = 2e-2, 2e-2


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
      passes=bool(np.all(d <= ATOL + RTOL * np.abs(b))),
  )


def _row(label: str, s: dict | None) -> str:
  if s is None:
    return f"    {label:<14} (missing)"
  flag = "PASS" if s["passes"] else "FAIL"
  return (f"    {label:<14} max|d|={s['max_abs']:.3e}  "
          f"mean|d|={s['mean_abs']:.3e}  maxrel={s['max_rel']:.3e}  "
          f"cos={s['cos']:.9f}  {flag}")


def compare_semantics(name: str, d: str) -> bool:
  """Stage 1: do the two fp32 references agree? No kernels involved.

  Both sides run in fp32, so anything above ~1e-5 is a genuine semantic
  mismatch in run_gpu.py's torch replication, not numerics.
  """
  tp = os.path.join(d, f"tpuref_{name}.npz")
  gp = os.path.join(d, f"gpuref_{name}.npz")
  if not (os.path.exists(tp) and os.path.exists(gp)):
    print(f"\n=== {name}: SKIP (no ref artifacts)")
    return True
  tpu, gpu = dict(np.load(tp)), dict(np.load(gp))
  print(f"\n=== {name} (semantics)")
  ok = True
  for t in TENSORS:
    s = stats(gpu[f"ref_{t}"], tpu[f"ref_{t}"])
    exact = s["max_abs"] <= 1e-4
    ok &= exact
    print(f"  {t:<18} max|d|={s['max_abs']:.3e}  "
          f"cos={s['cos']:.9f}  {'OK' if exact else 'MISMATCH'}")
  return ok


def compare_case(name: str, d: str) -> bool:
  tpu_p, gpu_p = os.path.join(d, f"tpu_{name}.npz"), os.path.join(d, f"gpu_{name}.npz")
  if not (os.path.exists(tpu_p) and os.path.exists(gpu_p)):
    print(f"\n=== {name}: SKIP (missing "
          f"{'tpu' if not os.path.exists(tpu_p) else 'gpu'} artifact)")
    return True

  tpu, gpu = dict(np.load(tpu_p)), dict(np.load(gpu_p))
  case = gdn_case.CASES[name]
  print(f"\n=== {name}  T={case.num_tokens} seqs={case.num_seqs} "
        f"n_kq={case.n_kq} n_v={case.n_v} d={case.d_k}")

  ok = True
  for t in TENSORS:
    print(f"  {t}")
    rt, rg = f"ref_{t}", f"ref_{t}"
    have_rt, have_rg = rt in tpu, rg in gpu

    if have_rt and have_rg:
      s = stats(gpu[rg], tpu[rt])
      print(_row("ref vs ref", s))
      if not s["passes"]:
        print("      ^ preprocessing mismatch: fix run_gpu.py before reading "
              "the rows below")
    if have_rt:
      print(_row("tpu vs ref", stats(tpu[t], tpu[rt])))
    if have_rg:
      print(_row("gpu vs ref", stats(gpu[t], gpu[rg])))

    s = stats(gpu[t], tpu[t])
    print(_row("tpu vs gpu", s))
    ok &= s["passes"]
  return ok


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--dir", default="artifacts")
  p.add_argument("--case", default="all", choices=["all", *gdn_case.CASES])
  p.add_argument("--stage", default="kernels", choices=["kernels", "semantics"],
                 help="'semantics' compares the two fp32 references "
                      "(tpuref_/gpuref_, no accelerator needed); 'kernels' "
                      "compares the real kernels (tpu_/gpu_)")
  args = p.parse_args()

  fn = compare_case if args.stage == "kernels" else compare_semantics
  names = list(gdn_case.CASES) if args.case == "all" else [args.case]
  results = {n: fn(n, args.dir) for n in names}

  tol = (f"(rtol={RTOL}, atol={ATOL})" if args.stage == "kernels"
         else "(fp32 vs fp32, threshold 1e-4)")
  print(f"\n{'=' * 60}\nsummary {tol}")
  for n, ok in results.items():
    print(f"  {n:<18} {'PASS' if ok else 'FAIL'}")
  raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
  main()
