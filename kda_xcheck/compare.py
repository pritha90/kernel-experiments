# Copyright 2026. Apache-2.0.
"""Compare the KDA artifacts produced by arbiter.py, run_tpu.py, run_gpu.py.

Runs anywhere -- it only reads .npz files.

Every backend is scored against one shared float64 arbiter, so a
disagreement is attributable rather than merely observed.

  npref_    numpy fp64 forward          (kda_case.reference)
  ref_      JAX fp64 forward + VJP      (arbiter.py)          <- the arbiter
  tpuref_   tokamax implementation="xla", fp32
  tpu_      tokamax implementation="mosaic" -- the Pallas TPU kernel
  gpuref_   FLA naive_recurrent_kda, fp32 torch
  gpuchunk_ FLA naive_chunk_kda -- the chunked algorithm in pure torch
  gpu_      FLA chunk_kda -- the Triton kernel

Stages:
  --stage semantics   npref/tpuref/gpuref vs ref. No kernels, CPU only. This
                      validates the conversions in both runners, and -- via
                      npref vs ref -- the arbiter's own forward, which is
                      what licenses trusting its autodiff gradients.
  --stage kernels     tpu and the FLA side vs the arbiter, and tpu vs FLA.
                      The FLA side is gpu_ when it exists, else gpuchunk_,
                      which is the same chunked algorithm without CUDA.

Comparison metric depends on the tensor. `output`, `final_state`, and the
per-token gradients are compared elementwise. `da_log` and `ddt_bias` are
reductions over every token and channel (shapes [H] and [H*K]), so their
magnitude and their accumulated rounding error both scale with B*T -- an
elementwise atol tuned for per-token tensors is meaningless there. Those two
are scored on relative norm instead.

Usage:
  python compare.py --stage semantics --dir artifacts
  python compare.py --stage kernels   --dir artifacts
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import kda_case

GRADS = tuple(f"d{n}" for n in kda_case.DIFFERENTIABLE)
TENSORS = ("output", "final_state") + GRADS

# Gradients that are full reductions over B*T -- scored on relative norm.
REDUCED = ("da_log", "ddt_bias")

# bf16 has ~3 decimal digits and the recurrence is sequential, so error grows
# with T. Appropriate for a chunked bf16 linear-attention kernel.
RTOL, ATOL = 2e-2, 2e-2
# Relative-norm threshold for the reduced gradients at kernel precision.
REDUCED_TOL = 2e-2

# fp32 vs fp64 on identical semantics.
SEMANTIC_TOL = 1e-4
# Gradients are scored on relative norm at every stage: dq/dk/dg magnitudes
# vary by orders of magnitude across cases, so a fixed atol is not meaningful.
SEMANTIC_REL = 1e-5

# `npref vs ref` is fp64-vs-fp64 -- two independent implementations of the
# same recurrence. Scoring it at SEMANTIC_TOL would be vacuous: it has ~7
# orders of headroom there and could not fail. This is the tolerance that
# makes the arbiter self-check an actual check.
ARBITER_REL = 1e-11


def stats(a: np.ndarray, b: np.ndarray) -> dict:
  a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
  d = np.abs(a - b)
  denom = np.maximum(np.abs(b), 1e-6)
  na, nb = np.linalg.norm(a), np.linalg.norm(b)
  return dict(
      max_abs=float(d.max()),
      mean_abs=float(d.mean()),
      max_rel=float((d / denom).max()),
      rel_norm=float(np.linalg.norm(a - b) / nb) if nb else float("nan"),
      cos=float(a @ b / (na * nb)) if na and nb else float("nan"),
      # Every predicate here is `<=`, never `not >`: a NaN anywhere makes the
      # comparison False and the case FAIL, which is what we want. A kernel
      # that produces NaN must never be reported as passing.
      passes=bool(np.all(d <= ATOL + RTOL * np.abs(b))),
      n_nonfinite=int((~np.isfinite(a)).sum()),
  )


def _ok(tensor: str, s: dict, stage: str) -> bool:
  """Pass predicate for one tensor. All comparisons are `<=`, so NaN fails."""
  if stage == "arbiter":
    return s["rel_norm"] <= ARBITER_REL
  if stage == "semantics":
    if tensor in GRADS:
      return s["rel_norm"] <= SEMANTIC_REL
    return s["max_abs"] <= SEMANTIC_TOL
  if tensor in REDUCED:
    return s["rel_norm"] <= REDUCED_TOL
  return s["passes"]


def _load(d: str, tag: str, name: str) -> dict | None:
  p = os.path.join(d, f"{tag}_{name}.npz")
  return dict(np.load(p)) if os.path.exists(p) else None


def _row(label: str, s: dict, ok: bool) -> str:
  nf = f"  NONFINITE={s['n_nonfinite']}" if s["n_nonfinite"] else ""
  return (f"    {label:<18} max|d|={s['max_abs']:.3e}  "
          f"relnorm={s['rel_norm']:.3e}  cos={s['cos']:.9f}  "
          f"{'PASS' if ok else 'FAIL'}{nf}")


def _header(name: str) -> None:
  c = kda_case.CASES[name]
  shape = (f"seqs={list(c.seq_lens)}" if c.is_varlen
           else f"B={c.batch} T={c.seq_len}")
  print(f"\n=== {name}  {shape} H={c.heads} D={c.head_dim} "
        f"state={c.with_initial_state} lb={c.lower_bound}")


def _shared(ref: dict, other: dict) -> list[str]:
  """Tensors present on both sides. A backward-less run simply has fewer."""
  return [t for t in TENSORS if t in ref and t in other]


def compare_semantics(name: str, d: str) -> bool | None:
  """Stage 1: every fp32/fp64 reference against the arbiter, on CPU.

  Three comparisons, all optional but each meaningful on its own:
    npref  vs ref -> the arbiter's forward is right (two independent fp64
                     implementations of the recurrence)
    tpuref vs ref -> run_tpu.py's conversions are right
    gpuref vs ref -> run_gpu.py's conversions are right
  """
  ref = _load(d, "ref", name)
  if ref is None:
    print(f"\n=== {name}: SKIP (no ref_ arbiter; run arbiter.py)")
    return None
  sides = [(tag, _load(d, tag, name))
           for tag in ("npref", "tpuref", "gpuref")]
  sides = [(t, v) for t, v in sides if v is not None]
  if not sides:
    print(f"\n=== {name}: SKIP (no npref_/tpuref_/gpuref_ artifact)")
    return None

  _header(name)
  ok = True
  for tag, side in sides:
    # npref is the other fp64 forward, so it is held to a far tighter bar
    # than the fp32 backends.
    stage = "arbiter" if tag == "npref" else "semantics"
    for t in _shared(ref, side):
      s = stats(side[t], ref[t])
      good = _ok(t, s, stage)
      ok &= good
      print(f"  {t:<16} {_row(f'{tag} vs arbiter', s, good).strip()}")
  return ok


def compare_kernels(name: str, d: str) -> bool | None:
  """Stage 2. Scores whatever kernel artifacts exist.

  The FLA side can be either of two things, and the report says which:

    gpu_       chunk_kda, the Triton kernel. Needs CUDA.
    gpuchunk_  naive_chunk_kda, the SAME chunked algorithm written in pure
               torch. Runs on cpu/mps/cuda.

  `gpuchunk_` is the fallback when there is no NVIDIA GPU. `tpu vs gpuchunk`
  is still a genuine cross-implementation check -- Pallas/Mosaic against
  FLA's chunked math, on shared inputs, scored by a common arbiter -- it just
  does not exercise the Triton implementation. That distinction is printed,
  not left to the reader.

  With a TPU artifact and some FLA artifact, the cross row is the verdict.
  With only one side, that side is scored against the arbiter instead, which
  is weaker but genuine, and is labelled as such.
  """
  ref = _load(d, "ref", name)
  tpu = _load(d, "tpu", name)
  gpu, gpuchunk = _load(d, "gpu", name), _load(d, "gpuchunk", name)
  gpuref = _load(d, "gpuref", name)

  # Prefer the real kernel; fall back to the torch chunked algorithm.
  fla, fla_tag = ((gpu, "gpu") if gpu is not None else
                  (gpuchunk, "gpuchunk") if gpuchunk is not None else
                  (None, None))
  if tpu is None and fla is None:
    print(f"\n=== {name}: SKIP (no tpu_, gpu_ or gpuchunk_ artifact)")
    return None
  if ref is None:
    print(f"\n=== {name}: SKIP (no ref_ arbiter; run arbiter.py)")
    return None

  both = tpu is not None and fla is not None
  _header(name)
  if fla_tag == "gpuchunk":
    print("  [FLA side is chunk_torch (pure torch), NOT the Triton kernel: "
          "this checks the chunked algorithm, not the CUDA implementation]")
  if not both:
    which = "tpu" if tpu is not None else fla_tag
    print(f"  [one-sided: only {which}_ present -> scored against the "
          f"arbiter, NOT a cross-implementation check]")

  ok = True
  # Union, not intersection: if one side ran with --backward and the other
  # did not, the gradients it did produce should still be scored against the
  # arbiter rather than vanishing from the report.
  have = set().union(*(s for s in (tpu, fla) if s is not None))
  for t in [t for t in TENSORS if t in ref and t in have]:
    print(f"  {t}")
    for tag, side in (("tpu", tpu), (fla_tag, fla)):
      if side is not None and t in side:
        s = stats(side[t], ref[t])
        good = _ok(t, s, "kernels")
        print(_row(f"{tag} vs arbiter", s, good))
        # One-sided runs have no cross row, so the arbiter rows are the
        # verdict; otherwise they are diagnostic only.
        if not both:
          ok &= good
    # Diagnostics. Neither votes.
    if gpuchunk is not None and fla_tag != "gpuchunk" and t in gpuchunk:
      # With the Triton kernel present, gpuchunk localizes a GPU-side
      # failure to the implementation rather than the algorithm.
      s = stats(gpuchunk[t], ref[t])
      print(_row("fla-chunk vs arb", s, _ok(t, s, "kernels")))
    if gpuref is not None and t in gpuref:
      s = stats(gpuref[t], ref[t])
      print(_row("fla-naive vs arb", s, _ok(t, s, "semantics")))
    if both and t in tpu and t in fla:
      s = stats(tpu[t], fla[t])
      good = _ok(t, s, "kernels")
      print(_row(f"tpu vs {fla_tag}", s, good))
      ok &= good
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

  tol = (f"(rtol={RTOL}, atol={ATOL}; reduced grads relnorm<={REDUCED_TOL})"
         if args.stage == "kernels"
         else f"(max|d|<={SEMANTIC_TOL}; grads relnorm<={SEMANTIC_REL})")
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
