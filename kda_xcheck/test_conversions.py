# Copyright 2026. Apache-2.0.
"""Negative control for the TPU-side conversions.

Stage 1 passing only proves the conversions in `run_tpu.to_tokamax` agree
with the arbiter. It does not prove the comparison is *sensitive* -- a
threshold that never fires would also pass. This injects each conversion bug
the harness is meant to catch and asserts stage 1 rejects it.

Runs on CPU (implementation="xla"). No accelerator needed.

  python kda_case.py --case all --ref     # first, to write artifacts/
  PYTHONPATH=<tokamax-pr1103> python test_conversions.py
"""

from __future__ import annotations

import os

import numpy as np

import compare
import kda_case
import run_tpu

TOL = compare.SEMANTIC_TOL


def _swap_state_batch_segment(args, inp):
  """Put the fixed-length state on the segment axis instead of the batch."""
  args["initial_state"] = inp["initial_state"][None]


# (case, description, mutation applied to the tokamax kwargs)
MUTATIONS = [
    ("fixed", "beta passed as raw logits",
     lambda a, i: a.update(
         beta=np.ascontiguousarray(i["beta_logits"].transpose(2, 0, 1)))),
    ("fixed", "dt_bias flattened column-major",
     lambda a, i: a.update(
         delta_time_bias=np.ascontiguousarray(i["dt_bias"].T.reshape(-1)))),
    ("fixed", "q/k/v transposed to [B,H,T,D]",
     lambda a, i: a.update(
         query=np.ascontiguousarray(i["q"].transpose(0, 2, 1, 3)))),
    ("fixed", "gate treated as already-log-space",
     lambda a, i: a.update(use_gate_in_kernel=False)),
    ("fixed", "l2norm disabled",
     lambda a, i: a.update(use_qk_l2norm=False)),
    ("fixed", "lower_bound dropped (softplus branch)",
     lambda a, i: a.update(lower_bound=None)),
    ("fixed_state_b2", "state on segment axis, not batch axis",
     _swap_state_batch_segment),
    ("varlen", "segment_ids 0-indexed",
     lambda a, i: a.update(segment_ids=np.maximum(a["segment_ids"] - 1, 0))),
    ("varlen", "segment order reversed",
     lambda a, i: a.update(segment_ids=a["segment_ids"][:, ::-1].copy())),
    ("varlen_state", "state segments swapped",
     lambda a, i: a.update(
         initial_state=i["initial_state"][::-1].copy()[None])),
]


def _error(case, args) -> float:
  ref = dict(np.load(os.path.join("artifacts", f"ref_{case.name}.npz")))
  out, st = run_tpu.invoke(case, args, "xla", "float32")
  return max(compare.stats(out, ref["output"])["max_abs"],
             compare.stats(st, ref["final_state"])["max_abs"])


def main() -> int:
  failures = []

  # Controls: unmutated conversions must agree with the arbiter.
  for name in kda_case.CASES:
    case = kda_case.CASES[name]
    inp = kda_case.load_inputs(os.path.join("artifacts", f"in_{name}.npz"))
    err = _error(case, run_tpu.to_tokamax(case, inp))
    ok = err <= TOL
    print(f"control  {name:<20} err={err:.2e}  {'OK' if ok else 'BROKEN'}")
    if not ok:
      failures.append(f"control {name}")

  print()
  for cname, label, mutate in MUTATIONS:
    case = kda_case.CASES[cname]
    inp = kda_case.load_inputs(os.path.join("artifacts", f"in_{cname}.npz"))
    args = run_tpu.to_tokamax(case, inp)
    mutate(args, inp)
    try:
      err = _error(case, args)
      # NOT (err <= TOL), never (err > TOL): a mutant that diverges to NaN
      # must count as detected, and every NaN comparison is False.
      detected = not (err <= TOL)
      note = f"err={err:.2e}"
    except Exception as e:  # a rejected shape/dtype is also detection
      detected, note = True, f"raised {type(e).__name__}"
    print(f"mutant   {cname:<16} {label:<40} {note:<24} "
          f"{'detected' if detected else '*** MISSED ***'}")
    if not detected:
      failures.append(f"mutant {cname}: {label}")

  print()
  if failures:
    print(f"FAIL: {len(failures)} problem(s):")
    for f in failures:
      print(f"  {f}")
    return 1
  print("PASS: controls agree, every injected conversion bug is detected")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
