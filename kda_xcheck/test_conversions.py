# Copyright 2026. Apache-2.0.
"""Negative control for the conversions on BOTH sides.

Stage 1 passing only proves that `run_tpu.to_tokamax` and `run_gpu.to_torch`
agree with the arbiter. It does not prove the comparison is *sensitive* -- a
threshold that never fires would also pass. This injects each conversion bug
the harness is meant to catch and asserts stage 1 rejects it.

Both backends are covered, forward and backward. That is new with the FLA
retarget: FlashKDA's reference needed CUDA, so only the TPU conversions could
be controlled this way. FLA's `naive_recurrent_kda` is pure PyTorch, so the
GPU-side conversions are now controllable on a laptop too.

A mutant counts as detected if ANY compared tensor -- output, final_state, or
one of the eight gradients -- moves outside its stage-1 tolerance, or if the
backend rejects the input outright.

  python kda_case.py --case all --ref
  python arbiter.py --case all
  FLA_ROOT=<fla> PYTHONPATH=<tokamax-pr1103> python test_conversions.py
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch

import compare
import kda_case
import run_gpu
import run_tpu

DIR = "artifacts"


# --------------------------------------------------------------------------
# TPU-side mutations: applied to the tokamax kwargs dict (and to a copy of
# the input dict, for the cotangents).
# --------------------------------------------------------------------------

def _t_double_sigmoid_beta(a, i):
  # The realistic bug: feeding post-activation beta to something that applies
  # sigmoid itself. Stays inside tokamax's [0,1] validation, so it is caught
  # numerically rather than by a raise -- which is what we want to test.
  a["beta"] = kda_case._sigmoid(a["beta"].astype(np.float64)).astype(np.float32)


def _t_state_on_segment_axis(a, i):
  a["initial_state"] = i["initial_state"][None]


def _t_dht_segments_reversed(a, i):
  i["dht"] = i["dht"][::-1].copy()


def _t_do_time_reversed(a, i):
  i["do"] = i["do"][:, ::-1].copy()


TPU_MUTATIONS = [
    ("fixed", "beta double-sigmoided", _t_double_sigmoid_beta),
    ("fixed", "dt_bias flattened column-major",
     lambda a, i: a.update(
         delta_time_bias=np.ascontiguousarray(
             i["dt_bias"].reshape(i["a_log"].shape[0], -1).T.reshape(-1)))),
    ("fixed", "q transposed to [B,H,T,D]",
     lambda a, i: a.update(
         query=np.ascontiguousarray(i["q"].transpose(0, 2, 1, 3)))),
    ("fixed", "gate treated as already-log-space",
     lambda a, i: a.update(use_gate_in_kernel=False)),
    ("fixed", "l2norm disabled",
     lambda a, i: a.update(use_qk_l2norm=False)),
    ("fixed", "lower_bound dropped (softplus branch)",
     lambda a, i: a.update(lower_bound=None)),
    ("fixed", "output cotangent time-reversed", _t_do_time_reversed),
    ("fixed_state_b2", "state on segment axis, not batch axis",
     _t_state_on_segment_axis),
    ("varlen", "segment_ids 0-indexed",
     lambda a, i: a.update(segment_ids=np.maximum(a["segment_ids"] - 1, 0))),
    ("varlen", "segment order reversed",
     lambda a, i: a.update(segment_ids=a["segment_ids"][:, ::-1].copy())),
    ("varlen_state", "state segments swapped",
     lambda a, i: a.update(
         initial_state=i["initial_state"][::-1].copy()[None])),
    ("varlen_state", "state cotangent segments swapped",
     _t_dht_segments_reversed),
]


# --------------------------------------------------------------------------
# GPU-side mutations: applied to the torch tensor dict. The two that change
# behaviour rather than data monkeypatch run_gpu's module-level helpers,
# which keeps run_gpu.py free of test hooks.
# --------------------------------------------------------------------------

def _rg(t, name, fn):
  """Replace a leaf tensor, preserving requires_grad."""
  t[name] = fn(t[name].detach()).requires_grad_(True)


def _g_double_sigmoid_beta(t, i):
  _rg(t, "beta", torch.sigmoid)


def _g_dt_bias_col_major(t, i):
  h = t["a_log"].shape[0]
  _rg(t, "dt_bias", lambda x: x.view(h, -1).T.reshape(-1).contiguous())


def _g_state_segments_swapped(t, i):
  _rg(t, "initial_state", lambda x: x.flip(0).contiguous())


def _g_q_head_first(t, i):
  # [B,T,H,D] -> [B,H,T,D]. Only shape-legal when T == H*... ; for these
  # cases it is caught either numerically or by FLA's shape assertion.
  _rg(t, "q", lambda x: x.transpose(1, 2).contiguous())


def _g_no_l2norm(t, i):
  return {"_l2norm": lambda x, eps=1e-6: x}


def _g_gate_not_activated(t, i):
  return {"_activate_gate": lambda g, a_log, dt_bias, lb: g}


def _g_softplus_branch(t, i):
  # tokamax's / FLA's other gate branch. Both libraries have it; it is
  # excluded from the cross-check, so this confirms picking it by mistake
  # would not go unnoticed.
  return {"_activate_gate": lambda g, a_log, dt_bias, lb: (
      -torch.exp(a_log)[None, None, :, None]
      * torch.nn.functional.softplus(g + dt_bias.view(g.shape[2], -1)))}


def _g_do_time_reversed(t, i):
  i["do"] = i["do"][:, ::-1].copy()


GPU_MUTATIONS = [
    ("fixed", "beta double-sigmoided", _g_double_sigmoid_beta),
    ("fixed", "dt_bias reshaped column-major", _g_dt_bias_col_major),
    ("fixed", "q transposed to [B,H,T,D]", _g_q_head_first),
    ("fixed", "gate treated as already-log-space", _g_gate_not_activated),
    ("fixed", "l2norm skipped", _g_no_l2norm),
    ("fixed", "softplus gate branch instead of sigmoid", _g_softplus_branch),
    ("fixed", "output cotangent time-reversed", _g_do_time_reversed),
    ("fixed_state_b2", "state segments swapped", _g_state_segments_swapped),
    ("varlen_state", "state segments swapped", _g_state_segments_swapped),
]


# --------------------------------------------------------------------------

def _worst(res: dict, ref: dict) -> tuple[str, float]:
  """Worst stage-1 offender across every tensor both sides produced.

  Returns the metric that the stage-1 predicate actually uses for that
  tensor, so "detected" here means exactly "compare.py --stage semantics
  would have failed".
  """
  worst, tag = 0.0, "-"
  for t in compare.TENSORS:
    if t not in res or t not in ref:
      continue
    s = compare.stats(res[t], ref[t])
    # Normalize each tensor's metric by its own threshold so they are
    # comparable; > 1.0 means that tensor failed.
    if t in compare.GRADS:
      score = s["rel_norm"] / compare.SEMANTIC_REL
    else:
      score = s["max_abs"] / compare.SEMANTIC_TOL
    # NaN must win: `score > worst` is False for NaN, so test it explicitly.
    if not (score <= worst):
      worst, tag = score, t
  return tag, worst


def _run_tpu(case, inp, args) -> dict:
  return run_tpu.invoke_vjp(case, args, inp, "xla", "float32")


def _run_gpu(case, inp, t, patches) -> dict:
  saved = {k: getattr(run_gpu, k) for k in patches}
  for k, v in patches.items():
    setattr(run_gpu, k, v)
  try:
    return run_gpu.run(case, t, inp, "naive", backward=True)
  finally:
    for k, v in saved.items():
      setattr(run_gpu, k, v)


def main() -> int:
  failures = []

  # ---- Controls: unmutated conversions must agree with the arbiter.
  for name, case in kda_case.CASES.items():
    ref = dict(np.load(os.path.join(DIR, f"ref_{name}.npz")))
    inp = kda_case.load_inputs(os.path.join(DIR, f"in_{name}.npz"))
    for side, res in (
        ("tpu", _run_tpu(case, dict(inp), run_tpu.to_tokamax(case, inp))),
        ("gpu", _run_gpu(case, dict(inp),
                         run_gpu.to_torch(case, inp, "float32", "cpu"), {})),
    ):
      tag, score = _worst(res, ref)
      ok = score <= 1.0
      print(f"control {side}  {name:<20} worst={tag}@{score:.2f}x  "
            f"{'OK' if ok else 'BROKEN'}")
      if not ok:
        failures.append(f"control {side} {name}")

  # ---- TPU mutants.
  print()
  for cname, label, mutate in TPU_MUTATIONS:
    case = kda_case.CASES[cname]
    ref = dict(np.load(os.path.join(DIR, f"ref_{cname}.npz")))
    inp = kda_case.load_inputs(os.path.join(DIR, f"in_{cname}.npz"))
    mut_inp = copy.deepcopy(inp)
    args = run_tpu.to_tokamax(case, inp)
    mutate(args, mut_inp)
    try:
      tag, score = _worst(_run_tpu(case, mut_inp, args), ref)
      # NOT (score <= 1), never (score > 1): a mutant that diverges to NaN
      # must count as detected, and every NaN comparison is False.
      detected = not (score <= 1.0)
      note = f"{tag}@{score:.1f}x"
    except Exception as e:  # a rejected shape/dtype/range is also detection
      detected, note = True, f"raised {type(e).__name__}"
    print(f"mutant tpu  {cname:<16} {label:<42} {note:<22} "
          f"{'detected' if detected else '*** MISSED ***'}")
    if not detected:
      failures.append(f"tpu mutant {cname}: {label}")

  # ---- GPU mutants.
  print()
  for cname, label, mutate in GPU_MUTATIONS:
    case = kda_case.CASES[cname]
    ref = dict(np.load(os.path.join(DIR, f"ref_{cname}.npz")))
    inp = kda_case.load_inputs(os.path.join(DIR, f"in_{cname}.npz"))
    mut_inp = copy.deepcopy(inp)
    t = run_gpu.to_torch(case, inp, "float32", "cpu")
    try:
      patches = mutate(t, mut_inp) or {}
      tag, score = _worst(_run_gpu(case, mut_inp, t, patches), ref)
      detected = not (score <= 1.0)
      note = f"{tag}@{score:.1f}x"
    except Exception as e:
      detected, note = True, f"raised {type(e).__name__}"
    print(f"mutant gpu  {cname:<16} {label:<42} {note:<22} "
          f"{'detected' if detected else '*** MISSED ***'}")
    if not detected:
      failures.append(f"gpu mutant {cname}: {label}")

  print()
  if failures:
    print(f"FAIL: {len(failures)} problem(s):")
    for f in failures:
      print(f"  {f}")
    return 1
  print(f"PASS: controls agree on both sides; all "
        f"{len(TPU_MUTATIONS) + len(GPU_MUTATIONS)} injected bugs detected")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
