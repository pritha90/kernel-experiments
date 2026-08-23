# Copyright 2026. Apache-2.0.
"""Shared cases, input generation, and the fp32 arbiter for the KDA
TPU(tokamax) vs GPU(FlashKDA) cross-check.

Canonical layout here is `[B, T, H, D]` with a K-major state `[N, H, K, V]`.
Both runners convert into their own convention; neither backend's layout is
privileged. Nothing in this file imports jax or torch.

KDA (Kimi Delta Attention) differs from Gated DeltaNet in the gate: the decay
is **per key channel**, so `g` is `[B, T, H, K]` and the state decays as a row
scaling `diag(exp(g_t)) S_{t-1}` rather than by a scalar.

Semantics, from tokamax reference.py:96-170 and FlashKDA tests/torch_ref.py:

  g     = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias))   [B,T,H,K]
  q, k  = l2norm(., eps=1e-6);  q *= scale
  beta  = sigmoid(beta_logits)                                    [B,T,H]
  S_t   = diag(exp(g_t)) S_{t-1}
  S_t  += beta_t k_t (v_t - k_t^T S_t)^T
  o_t   = q_t^T S_t
"""

from __future__ import annotations

import dataclasses
import numpy as np

# FlashKDA works in log2 space: it computes lower_bound*LOG2E*sigmoid(...) and
# exponentiates with exp2. That is the same decay as tokamax's natural-log
# formulation, so no conversion is needed at the API boundary -- but it is why
# the two disagree in the last bits, and why this file uses natural log.
LOG2E = 1.4426950408889634


@dataclasses.dataclass(frozen=True)
class Case:
  """One test configuration.

  `seq_lens` non-empty selects varlen mode: B is forced to 1, tokens are
  packed, and the runners emit `segment_ids` (tokamax) / `cu_seqlens`
  (FlashKDA).
  """

  name: str
  batch: int = 1
  seq_len: int = 256
  heads: int = 4
  # FlashKDA hard-requires K == V == 128.
  head_dim: int = 128
  seq_lens: tuple[int, ...] = ()
  with_initial_state: bool = False
  lower_bound: float = -2.0
  seed: int = 0

  @property
  def is_varlen(self) -> bool:
    return bool(self.seq_lens)

  @property
  def num_states(self) -> int:
    return len(self.seq_lens) if self.is_varlen else self.batch

  @property
  def total_tokens(self) -> int:
    return sum(self.seq_lens) if self.is_varlen else self.seq_len

  @property
  def cu_seqlens(self) -> np.ndarray:
    lens = self.seq_lens if self.is_varlen else (self.seq_len,) * self.batch
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


CASES: dict[str, Case] = {
    # Baseline. T is a multiple of 64 because tokamax's Mosaic kernel requires
    # it in fixed-length mode (pallas_mosaic_tpu.py:165).
    "fixed": Case(name="fixed", batch=2, seq_len=256, heads=4),
    # Live initial state.
    "fixed_state": Case(
        name="fixed_state", batch=1, seq_len=512, heads=4,
        with_initial_state=True,
    ),
    # B>1 *and* a live state. This is the only case where the fixed-length
    # state mapping [N=B,H,K,V] -> [B,1,H,K,V] is non-trivial: with B=1 a
    # wrong axis still lines up by accident.
    "fixed_state_b2": Case(
        name="fixed_state_b2", batch=2, seq_len=192, heads=2,
        with_initial_state=True,
    ),
    # Varlen, every segment a multiple of both chunk sizes (64 TPU / 16 GPU).
    "varlen": Case(
        name="varlen", seq_lens=(192, 64, 256), heads=4,
    ),
    # Varlen with segments aligned to neither chunk size.
    "varlen_unaligned": Case(
        name="varlen_unaligned", seq_lens=(100, 156, 57), heads=4,
    ),
    # Varlen carrying state, the chunked-prefill shape.
    "varlen_state": Case(
        name="varlen_state", seq_lens=(128, 384), heads=2,
        with_initial_state=True,
    ),
    # Long sequence: the recurrence is sequential, so error accumulates in T.
    # This is the case most likely to separate the two kernels.
    "long": Case(name="long", batch=1, seq_len=2048, heads=2),
}


def make_inputs(case: Case) -> dict[str, np.ndarray]:
  """Deterministic fp32 inputs in canonical [B, T, H, D] layout."""
  rng = np.random.default_rng(case.seed)
  n = rng.standard_normal
  b = 1 if case.is_varlen else case.batch
  t, h, d = case.total_tokens, case.heads, case.head_dim

  out = dict(
      q=n((b, t, h, d), dtype=np.float32),
      k=n((b, t, h, d), dtype=np.float32),
      v=n((b, t, h, d), dtype=np.float32),
      # Raw pre-activation gate.
      g=n((b, t, h, d), dtype=np.float32),
      # Pre-activation beta logits. tokamax wants sigmoid() applied, FlashKDA
      # wants the logits -- see the runners.
      beta_logits=n((b, t, h), dtype=np.float32),
      a_log=(n((h,), dtype=np.float32) * 0.5),
      dt_bias=n((h, d), dtype=np.float32),
      cu_seqlens=case.cu_seqlens,
  )
  # Canonical state layout is K-major [N, H, K, V]. FlashKDA wants [N, H, V, K].
  out["initial_state"] = (
      n((case.num_states, h, d, d), dtype=np.float32) * 0.1
      if case.with_initial_state
      else np.zeros((case.num_states, h, d, d), np.float32)
  )
  out["has_initial_state"] = np.array(case.with_initial_state)
  return out


def activate_gate(g: np.ndarray, a_log: np.ndarray, dt_bias: np.ndarray,
                  lower_bound: float) -> np.ndarray:
  """[B,T,H,K] raw gate -> natural-log decay, matching reference.py:48."""
  gf = g.astype(np.float64) + dt_bias.astype(np.float64)[None, None]
  a = np.exp(a_log.astype(np.float64))[None, None, :, None]
  return lower_bound * _sigmoid(a * gf)


def _sigmoid(x):
  return 0.5 * (1.0 + np.tanh(0.5 * x))


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  xf = x.astype(np.float64)
  return xf / np.sqrt((xf * xf).sum(-1, keepdims=True) + eps)


def reference(case: Case, inp: dict[str, np.ndarray]
              ) -> tuple[np.ndarray, np.ndarray]:
  """Arbiter: token-by-token KDA recurrence in float64.

  Deliberately independent of both backends -- it is the only thing that can
  say *which* kernel drifted. Slow (python loop over T*H) but the cases are
  sized for it.
  """
  d = case.head_dim
  scale = d ** -0.5
  q = l2norm(inp["q"]) * scale
  k = l2norm(inp["k"])
  v = inp["v"].astype(np.float64)
  g = activate_gate(inp["g"], inp["a_log"], inp["dt_bias"], case.lower_bound)
  beta = _sigmoid(inp["beta_logits"].astype(np.float64))

  cu = inp["cu_seqlens"]
  out = np.zeros_like(v)
  states = inp["initial_state"].astype(np.float64).copy()  # [N,H,K,V]

  for nseq in range(len(cu) - 1):
    lo, hi = int(cu[nseq]), int(cu[nseq + 1])
    # Fixed-length mode packs one sequence per batch row; varlen packs them
    # all into row 0.
    bi = 0 if case.is_varlen else nseq
    t0 = lo if case.is_varlen else 0
    t1 = hi if case.is_varlen else hi - lo
    for hh in range(case.heads):
      s = states[nseq, hh]  # [K, V]
      for t in range(t0, t1):
        s = s * np.exp(g[bi, t, hh])[:, None]
        s = s + (beta[bi, t, hh] * k[bi, t, hh])[:, None] * (
            v[bi, t, hh] - k[bi, t, hh] @ s)[None, :]
        out[bi, t, hh] = q[bi, t, hh] @ s
      states[nseq, hh] = s
  return out.astype(np.float32), states.astype(np.float32)


def save_inputs(path: str, case: Case) -> None:
  np.savez(path, **make_inputs(case))


def load_inputs(path: str) -> dict[str, np.ndarray]:
  with np.load(path) as f:
    return {k: f[k] for k in f.files}


if __name__ == "__main__":
  import argparse
  import os
  import time

  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *CASES])
  p.add_argument("--outdir", default="artifacts")
  p.add_argument("--ref", action="store_true",
                 help="also compute the fp64 arbiter (slow)")
  args = p.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  for name in (list(CASES) if args.case == "all" else [args.case]):
    case = CASES[name]
    inp = make_inputs(case)
    np.savez(os.path.join(args.outdir, f"in_{name}.npz"), **inp)
    msg = f"wrote in_{name}.npz  T={case.total_tokens} H={case.heads}"
    if args.ref:
      t0 = time.time()
      o, s = reference(case, inp)
      np.savez(os.path.join(args.outdir, f"ref_{name}.npz"),
               output=o, final_state=s)
      msg += f"  + ref_{name}.npz ({time.time() - t0:.1f}s)"
    print(msg)
