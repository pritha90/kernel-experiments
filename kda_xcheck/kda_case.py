# Copyright 2026. Apache-2.0.
"""Shared cases, input generation, and the fp64 arbiter for the KDA
TPU(tokamax) vs GPU(FLA) cross-check.

Canonical layout is `[B, T, H, D]` with a K-major state `[N, H, K, V]` --
which is FLA's native layout, so the GPU side needs no transpose and the TPU
side does. Neither is privileged; the canonical form was chosen to match
whichever side had the more conventional convention.

Nothing in this file imports jax or torch.

KDA (Kimi Delta Attention) is the delta rule with a **per-channel** gate:
unlike Gated DeltaNet's scalar `exp(g_t)`, the state decays as a row scaling
`diag(exp(g_t)) S`, with `g` shaped `[..., K]`.

  g    = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias))   [B,T,H,K]
  q, k = l2norm(., eps=1e-6);  q *= scale
  S_t  = diag(exp(g_t)) S_{t-1}
  S_t += beta_t k_t (v_t - k_t^T S_t)^T
  o_t  = q_t^T S_t

`beta` is carried in POST-activation form (already in [0,1]) because that is
what tokamax accepts, and it makes `dbeta` mean the same thing on both sides.
FLA is therefore called with `use_beta_sigmoid_in_kernel=False`.
"""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np

# The eight differentiable inputs. tokamax's VJP returns grads for exactly
# this set (pallas_mosaic_tpu.py `grads`), as does FLA's ChunkKDAFunction
# backward (dq, dk, dv, dg, db, dA, dbias, dh0).
DIFFERENTIABLE = ("q", "k", "v", "g", "beta", "a_log", "dt_bias",
                  "initial_state")


@dataclasses.dataclass(frozen=True)
class Case:
  """One test configuration.

  `seq_lens` non-empty selects varlen mode: B is forced to 1, tokens are
  packed, and the runners emit `segment_ids` (tokamax) / `cu_seqlens` (FLA).
  """

  name: str
  batch: int = 1
  seq_len: int = 256
  heads: int = 4
  head_dim: int = 128
  seq_lens: tuple[int, ...] = ()
  with_initial_state: bool = False
  # Must be non-None and in [-5, 0): it selects the sigmoid gate branch that
  # both backends implement identically. tokamax's other branch
  # (-exp(a_log)*softplus(g)) exists in FLA too, but is not exercised here --
  # see README "Known exclusions".
  lower_bound: float = -2.0
  seed: int = 0

  @property
  def value_dim(self) -> int:
    return self.head_dim

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
    # it in fixed-length mode (pallas_mosaic_tpu.py:165); FLA's chunk_size is
    # also 64, so the two chunk boundaries coincide.
    "fixed": Case(name="fixed", batch=2, seq_len=256, heads=4),
    # Live initial state.
    "fixed_state": Case(
        name="fixed_state", batch=1, seq_len=512, heads=4,
        with_initial_state=True,
    ),
    # B>1 *and* a live state. The only case where the fixed-length state
    # mapping [N=B,H,K,V] -> [B,1,H,K,V] is non-trivial: with B=1 a wrong
    # axis still lines up by accident.
    "fixed_state_b2": Case(
        name="fixed_state_b2", batch=2, seq_len=192, heads=2,
        with_initial_state=True,
    ),
    # Head dim below 128. FLA allows K <= 256 and Mosaic allows K <= 256, so
    # unlike the FlashKDA comparison (which hard-required K == V == 128) this
    # is now testable.
    "small_dim": Case(
        name="small_dim", batch=2, seq_len=256, heads=4, head_dim=64,
    ),
    # Varlen, segments aligned to the shared chunk size of 64.
    "varlen": Case(name="varlen", seq_lens=(192, 64, 256), heads=4),
    # Varlen with segments aligned to nothing; exercises ragged chunk tails.
    "varlen_unaligned": Case(
        name="varlen_unaligned", seq_lens=(100, 156, 57), heads=4),
    # Varlen carrying state, the chunked-prefill shape.
    "varlen_state": Case(
        name="varlen_state", seq_lens=(128, 384), heads=2,
        with_initial_state=True,
    ),
    # Long sequence: the recurrence is sequential, so error accumulates in T.
    "long": Case(name="long", batch=1, seq_len=2048, heads=2),
}


def make_inputs(case: Case) -> dict[str, np.ndarray]:
  """Deterministic fp32 inputs in canonical [B, T, H, D] layout.

  The seed is mixed with the case name, not just `case.seed`. Cases with the
  same total element count would otherwise draw the *same* numbers from the
  stream and differ only in how they are reshaped -- `fixed` (B=2,T=256),
  `fixed_state` (B=1,T=512), and `varlen` (192+64+256) are all 512x4x128, and
  were byte-identical before this. They then report identical errors, which
  looks like agreement but is really one sample counted three times.

  crc32, not hash(): hash() is salted per process.
  """
  rng = np.random.default_rng([case.seed, zlib.crc32(case.name.encode())])
  n = rng.standard_normal
  b = 1 if case.is_varlen else case.batch
  t, h = case.total_tokens, case.heads
  dk, dv = case.head_dim, case.value_dim

  out = dict(
      q=n((b, t, h, dk), dtype=np.float32),
      k=n((b, t, h, dk), dtype=np.float32),
      v=n((b, t, h, dv), dtype=np.float32),
      # Raw pre-activation gate.
      g=n((b, t, h, dk), dtype=np.float32),
      # POST-activation beta in (0, 1). tokamax validates this range; FLA is
      # called with use_beta_sigmoid_in_kernel=False so it takes the same.
      beta=_sigmoid(n((b, t, h), dtype=np.float32)).astype(np.float32),
      a_log=(n((h,), dtype=np.float32) * 0.5),
      # Flattened [H*K] -- both backends take it flat and reshape to [H, K].
      dt_bias=n((h * dk,), dtype=np.float32),
      cu_seqlens=case.cu_seqlens,
  )
  out["initial_state"] = (
      n((case.num_states, h, dk, dv), dtype=np.float32) * 0.1
      if case.with_initial_state
      else np.zeros((case.num_states, h, dk, dv), np.float32)
  )
  out["has_initial_state"] = np.array(case.with_initial_state)

  # Output cotangents for the backward comparison. Both backends are seeded
  # with these identical values so the gradients are directly comparable.
  out["do"] = n((b, t, h, dv), dtype=np.float32)
  out["dht"] = n((case.num_states, h, dk, dv), dtype=np.float32)
  return out


def _sigmoid(x):
  return 0.5 * (1.0 + np.tanh(0.5 * x))


def activate_gate(g, a_log, dt_bias, lower_bound):
  """[B,T,H,K] raw gate -> log-space decay.

  Identical formula in both backends:
    tokamax reference.py:48  `lower_bound * sigmoid(exp(a_log) * g_f)`
    FLA gate.py:88-92        `lower_bound * sigmoid(exp(A_log) * g)`
  """
  h, dk = g.shape[2], g.shape[3]
  gf = g.astype(np.float64) + dt_bias.astype(np.float64).reshape(h, dk)
  a = np.exp(a_log.astype(np.float64))[None, None, :, None]
  return lower_bound * _sigmoid(a * gf)


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  xf = x.astype(np.float64)
  return xf / np.sqrt((xf * xf).sum(-1, keepdims=True) + eps)


def reference(case: Case, inp: dict[str, np.ndarray]
              ) -> tuple[np.ndarray, np.ndarray]:
  """Forward arbiter: token-by-token KDA recurrence in float64.

  Independent of both backends. `arbiter.py` reimplements this in JAX to get
  gradients by autodiff, and stage 1 checks the two agree -- so this function
  also pins down the JAX version.
  """
  scale = case.head_dim ** -0.5
  q = l2norm(inp["q"]) * scale
  k = l2norm(inp["k"])
  v = inp["v"].astype(np.float64)
  g = activate_gate(inp["g"], inp["a_log"], inp["dt_bias"], case.lower_bound)
  beta = inp["beta"].astype(np.float64)

  cu = inp["cu_seqlens"]
  out = np.zeros_like(v)
  states = inp["initial_state"].astype(np.float64).copy()  # [N,H,K,V]

  for nseq in range(len(cu) - 1):
    lo, hi = int(cu[nseq]), int(cu[nseq + 1])
    # Fixed-length mode puts one sequence per batch row; varlen packs them
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
  # float64 out, not float32: this is compared against arbiter.py's float64
  # forward, and rounding either side to fp32 would swamp that comparison
  # (~2e-09 of pure storage error) and hide whatever it was meant to measure.
  return out, states


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
                 help="also compute the float64 forward arbiter")
  args = p.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  for name in (list(CASES) if args.case == "all" else [args.case]):
    case = CASES[name]
    inp = make_inputs(case)
    np.savez(os.path.join(args.outdir, f"in_{name}.npz"), **inp)
    msg = f"wrote in_{name}.npz  T={case.total_tokens} H={case.heads} D={case.head_dim}"
    if args.ref:
      t0 = time.time()
      o, s = reference(case, inp)
      np.savez(os.path.join(args.outdir, f"npref_{name}.npz"),
               output=o, final_state=s)
      msg += f"  + npref_{name}.npz ({time.time() - t0:.1f}s)"
    print(msg)
