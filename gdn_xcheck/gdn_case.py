# Copyright 2026. Apache-2.0.
"""Shared case definitions + deterministic input generation for the
TPU(tokamax) vs GPU(FlashQLA) gated-delta-rule cross-check.

Inputs are generated once with numpy and serialised to an .npz so that both
backends consume *bit-identical* bytes. Nothing here imports jax or torch.

Semantics being matched (from
tokamax/_src/ops/causal_conv1d_gated_delta_rule/reference.py):

  x        = qkv                                     [T, 2*n_kq*d_k + n_v*d_v]
  x        = depthwise_causal_conv1d(x, w, bias)      per-sequence, conv_state as
                                                      left context
  x        = silu(x)
  q,k,v    = split(x)
  q,k      = l2norm(., eps=1e-6)                      then q *= d_k**-0.5
  beta     = sigmoid(b)                               [T, n_v]
  g        = -exp(a_log) * softplus(a + dt_bias)      [T, n_v]  (log space)
  S_t      = exp(g) * S_{t-1} (I - beta k k^T) + beta v k^T
  o_t      = S_t^T q_t
"""

from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class Case:
  """One test configuration."""

  name: str
  # Per-request number of *new* tokens in this forward pass.
  query_lens: tuple[int, ...]
  # Per-request number of tokens already consumed before this pass. Non-zero
  # => has_initial_state, i.e. conv_state / recurrent_state are live.
  context_lens: tuple[int, ...]
  n_kq: int = 2
  n_v: int = 8
  d_k: int = 128
  d_v: int = 128
  kernel_size: int = 4
  # Number of padded (invalid) request slots appended after the valid ones.
  pad_reqs: int = 0
  dtype: str = "bfloat16"
  seed: int = 0

  @property
  def num_tokens(self) -> int:
    return sum(self.query_lens)

  @property
  def num_seqs(self) -> int:
    return len(self.query_lens)

  @property
  def max_reqs(self) -> int:
    return self.num_seqs + self.pad_reqs

  @property
  def conv_dim(self) -> int:
    return 2 * self.n_kq * self.d_k + self.n_v * self.d_v

  @property
  def query_start_loc(self) -> np.ndarray:
    """`cu_seqlens`, padded out to max_reqs+1 by repeating the last value."""
    cu = np.concatenate([[0], np.cumsum(self.query_lens)]).astype(np.int32)
    if self.pad_reqs:
      cu = np.concatenate([cu, np.full(self.pad_reqs, cu[-1], np.int32)])
    return cu

  @property
  def distribution(self) -> np.ndarray:
    # [num_decode_tokens, num_seqs, num_valid_seqs]. All-prefill => decode
    # count 0, which keeps tokamax off its decode-only branch (that branch
    # triggers when distribution[0] == distribution[2]).
    return np.array([0, self.num_seqs, self.num_seqs], np.int32)


# All-prefill only: FlashQLA's chunked kernel has no single-token decode path.
CASES: dict[str, Case] = {
    # Baseline: one long prefill, no history, no GQA.
    "single_prefill": Case(
        name="single_prefill",
        query_lens=(1024,),
        context_lens=(0,),
        n_kq=8,
        n_v=8,
    ),
    # Ragged batch, still no GQA and no history.
    "ragged": Case(
        name="ragged",
        query_lens=(256, 128, 384),
        context_lens=(0, 0, 0),
        n_kq=8,
        n_v=8,
    ),
    # GVA / GQA: n_v // n_kq == 4 value heads per k/q head.
    "gqa": Case(
        name="gqa",
        query_lens=(256, 128, 128),
        context_lens=(0, 0, 0),
        n_kq=2,
        n_v=8,
    ),
    # Chunked-prefill continuation: live conv_state and recurrent_state.
    "with_state": Case(
        name="with_state",
        query_lens=(192, 320),
        context_lens=(64, 128),
        n_kq=2,
        n_v=8,
    ),
    # Padded request slots after the valid ones (vLLM pads to a bucket).
    "padded": Case(
        name="padded",
        query_lens=(128, 64, 32),
        context_lens=(0, 0, 0),
        n_kq=2,
        n_v=8,
        pad_reqs=5,
    ),
    # Sequence length not a multiple of the 64-wide chunk.
    "unaligned": Case(
        name="unaligned",
        query_lens=(100, 37, 255),
        context_lens=(0, 0, 0),
        n_kq=8,
        n_v=8,
    ),
}


def make_inputs(case: Case) -> dict[str, np.ndarray]:
  """Deterministic fp32 inputs. Both backends load exactly these bytes."""
  rng = np.random.default_rng(case.seed)
  n = rng.standard_normal

  t, c = case.num_tokens, case.conv_dim
  # Slot 0 of the state tables is the reserved null block; valid requests use
  # 1..max_reqs, matching tokamax's own tests.
  num_blocks = case.max_reqs + 1

  qkv = n((t, c), dtype=np.float32)
  b = n((t, case.n_v), dtype=np.float32)
  a = n((t, case.n_v), dtype=np.float32)

  conv_weight = n((c, 1, case.kernel_size), dtype=np.float32)
  conv_bias = n((c,), dtype=np.float32)
  # a_log is fed through -exp(); keep it small so the decay doesn't underflow
  # to a degenerate all-zero state, which would hide real disagreement.
  a_log = n((case.n_v,), dtype=np.float32) * 0.5
  dt_bias = n((case.n_v,), dtype=np.float32)

  conv_state = n((num_blocks, case.kernel_size - 1, c), dtype=np.float32)
  recurrent_state = n(
      (num_blocks, case.n_v, case.d_k, case.d_v), dtype=np.float32
  ) * 0.1

  query_lens = np.asarray(case.query_lens, np.int32)
  context_lens = np.asarray(case.context_lens, np.int32)
  # tokamax derives has_initial_state as (seq_lens - query_lens) > 0.
  seq_lens = query_lens + context_lens
  if case.pad_reqs:
    seq_lens = np.concatenate([seq_lens, np.zeros(case.pad_reqs, np.int32)])

  return dict(
      qkv=qkv,
      b=b,
      a=a,
      conv_state=conv_state,
      recurrent_state=recurrent_state,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      a_log=a_log,
      dt_bias=dt_bias,
      query_start_loc=case.query_start_loc,
      state_indices=np.arange(1, case.max_reqs + 1, dtype=np.int32),
      distribution=case.distribution,
      seq_lens=seq_lens.astype(np.int32),
      # Carried along so the runners don't need to re-derive them.
      query_lens=query_lens,
      context_lens=context_lens,
  )


def case_meta(case: Case) -> dict[str, np.ndarray]:
  return {
      f"meta_{k}": np.asarray(v)
      for k, v in dataclasses.asdict(case).items()
  }


def save_inputs(path: str, case: Case) -> None:
  np.savez(path, **make_inputs(case), **case_meta(case))


def load_inputs(path: str) -> dict[str, np.ndarray]:
  with np.load(path) as f:
    return {k: f[k] for k in f.files if not k.startswith("meta_")}


if __name__ == "__main__":
  import argparse
  import os

  p = argparse.ArgumentParser(description="Generate cross-check inputs.")
  p.add_argument("--case", default="all", choices=["all", *CASES])
  p.add_argument("--outdir", default="artifacts")
  args = p.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  names = list(CASES) if args.case == "all" else [args.case]
  for name in names:
    path = os.path.join(args.outdir, f"in_{name}.npz")
    save_inputs(path, CASES[name])
    print(f"wrote {path}")
