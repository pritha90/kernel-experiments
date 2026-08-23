# KDA cross-device numerical correctness

Numerical correctness tests between two independent Kimi Delta Attention
kernels:

| | |
|---|---|
| **TPU** | [tokamax PR #1103](https://github.com/openxla/tokamax/pull/1103), `tokamax/_src/ops/experimental/kda/` — Pallas/Mosaic |
| **GPU** | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) — CUDA |

KDA is the delta rule with a **per-channel** gate: unlike Gated DeltaNet's
scalar `exp(g_t)`, the state decays as a row scaling `diag(exp(g_t)) S`, with
`g` shaped `[..., K]`.

```
g    = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias))
q, k = l2norm(., eps=1e-6);  q *= K**-0.5
beta = sigmoid(beta_logits)
S_t  = diag(exp(g_t)) S_{t-1}
S_t += beta_t k_t (v_t - k_t^T S_t)^T
o_t  = q_t^T S_t
```

## Why this needs a harness and not just `allclose`

The two APIs describe the same math in incompatible conventions. Nine things
differ, and every one of them is silently wrong rather than loudly wrong —
`K == V == 128` is required by FlashKDA, so **a transposed state has the
right shape**.

| | tokamax (TPU) | FlashKDA (GPU) |
|---|---|---|
| layout | `[H,B,T,K]` head-first | `[B,T,H,K]` |
| beta | **post-activation**, validated ∈ [0,1] | **logits**, sigmoid applied internally |
| gate | raw, when `use_gate_in_kernel=True` | raw |
| `dt_bias` | flattened `[H*K]` | `[H,K]` fp32 |
| gate internals | natural log, `exp` | base-2, `exp2` with flush-to-zero |
| varlen | `segment_ids [B,T]`, **1-indexed**, 0 = pad | `cu_seqlens [N+1]` int64, B=1 |
| state | `[B,N,H,K,V]` **K-major** | `[N,H,V,K]` **V-major** |
| output | returned | written in place into `out` |
| chunk | 64 (Mosaic requirement) | 16 |

The canonical layout in `kda_case.py` is `[B,T,H,D]` with a K-major state;
each runner converts into its own convention. Neither backend's layout is
privileged.

## Design: one shared arbiter

Every backend is scored against **`kda_case.reference()`**, a token-by-token
recurrence in float64. This is what makes a disagreement *attributable*
rather than merely observable — with only a kernel-vs-kernel number, a
mismatch tells you nothing about which side moved.

```
                    ref_      float64 arbiter  (kda_case.py, CPU)
                      |
        +-------------+-------------+
        |                           |
   tpuref_  tokamax xla        gpuref_  FlashKDA tests/torch_ref.py
   tpu_     tokamax mosaic     gpu_     FlashKDA CUDA kernel
```

**`gpuref_` is not an fp32 reference.** FlashKDA's `tests/torch_ref.py`
deliberately emulates its own kernel bit-for-bit: inline-PTX
`tanh.approx.f32` sigmoid, cuBLAS GEMMs with `CUBLAS_COMPUTE_16F` (fp16
accumulation), `exp2` with flush-to-zero, and an l2-norm reproducing the
warp-shuffle tree reduction. It requires CUDA and is *expected* to sit at
bf16-level error against the arbiter. Its value is localizing a GPU-side
failure to the CUDA code rather than to the algorithm.

## Stages

### Stage 1 — semantics (runs on CPU, no accelerator)

Does tokamax's own fp32 reference agree with the arbiter? This validates
every conversion in `run_tpu.py` before a kernel is ever involved.

```bash
TOKAMAX=<tokamax-pr1103> ./run_stage1.sh     # all of the below
```

or by hand:

```bash
python kda_case.py --case all --ref
PYTHONPATH=<tokamax-pr1103> python run_tpu.py --case all --impl xla --dtype float32
python compare.py --stage semantics
```

**Status: passing, all 7 cases, max |Δ| ≤ 1.2e-07** on output and final
state.

`test_conversions.py` is the negative control for stage 1 — it injects ten
conversion bugs (beta as logits, column-major `dt_bias`, 0-indexed
`segment_ids`, state on the wrong axis, reversed segments, …) and asserts
each is caught. Without it, a stage that never fails would also "pass".

```bash
PYTHONPATH=<tokamax-pr1103> python test_conversions.py
```

**Status: passing — controls agree, all 10 mutants detected.**

### Stage 2 — kernels

```bash
# on a TPU host
PYTHONPATH=<tokamax-pr1103> python run_tpu.py --case all --impl mosaic --dtype bfloat16
# on a CUDA host
python run_gpu.py --case all --impl kernel
python run_gpu.py --case all --impl torch_ref --flashkda-root <FlashKDA>
# anywhere
python compare.py --stage kernels
```

**Status: not run — needs both a TPU and an NVIDIA GPU.** Stage 2 is the
only part of this harness that has not been executed.

`compare.py --stage kernels` prints five rows per tensor (`tpu vs arbiter`,
`gpu vs arbiter`, `torchref vs arbiter`, `gpu vs torchref`, `tpu vs gpu`) so
that a cross-backend delta can be assigned to a specific kernel. Only
`tpu vs gpu` decides pass/fail, at `rtol=atol=2e-2` — appropriate for a
chunked bf16 linear-attention kernel over a sequential recurrence.

The `.npz` artifacts are the exchange format: generate inputs once, carry
`artifacts/` between the two hosts, compare offline. Both kernels consume
identical bytes.

## Cases

| case | shape | state | exercises |
|---|---|---|---|
| `fixed` | B=2, T=256, H=4 | — | baseline |
| `fixed_state` | B=1, T=512, H=4 | ✓ | state carry |
| `fixed_state_b2` | B=2, T=192, H=2 | ✓ | `[N=B,H,K,V] → [B,1,H,K,V]`; the only case where this mapping is non-trivial |
| `varlen` | 192+64+256 | — | segments aligned to both chunk sizes (64 / 16) |
| `varlen_unaligned` | 100+156+57 | — | segments aligned to neither; ragged tails |
| `varlen_state` | 128+384 | ✓ | chunked prefill; catches swapped state segments |
| `long` | B=1, T=2048, H=2 | — | error accumulation along the sequential recurrence |

All cases use `K = V = 128` (FlashKDA requires it) and `lower_bound = -2.0`.

## Known exclusions

- **Backward.** FlashKDA exposes forward only (`fwd`); tokamax has a custom
  VJP. Nothing here compares gradients.
- **`lower_bound=None`.** tokamax's other gate branch is `-exp(a_log) *
  softplus(g)`. FlashKDA has no equivalent, so it cannot be cross-checked —
  `test_conversions.py` confirms the two branches are distinguishable
  (err=2.2e-01), i.e. picking the wrong one would not go unnoticed.
- **Context parallelism.** tokamax's CP path has no FlashKDA counterpart.
- **`use_gate_in_kernel=False`** and **`use_qk_l2norm=False`**. FlashKDA
  always does both internally.
- **Mosaic shape limits** (`pallas_mosaic_tpu.py`): bf16/fp32 only, `K ≤ 256`,
  `chunk_size == 64`, `T % 64 == 0` when `segment_ids is None`, and exactly
  one state per batch item in fixed-length mode. Every case above satisfies
  these, so a `NotImplementedError` in stage 2 would be a regression, not an
  expected skip — `run_tpu.py` reports it and continues rather than aborting
  the sweep.
