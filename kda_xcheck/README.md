# KDA cross-device numerical correctness

Numerical correctness tests between two independent Kimi Delta Attention
implementations, **forward and backward**:

| | |
|---|---|
| **TPU** | [tokamax PR #1103](https://github.com/openxla/tokamax/pull/1103), `tokamax/_src/ops/experimental/kda/` — Pallas/Mosaic |
| **GPU** | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), `fla/ops/kda/` — Triton (`chunk_fwd.py` / `chunk_bwd.py`) |

KDA is the delta rule with a **per-channel** gate: unlike Gated DeltaNet's
scalar `exp(g_t)`, the state decays as a row scaling `diag(exp(g_t)) S`, with
`g` shaped `[..., K]`.

```
g    = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias))
q, k = l2norm(., eps=1e-6);  q *= K**-0.5
S_t  = diag(exp(g_t)) S_{t-1}
S_t += beta_t k_t (v_t - k_t^T S_t)^T
o_t  = q_t^T S_t
```

FLA is the right comparison target for this op: Moonshot did not open-source
a training kernel of their own. FlashKDA is inference-only (every entry point
is `fwd_*`, and its README mandates `torch.inference_mode()`), and
`MoonshotAI/Kimi-Linear` states the KDA kernel was contributed to FLA. So FLA
is *the* KDA training kernel, and it is the only one with a backward to
compare against tokamax's.

## What is compared

Both sides differentiate the same eight inputs, and they line up exactly —
tokamax's VJP returns a `grads` dict keyed `query, key, value, gate, beta,
a_log, delta_time_bias, initial_state` (`pallas_mosaic_tpu.py`), and FLA's
`ChunkKDAFunction.backward` returns `dq, dk, dv, dg, db, dA, dbias, dh0`.
Ten tensors are compared per case: `output`, `final_state`, and those eight.

## API differences the harness has to bridge

FLA is a far closer match than FlashKDA was — it shares the canonical
token-first layout, the flat `[H*K]` dt_bias, the K-major fp32 state, and the
gate formula, so most of the old conversion surface is simply gone. What is
left:

| | tokamax (TPU) | FLA (GPU) |
|---|---|---|
| layout | `[H,B,T,K]` head-first | `[B,T,H,K]` — canonical |
| value | `[H,B,T,V]` | `[B,T,HV,V]`, GVA when `HV > H` |
| beta | post-activation, validated ∈ [0,1] | either; `use_beta_sigmoid_in_kernel` selects |
| gate | raw, `use_gate_in_kernel=True` | raw, `use_gate_in_kernel=True` |
| `dt_bias` | flattened `[H*K]` | flattened `[HV*K]` — same |
| state | `[B,N,H,K,V]` K-major | `[N,HV,K,V]` K-major, **must be fp32** |
| varlen | `segment_ids [B,T]`, **1-indexed**, 0 = pad | `cu_seqlens [N+1]` int64, B=1 |
| chunk | 64 (Mosaic requirement) | 32 or 64 → **set to 64**, so boundaries coincide |

Two deliberate choices:

- **Post-sigmoid beta on both sides** (`use_beta_sigmoid_in_kernel=False`).
  tokamax only accepts post-activation beta, so feeding FLA the logits would
  make its `db` a different derivative — chained through `sigmoid'` — and the
  two gradients would not be comparable even when both are correct.
- **`safe_gate=False`.** It is an independent TensorCore/clamping option with
  no tokamax counterpart, so enabling it would confound the comparison.

> **FLA's `chunk_kda` docstring is wrong about `lower_bound`.** It says the
> sigmoid gate branch applies "when set together with `safe_gate=True`". In
> the code the branch keys off `lower_bound` alone — `USE_LOWER_BOUND` in
> `gate.py:100/163/395` — and `lower_bound` is forwarded unconditionally in
> both `chunk_fwd.py:55` and `chunk_bwd.py:474`. `safe_gate` only adds
> clamping and the M=16 path. Taking the docstring at face value would mean
> comparing tokamax's sigmoid gate against FLA's softplus gate.

## Design: one shared arbiter

Every backend is scored against a **float64 arbiter**, which is what makes a
disagreement *attributable* rather than merely observable — a bare
kernel-vs-kernel number tells you nothing about which side moved.

```
   npref_  numpy fp64 forward (kda_case.reference)
      |
      +--> ref_  JAX fp64 forward + VJP (arbiter.py)      <- the arbiter
                  |
        +---------+---------+
        |                   |
   tpuref_  tokamax xla   gpuref_  FLA naive_recurrent_kda
   tpu_     tokamax mosaic   gpu_  FLA chunk_kda (Triton)
```

The arbiter's gradients come from autodiff, so they are only as trustworthy
as the forward they differentiate. That forward is therefore checked against
`kda_case.reference()`, a *separate* fp64 implementation written as an
imperative numpy loop. Both artifacts are stored in float64 for this reason:
rounding either to fp32 would inject ~2e-09 and swamp the comparison.

**Both `*ref_` references run on CPU.** That is new with the FLA retarget —
FlashKDA's torch reference is a bit-exact CUDA emulation and needed a device,
so only the TPU conversions could be validated without hardware. FLA's
`naive_recurrent_kda` is pure PyTorch, so the whole of stage 1 is now
two-sided and laptop-runnable. Triton is not required either: if it is
missing, `run_gpu.py` loads `naive.py` directly from the source tree, since
`fla/ops/__init__.py` eagerly imports every Triton op in the library.

## Stages

### Stage 1 — semantics (CPU only, no accelerator)

```bash
TOKAMAX=<tokamax-pr1103> FLA_ROOT=<fla> ./run_stage1.sh
```

or by hand:

```bash
python kda_case.py --case all --ref                 # inputs + numpy fp64 fwd
python arbiter.py  --case all                       # fp64 fwd + VJP
python run_tpu.py  --case all --impl xla   --dtype float32 --backward
python run_gpu.py  --case all --impl naive --dtype float32 --backward
python compare.py  --stage semantics
```

**Status: passing, 8/8 cases, both backends, forward and backward.**

| comparison | worst across all cases |
|---|---|
| `npref vs ref` (fp64 vs fp64) | max\|Δ\| 1.9e-15 |
| `tpuref vs ref` forward | max\|Δ\| 3.6e-07 |
| `tpuref vs ref` gradients | rel-norm 9.4e-07 |
| `gpuref vs ref` forward | max\|Δ\| 3.8e-07 |
| `gpuref vs ref` gradients | rel-norm 1.6e-06 |

`test_conversions.py` is the negative control. It injects 21 conversion bugs
— 12 on the TPU side, 9 on the GPU side — and asserts each is caught:

```bash
FLA_ROOT=<fla> PYTHONPATH=<tokamax-pr1103> python test_conversions.py
```

**Status: passing — controls sit at ≤0.16× threshold on both sides, all 21
mutants detected** (the smallest offender is 8.8e+03× over threshold; three
are rejected outright by shape/type validation).

Writing it found a real defect: with `a_log` out of the graph — which FLA
permits, `A_log=None` is legal whenever `lower_bound` is set — `run_gpu.py`
raised "does not have been used in the graph" instead of reporting a zero
gradient. It now zero-fills unused grads, matching what tokamax's VJP already
did.

### Stage 2 — kernels

```bash
# on a TPU host
PYTHONPATH=<tokamax-pr1103> python run_tpu.py --case all --impl mosaic \
    --dtype bfloat16 --backward
# on a CUDA host
python run_gpu.py --case all --impl chunk --dtype bfloat16 --backward
# anywhere
python compare.py --stage kernels
```

**Status: not run against the current harness.** An earlier forward-only
revision was executed on a v7x TPU and all 7 cases then defined ran on Mosaic
with no `NotImplementedError` (bf16 output max\|Δ\| 4.0e-04–7.0e-04 vs the
arbiter, final_state 3.3e-03–5.6e-03). Those artifacts predate the
post-sigmoid-beta input change, the added cotangents, and the `small_dim`
case, so they are stale and are being regenerated rather than reused. The
CUDA half has not run at all — FLA's KDA kernels are Triton-only.

`compare.py --stage kernels` prints, per tensor, `tpu vs arbiter`,
`gpu vs arbiter`, `fla-naive vs arbiter`, and `tpu vs gpu`, so a
cross-backend delta can be assigned to a specific kernel. Only `tpu vs gpu`
decides pass/fail when both are present; with only one side present it says
so explicitly and falls back to scoring against the arbiter.

`da_log` and `ddt_bias` are scored on relative norm rather than elementwise
tolerance: they are reductions over every token and channel (shapes `[H]` and
`[H*K]`), so both their magnitude and their accumulated rounding error scale
with `B*T`, and an atol tuned for per-token tensors is meaningless there.

The `.npz` artifacts are the exchange format: generate inputs once, carry
`artifacts/` between the two hosts, compare offline. Both kernels consume
identical bytes — including the `do`/`dht` output cotangents, so the
gradients are directly comparable rather than only comparable in
distribution.

## Cases

| case | shape | state | exercises |
|---|---|---|---|
| `fixed` | B=2, T=256, H=4 | — | baseline |
| `fixed_state` | B=1, T=512, H=4 | ✓ | state carry |
| `fixed_state_b2` | B=2, T=192, H=2 | ✓ | `[N=B,H,K,V] → [B,1,H,K,V]`; the only case where this mapping is non-trivial |
| `small_dim` | B=2, T=256, H=4, D=64 | — | K ≠ 128, newly testable (FlashKDA hard-required K=V=128) |
| `varlen` | 192+64+256 | — | segments aligned to the shared chunk size of 64 |
| `varlen_unaligned` | 100+156+57 | — | segments aligned to nothing; ragged chunk tails |
| `varlen_state` | 128+384 | ✓ | chunked prefill; catches swapped state segments |
| `long` | B=1, T=2048, H=2 | — | error accumulation along the sequential recurrence |

All cases use `lower_bound = -2.0`. Seeds are mixed with the case *name*
(`crc32`, not `hash()`, which is salted per process): several cases have
identical total element counts and would otherwise draw byte-identical
tensors, reporting one sample three times as if it were three agreements.

## Known exclusions

- **`lower_bound=None`.** Both libraries have the other gate branch,
  `-exp(a_log) * softplus(g + dt_bias)`, but no case exercises it.
  `test_conversions.py` confirms the two branches are distinguishable
  (2.3e+05× over threshold), i.e. picking the wrong one would not go
  unnoticed.
- **GVA (`HV > H`).** FLA supports more value heads than qk heads; tokamax
  has no equivalent.
- **Context parallelism.** Both have a CP path; they are not compared.
- **`chunk_size=32`.** FLA allows it, Mosaic requires 64.
- **`safe_gate=True`**, **`allow_neg_eigval=True`**, **`state_v_first=True`** —
  FLA-only options with no tokamax counterpart.
- **`use_gate_in_kernel=False`** and **`use_qk_l2norm=False`**: both
  libraries support them, but the fused path is what production uses.
- **Mosaic shape limits** (`pallas_mosaic_tpu.py`): bf16/fp32 only, `K ≤ 256`,
  `chunk_size == 64`, `T % 64 == 0` when `segment_ids is None`, and exactly
  one state per batch item in fixed-length mode. Every case above satisfies
  these, so a `NotImplementedError` in stage 2 would be a regression, not an
  expected skip — `run_tpu.py` reports it and continues rather than aborting
  the sweep.
