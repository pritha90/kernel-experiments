# kernel-experiments

Cross-device numerical correctness harnesses for linear-attention kernels.
Each compares a TPU implementation against an independent GPU implementation
of the same operator.

| harness | TPU | GPU | scope | status |
|---|---|---|---|---|
| [`gdn_xcheck/`](gdn_xcheck/) | tokamax `causal_conv1d_gated_delta_rule` | [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) | forward | stage 1 passing; stage 2 needs hardware |
| [`kda_xcheck/`](kda_xcheck/) | tokamax `experimental/kda` ([PR #1103](https://github.com/openxla/tokamax/pull/1103)) | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla/ops/kda` | forward **+ backward** | stage 1 passing (both sides); stage 2 needs hardware |

## Shared design

Both harnesses are built the same way, because the hard part is not the
comparison — it is making a disagreement *mean* something.

1. **Generate inputs once, in numpy, to `.npz`.** Both backends consume
   identical bytes. The artifacts are the exchange format between a TPU host
   and a GPU host; comparison happens offline on either.

2. **Split into two stages.**
   - *Stage 1 (semantics)* compares fp32 references only, no kernels. It
     validates every layout, activation, and indexing conversion between the
     two incompatible APIs. It runs on CPU, so it is checkable without any
     accelerator — and it is where the real bugs have shown up.
   - *Stage 2 (kernels)* compares the actual kernels, at bf16 tolerances.

3. **Score every backend against a shared reference**, so a cross-backend
   delta is attributable to one side rather than merely observed.

4. **Never report a comparison that did not happen.** Missing artifacts are
   `SKIP`, not `PASS`; an all-skip run exits non-zero.

Stage 1 is not self-validating — a check that can never fail would also pass
— so `kda_xcheck` carries `test_conversions.py`, which injects each
conversion bug the harness is meant to catch and asserts it is caught. It has
found real defects in the harness twice.

## Status

Stage 2 has not been run for either harness: it needs a TPU host and an
NVIDIA GPU host, and neither GPU kernel runs on CPU (FlashQLA dispatches on
`sm90/100/103/120/121`; FLA's KDA kernels are Triton). Everything up to that
point is executed and passing.

`kda_xcheck` goes further than `gdn_xcheck` in two ways, because its GPU
target supports it. It compares **gradients** — tokamax's VJP and FLA's
`ChunkKDAFunction.backward` return grads for the same eight inputs — and its
stage 1 is **two-sided**, since FLA ships a pure-PyTorch reference that runs
on CPU where FlashQLA and FlashKDA do not.
