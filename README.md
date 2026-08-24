# kernel-experiments

Cross-device numerical correctness harnesses for linear-attention kernels.
Each compares a TPU implementation against an independent GPU implementation
of the same operator.

| harness | TPU | GPU | scope | status |
|---|---|---|---|---|
| [`gdn_xcheck/`](gdn_xcheck/) | tokamax `causal_conv1d_gated_delta_rule` | [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) | forward | stage 1 passing; stage 2 needs hardware |
| [`kda_xcheck/`](kda_xcheck/) | tokamax `experimental/kda` ([PR #1103](https://github.com/openxla/tokamax/pull/1103)) | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `fla/ops/kda` | forward **+ backward** | stage 1 passing (both sides); stage 2 GPU half passing on CPU/MPS, TPU half needs hardware |

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

Neither harness has run its TPU kernel against the current revision, and
neither has run a real CUDA kernel at all — FlashQLA dispatches on
`sm90/100/103/120/121` and FLA's `chunk_kda` is Triton, so both need hardware
that is not available here. Everything else is executed and passing.

`kda_xcheck` goes further than `gdn_xcheck` in three ways, because its GPU
target supports it:

- It compares **gradients** — tokamax's VJP and FLA's
  `ChunkKDAFunction.backward` return grads for the same eight inputs.
- Its stage 1 is **two-sided**, since FLA ships a pure-PyTorch statement of
  the recurrence that runs anywhere, where FlashQLA and FlashKDA do not.
- Its stage 2 **GPU half runs without CUDA**, via `--impl chunk_torch`: FLA's
  `naive_chunk_kda` is the same chunked algorithm as the Triton kernel,
  written in torch, so the algorithm can be scored against the arbiter (and
  against Mosaic) on a laptop. It passes 7/8 cases forward and backward on
  both CPU and MPS.

That last path immediately earned its keep: it exposed a NaN in
`naive_chunk_kda`'s backward at `chunk_size=64`, where an `exp` over the
masked-off upper triangle overflows fp32 and the discarded `inf` comes back
as `0 * inf`. See [`kda_xcheck/README.md`](kda_xcheck/README.md).
