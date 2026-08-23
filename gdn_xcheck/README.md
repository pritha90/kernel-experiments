# GDN cross-check: tokamax (TPU) vs FlashQLA (GPU)

Numerical correctness comparison between two gated-delta-rule kernels that
live on different accelerators and were never meant to be compared:

| | |
|---|---|
| TPU | `tokamax/_src/ops/causal_conv1d_gated_delta_rule` — `PallasMosaicTpuCausalConv1dGatedDeltaRule` (Pallas/Mosaic, TPU v6+) |
| GPU | `QwenLM/FlashQLA` — `chunk_gated_delta_rule` (TileLang, SM90/100/103/120/121) |

Inputs are generated once by numpy, serialised to `.npz`, and consumed
byte-identically by both sides. Neither machine needs the other's toolchain.

## The APIs are not the same op

tokamax's is a **serving** op: it fuses a ragged causal conv1d (with
`conv_state` carry, paged slot table, padded request slots) in front of the
recurrence, forward-only. FlashQLA's is a **training** op: the recurrence
alone, chunked, with a backward pass.

The overlap is prefill. `run_gpu.py` reimplements everything ahead of the
recurrence in torch — ragged depthwise conv + `conv_state` update, silu,
q/k/v split, GVA head expansion, l2norm, `beta = sigmoid(b)`,
`g = -exp(a_log) * softplus(a + dt_bias)`, `scale = d_k**-0.5` — then hands
the result to FlashQLA.

Excluded, and why:

- **Decode.** FlashQLA's chunked kernel has no single-token recurrent path.
  All cases are all-prefill (`distribution = [0, S, S]`).
- **Backward.** tokamax's op has no backward at all.
- **conv1d on the GPU side** is torch, not a kernel — so `conv_state`
  agreement tests the harness, not FlashQLA.

## Two stages

**Stage 1 — semantics.** Both fp32 references, no accelerator, no kernels.
Catches "did I translate the op correctly" before any kernel numerics muddy
the picture. This currently passes at ~1e-7 on all six cases
(`conv_state` is bit-identical), which is what makes stage 2 interpretable.

```bash
python gdn_case.py                                  # write in_*.npz
python run_tpu.py --ref-only                        # tpuref_*.npz  (jax, CPU ok)
python run_gpu.py --ref-only --dtype float32        # gpuref_*.npz  (torch, CPU ok)
python compare.py --stage semantics
```

**Stage 2 — kernels.** The real thing.

```bash
# TPU box (v6+).
TPU_VISIBLE_CHIPS=2 python run_tpu.py               # -> tpu_*.npz

# GPU box.
python run_gpu.py --ref                             # -> gpu_*.npz

# Either box, once both .npz sets are in one directory.
python compare.py --stage kernels
```

`compare.py` prints four rows per tensor so a disagreement is attributable
rather than just observed:

```
ref vs ref    both fp32 scans        -> is the harness right?
tpu vs ref    tokamax kernel error   -> did the TPU kernel drift?
gpu vs ref    FlashQLA kernel error  -> did the GPU kernel drift?
tpu vs gpu    the cross-backend number
```

Tolerance is `rtol=atol=2e-2`, matching tokamax's own
`pallas_mosaic_tpu_test.py`. That is loose, but both kernels run bf16
tensor-core MMAs over a sequential recurrence, so error accumulates along T.
The `ref` rows are what tell you whether a 2e-2 is expected rounding or a bug.

## Cases

| case | shape | exercises |
|---|---|---|
| `single_prefill` | 1×1024, no GQA | baseline |
| `ragged` | 256+128+384 | varlen packing / `cu_seqlens` |
| `gqa` | n_kq=2, n_v=8 | GVA head expansion (`repeat_interleave` vs FlashQLA's native grouping) |
| `with_state` | context 64/128 | live `conv_state` + `recurrent_state` (chunked-prefill continuation) |
| `padded` | 3 valid + 5 padded slots | invalid request slots |
| `unaligned` | 100+37+255 | lengths that are not multiples of `CHUNK_SIZE=64` |

## Notes

- `auto_cp` is **off** by default on the GPU side. FlashQLA's intra-card
  context parallelism splits sequences internally, which perturbs the
  reduction order; `--auto-cp` turns it back on and is worth a second run,
  since that is the production default.
- `use_qk_l2norm_in_kernel=True` is used so FlashQLA's fused l2norm is on the
  critical path. Its eps (1e-6) matches tokamax's `l2_normalize_ref`.
- q/k are explicitly `repeat_interleave`d to `n_v` heads before the call, so
  the `gqa` case compares recurrence math rather than FlashQLA's internal GVA
  head mapping. Dropping the expansion and passing `n_kq` heads directly is
  the way to test that mapping — the two should agree, and if they don't, the
  grouping convention differs.
- tokamax zeroes `recurrent_state` at every slot in `state_indices` whose
  `has_initial_state` is false, **including padded slots the scan never
  visits**. `run_gpu.py` replicates this; without it the `padded` case
  disagrees by ~0.5 on untouched slots.
