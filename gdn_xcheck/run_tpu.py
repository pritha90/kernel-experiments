# Copyright 2026. Apache-2.0.
"""TPU side of the cross-check: tokamax Pallas-Mosaic-TPU GDN kernel.

Runs, for each case:
  * the Pallas TPU kernel (PallasMosaicTpuCausalConv1dGatedDeltaRule)
  * the fp32 token-by-token scan reference (CausalConv1dGatedDeltaRule)

The reference is the neutral arbiter: if GPU and TPU disagree, it tells us
which one moved.

Usage (set TPU_VISIBLE_CHIPS if the host's other chips are in use):

  python run_tpu.py --indir artifacts --outdir artifacts
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from tokamax._src.ops.causal_conv1d_gated_delta_rule import base

import gdn_case

# Everything else stays fp32: only the activations get cast, mirroring how the
# GPU side has to feed bf16 into tensor-core MMAs.
_CAST = ("qkv", "conv_weight", "conv_bias", "conv_state")


def _to_jax(inputs: dict[str, np.ndarray], dtype) -> dict:
  out = {}
  for k, v in inputs.items():
    if k in ("query_lens", "context_lens"):
      continue
    arr = jnp.asarray(v)
    if k in _CAST and dtype is not None:
      arr = arr.astype(dtype)
    out[k] = arr
  return out


def run_case(case: gdn_case.Case, inputs: dict[str, np.ndarray], dtype,
             *, ref_only: bool = False):
  static = dict(
      n_kq=case.n_kq,
      n_v=case.n_v,
      d_k=case.d_k,
      d_v=case.d_v,
      kernel_size=case.kernel_size,
  )
  names = ["n_kq", "n_v", "d_k", "d_v", "kernel_size", "config"]

  res = {}
  if not ref_only:
    # TPU-only import: pulls in the Mosaic wrapper.
    from tokamax._src.ops.causal_conv1d_gated_delta_rule import (  # noqa: PLC0415
        pallas_mosaic_tpu,
    )

    kernel = jax.jit(
        pallas_mosaic_tpu.PallasMosaicTpuCausalConv1dGatedDeltaRule(),
        static_argnames=names,
    )
    (conv_k, rec_k), out_k = kernel(**_to_jax(inputs, dtype), **static)
    res |= {
        "output": np.asarray(out_k, np.float32),
        "conv_state": np.asarray(conv_k, np.float32),
        "recurrent_state": np.asarray(rec_k, np.float32),
    }

  # Arbiter runs in full fp32 regardless of the case dtype.
  reference = jax.jit(base.CausalConv1dGatedDeltaRule(), static_argnames=names)
  (conv_r, rec_r), out_r = reference(**_to_jax(inputs, None), **static)
  res |= {
      "ref_output": np.asarray(out_r, np.float32),
      "ref_conv_state": np.asarray(conv_r, np.float32),
      "ref_recurrent_state": np.asarray(rec_r, np.float32),
  }
  return res


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *gdn_case.CASES])
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  p.add_argument("--dtype", default="bfloat16",
                 choices=["bfloat16", "float16", "float32"])
  p.add_argument("--ref-only", action="store_true",
                 help="skip the Pallas kernel and emit only the fp32 "
                      "reference. Runs anywhere, including CPU.")
  args = p.parse_args()

  dtype = getattr(jnp, args.dtype)
  print(f"jax {jax.__version__} backend={jax.default_backend()} "
        f"devices={jax.devices()}")

  os.makedirs(args.outdir, exist_ok=True)
  names = list(gdn_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    case = gdn_case.CASES[name]
    src = os.path.join(args.indir, f"in_{name}.npz")
    inputs = gdn_case.load_inputs(src)
    try:
      res = run_case(case, inputs, dtype, ref_only=args.ref_only)
    except Exception as e:  # noqa: BLE001 - one bad case shouldn't kill the run
      print(f"[{name}] FAILED: {type(e).__name__}: {e}")
      continue
    dst = os.path.join(
        args.outdir, f"{'tpuref' if args.ref_only else 'tpu'}_{name}.npz")
    np.savez(dst, **res)
    extra = ""
    if "output" in res:
      d = np.abs(res["output"] - res["ref_output"]).max()
      extra = f"  (kernel vs fp32 ref, max|d| = {d:.3e})"
    print(f"[{name}] ok -> {dst}{extra}")


if __name__ == "__main__":
  main()
