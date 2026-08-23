# Copyright 2026. Apache-2.0.
"""Run the tokamax KDA op (PR #1103) on the canonical inputs.

Two implementations are exposed:
  --impl mosaic   the Pallas/Mosaic TPU kernel  (requires a TPU)
  --impl xla      the token-by-token JAX reference (runs anywhere, incl. CPU)

The `xla` path is what makes the harness checkable without hardware: it is
tokamax's own statement of KDA semantics, so agreement between it and the
numpy arbiter in kda_case.py validates every layout/activation conversion
below before either kernel is ever run.

Usage:
  python run_tpu.py --case all --impl xla    --dtype float32   # stage 1
  python run_tpu.py --case all --impl mosaic --dtype bfloat16  # stage 2
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import kda_case


# Activations cast to the run dtype; states and per-head params stay fp32
# because both backends accumulate the recurrence in fp32.
_CAST = ("query", "key", "value", "gate", "beta")


def to_tokamax(case: kda_case.Case, inp: dict[str, np.ndarray]) -> dict:
  """Canonical [B,T,H,D] -> tokamax's head-first [H,B,T,D] convention."""
  hbtd = lambda x: np.ascontiguousarray(x.transpose(2, 0, 1, 3))

  args = dict(
      query=hbtd(inp["q"]),
      key=hbtd(inp["k"]),
      value=hbtd(inp["v"]),
      gate=hbtd(inp["g"]),
      # tokamax validates beta in [0, 1] (base.py `_validate_beta`), i.e. it
      # takes POST-activation beta. FlashKDA takes the logits and applies
      # sigmoid internally. This is the single easiest thing to get wrong.
      beta=np.ascontiguousarray(
          kda_case._sigmoid(inp["beta_logits"].astype(np.float64))
          .astype(np.float32).transpose(2, 0, 1)),
      a_log=inp["a_log"],
      # delta_time_bias is flattened to [H*K] and reshaped to
      # [H,1,1,K] internally (reference.py:44), so row-major [H,K] is correct.
      delta_time_bias=np.ascontiguousarray(inp["dt_bias"].reshape(-1)),
      scale=float(case.head_dim ** -0.5),
      use_qk_l2norm=True,
      use_gate_in_kernel=True,
      lower_bound=case.lower_bound,
      output_final_state=True,
  )

  if case.is_varlen:
    # 1-indexed segment IDs, 0 = padding. Every token here is valid.
    seg = np.zeros((1, case.total_tokens), np.int32)
    cu = case.cu_seqlens
    for i in range(len(cu) - 1):
      seg[0, int(cu[i]):int(cu[i + 1])] = i + 1
    args["segment_ids"] = seg
    args["max_num_segments"] = case.num_states
    # [N,H,K,V] -> [B=1,N,H,K,V]
    state_5d = inp["initial_state"][None]
  else:
    args["segment_ids"] = None
    args["max_num_segments"] = None
    # Fixed-length: one state per batch row. Mosaic requires N == 1 here
    # (pallas_mosaic_tpu.py), so [N=B,H,K,V] -> [B,1,H,K,V].
    state_5d = inp["initial_state"][:, None]

  args["initial_state"] = state_5d if bool(inp["has_initial_state"]) else None
  return args


def from_tokamax(case: kda_case.Case, output, final_state):
  """tokamax [H,B,T,V] / [B,N,H,K,V] -> canonical [B,T,H,V] / [N,H,K,V]."""
  out = np.asarray(output).transpose(1, 2, 0, 3)
  st = np.asarray(final_state)
  st = st[0] if case.is_varlen else st[:, 0]
  return np.ascontiguousarray(out), np.ascontiguousarray(st)


def invoke(case: kda_case.Case, args: dict, impl: str, dtype: str):
  """Cast to jax arrays and call the op. Returns canonical-layout numpy.

  Split out from `run_case` so tests can inject a mutated `args` dict --
  jaxtyping rejects numpy inputs, so callers must not skip this step.
  """
  import jax
  import jax.numpy as jnp
  from tokamax._src.ops.experimental.kda import api as kda_api

  jdt = getattr(jnp, dtype)
  arrays = {
      k: (jnp.asarray(v, jdt if k in _CAST else None)
          if isinstance(v, np.ndarray) else v)
      for k, v in args.items()
  }
  output, final_state = kda_api.kimi_delta_attention(
      arrays.pop("query"), arrays.pop("key"), arrays.pop("value"),
      arrays.pop("gate"), arrays.pop("beta"),
      implementation=impl, **arrays)
  jax.block_until_ready((output, final_state))
  return from_tokamax(case, output, final_state)


def run_case(name: str, impl: str, dtype: str, indir: str, outdir: str) -> None:
  case = kda_case.CASES[name]
  inp = kda_case.load_inputs(os.path.join(indir, f"in_{name}.npz"))
  out, st = invoke(case, to_tokamax(case, inp), impl, dtype)
  tag = "tpu" if impl == "mosaic" else "tpuref"
  np.savez(os.path.join(outdir, f"{tag}_{name}.npz"),
           output=out.astype(np.float32), final_state=st.astype(np.float32))
  print(f"{tag}_{name}: out{out.shape} state{st.shape} "
        f"dtype={dtype} impl={impl}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--impl", default="xla", choices=["xla", "mosaic"])
  p.add_argument("--dtype", default="float32",
                 choices=["bfloat16", "float32"])
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  args = p.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  names = list(kda_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    try:
      run_case(name, args.impl, args.dtype, args.indir, args.outdir)
    except NotImplementedError as e:
      # Mosaic rejects shapes it cannot tile; that is a real result, not a
      # harness failure, so record it rather than aborting the sweep.
      print(f"SKIP {name}: NotImplementedError: {e}")


if __name__ == "__main__":
  main()
