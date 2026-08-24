# Copyright 2026. Apache-2.0.
"""Run the tokamax KDA op (PR #1103) on the canonical inputs.

Two implementations are exposed:
  --impl mosaic   the Pallas/Mosaic TPU kernel  (requires a TPU)
  --impl xla      the token-by-token JAX reference (runs anywhere, incl. CPU)

The `xla` path is what makes the harness checkable without hardware: it is
tokamax's own statement of KDA semantics, so agreement between it and the
arbiter validates every layout/activation conversion below before either
kernel is ever run.

`--backward` additionally takes the VJP. tokamax returns gradients for
exactly the eight inputs FLA's `ChunkKDAFunction.backward` returns
(pallas_mosaic_tpu.py `grads`: query, key, value, gate, beta, a_log,
delta_time_bias, initial_state), so the two are directly comparable.

Usage:
  python run_tpu.py --case all --impl xla    --dtype float32   # stage 1
  python run_tpu.py --case all --impl mosaic --dtype bfloat16 --backward
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import kda_case


# Activations cast to the run dtype; states and per-head params stay fp32
# because both backends accumulate the recurrence in fp32.
_CAST = ("query", "key", "value", "gate", "beta")

# tokamax kwarg names for the eight differentiable inputs, in the same order
# as kda_case.DIFFERENTIABLE.
_PRIMALS = ("query", "key", "value", "gate", "beta", "a_log",
            "delta_time_bias", "initial_state")


def to_tokamax(case: kda_case.Case, inp: dict[str, np.ndarray]) -> dict:
  """Canonical [B,T,H,D] -> tokamax's head-first [H,B,T,D] convention.

  This is the only layout difference from FLA, which is already canonical.
  """
  hbtd = lambda x: np.ascontiguousarray(x.transpose(2, 0, 1, 3))

  args = dict(
      query=hbtd(inp["q"]),
      key=hbtd(inp["k"]),
      value=hbtd(inp["v"]),
      gate=hbtd(inp["g"]),
      # POST-activation beta, validated in [0,1] (base.py `_validate_beta`).
      # FLA is called with use_beta_sigmoid_in_kernel=False so it takes the
      # same form -- which also makes `dbeta` the same derivative on both
      # sides rather than differing by a sigmoid'.
      beta=np.ascontiguousarray(inp["beta"].transpose(2, 0, 1)),
      a_log=inp["a_log"],
      # delta_time_bias is flattened to [H*K] and reshaped to [H,1,1,K]
      # internally (reference.py:44), so row-major [H,K] is correct. FLA
      # reshapes the same flat buffer to [H,K] too (gate.py).
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
    state_5d = inp["initial_state"][None]           # [N,H,K,V] -> [1,N,H,K,V]
  else:
    args["segment_ids"] = None
    args["max_num_segments"] = None
    # Fixed-length: one state per batch row. Mosaic requires N == 1 here
    # (pallas_mosaic_tpu.py), so [N=B,H,K,V] -> [B,1,H,K,V].
    state_5d = inp["initial_state"][:, None]

  args["initial_state"] = state_5d if bool(inp["has_initial_state"]) else None
  return args


def _state_to_canonical(case, st):
  st = np.asarray(st)
  return np.ascontiguousarray(st[0] if case.is_varlen else st[:, 0])


def from_tokamax(case: kda_case.Case, output, final_state):
  """tokamax [H,B,T,V] / [B,N,H,K,V] -> canonical [B,T,H,V] / [N,H,K,V]."""
  out = np.ascontiguousarray(np.asarray(output).transpose(1, 2, 0, 3))
  return out, _state_to_canonical(case, final_state)


def _to_jax(args: dict, dtype: str) -> dict:
  import jax.numpy as jnp

  jdt = getattr(jnp, dtype)
  return {
      k: (jnp.asarray(v, jdt if k in _CAST else None)
          if isinstance(v, np.ndarray) else v)
      for k, v in args.items()
  }


def _call(kda_api, kw: dict, impl: str):
  return kda_api.kimi_delta_attention(
      kw.pop("query"), kw.pop("key"), kw.pop("value"),
      kw.pop("gate"), kw.pop("beta"), implementation=impl, **kw)


def invoke(case: kda_case.Case, args: dict, impl: str, dtype: str):
  """Cast to jax arrays and call the op. Returns canonical-layout numpy.

  Split out from `run_case` so tests can inject a mutated `args` dict --
  jaxtyping rejects numpy inputs, so callers must not skip this step.
  """
  import jax
  from tokamax._src.ops.experimental.kda import api as kda_api

  output, final_state = _call(kda_api, _to_jax(args, dtype), impl)
  jax.block_until_ready((output, final_state))
  return from_tokamax(case, output, final_state)


def invoke_vjp(case: kda_case.Case, args: dict, inp: dict, impl: str,
               dtype: str) -> dict:
  """Forward + VJP. Returns canonical-layout output, final_state, and grads.

  Cotangents come from the shared `.npz` (`do`, `dht`), so the GPU side is
  seeded with bit-identical values and the gradients can be compared directly
  rather than only in distribution.
  """
  import jax
  from tokamax._src.ops.experimental.kda import api as kda_api

  arrays = _to_jax(args, dtype)
  # initial_state is None for stateless cases, and jax.vjp cannot
  # differentiate a None primal -- drop it from the primal tuple and let it
  # ride along as a static kwarg instead.
  keys = [k for k in _PRIMALS if arrays.get(k) is not None]
  primals = tuple(arrays.pop(k) for k in keys)
  static = arrays

  def fn(*p):
    return _call(kda_api, {**dict(zip(keys, p)), **static}, impl)

  (output, final_state), vjp = jax.vjp(fn, *primals)
  # jax.vjp requires cotangent dtypes to match the primal outputs exactly.
  do = jax.numpy.asarray(inp["do"].transpose(2, 0, 1, 3), output.dtype)
  dht = inp["dht"][None] if case.is_varlen else inp["dht"][:, None]
  grads = vjp((do, jax.numpy.asarray(dht, final_state.dtype)))
  jax.block_until_ready(grads)

  out, st = from_tokamax(case, output, final_state)
  res = {"output": out, "final_state": st}
  # Back to canonical layout, mirroring `to_tokamax` exactly.
  back = {
      "query": lambda x: np.asarray(x).transpose(1, 2, 0, 3),
      "key": lambda x: np.asarray(x).transpose(1, 2, 0, 3),
      "value": lambda x: np.asarray(x).transpose(1, 2, 0, 3),
      "gate": lambda x: np.asarray(x).transpose(1, 2, 0, 3),
      "beta": lambda x: np.asarray(x).transpose(1, 2, 0),
      "a_log": np.asarray,
      "delta_time_bias": np.asarray,
      "initial_state": lambda x: _state_to_canonical(case, x),
  }
  names = dict(zip(_PRIMALS, kda_case.DIFFERENTIABLE))
  for key, gr in zip(keys, grads):
    res[f"d{names[key]}"] = np.ascontiguousarray(back[key](gr))
  return res


def run_case(name: str, impl: str, dtype: str, indir: str, outdir: str,
             backward: bool) -> None:
  case = kda_case.CASES[name]
  inp = kda_case.load_inputs(os.path.join(indir, f"in_{name}.npz"))
  args = to_tokamax(case, inp)
  if backward:
    res = invoke_vjp(case, args, inp, impl, dtype)
  else:
    out, st = invoke(case, args, impl, dtype)
    res = {"output": out, "final_state": st}

  tag = "tpu" if impl == "mosaic" else "tpuref"
  np.savez(os.path.join(outdir, f"{tag}_{name}.npz"),
           **{k: np.asarray(v, np.float32) for k, v in res.items()})
  print(f"{tag}_{name}: {' '.join(sorted(res))}  dtype={dtype} impl={impl}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--impl", default="xla", choices=["xla", "mosaic"])
  p.add_argument("--dtype", default="float32",
                 choices=["bfloat16", "float32"])
  p.add_argument("--backward", action="store_true",
                 help="also take the VJP and emit the eight gradients")
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  args = p.parse_args()

  os.makedirs(args.outdir, exist_ok=True)
  names = list(kda_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    try:
      run_case(name, args.impl, args.dtype, args.indir, args.outdir,
               args.backward)
    except NotImplementedError as e:
      # Mosaic rejects shapes it cannot tile; that is a real result, not a
      # harness failure, so record it rather than aborting the sweep.
      print(f"SKIP {name}: NotImplementedError: {e}")


if __name__ == "__main__":
  main()
