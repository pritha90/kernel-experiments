# Copyright 2026. Apache-2.0.
"""The shared float64 arbiter: forward *and* gradients.

`kda_case.reference()` is a numpy fp64 forward, but it is written as an
imperative loop with in-place writes, so it cannot be differentiated. This
module restates the same recurrence functionally in JAX with x64 enabled and
gets the eight gradients by `jax.vjp`.

The two forwards are independent implementations of the same equations, and
`compare.py --stage semantics` checks they agree (~1e-15). That check is what
licenses using the JAX version as the gradient reference: an autodiff
reference is only as trustworthy as the forward it differentiates.

Runs on CPU. JAX is required, a TPU is not -- so the gradient reference for
*both* backends is computable on a laptop.

  python arbiter.py --case all
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import kda_case

# Must precede any jax.numpy use. Without this the "float64" arbiter is
# silently float32 and every tolerance below is meaningless.
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _sigmoid(x):
  return jax.nn.sigmoid(x)


def _l2norm(x, eps: float = 1e-6):
  # rsqrt(sum(x*x) + eps), matching tokamax reference.py `_l2_normalize` and
  # FLA's l2norm_fwd. The eps is inside the sqrt, not added to the norm.
  return x * jax.lax.rsqrt((x * x).sum(-1, keepdims=True) + eps)


def _activate_gate(g, a_log, dt_bias, lower_bound):
  h, dk = g.shape[2], g.shape[3]
  gf = g + dt_bias.reshape(h, dk)
  a = jnp.exp(a_log)[None, None, :, None]
  return lower_bound * _sigmoid(a * gf)


def _seq_scan(q, k, v, g, beta, s0):
  """One (sequence, head). q,k,g:[T,K] v:[T,V] beta:[T] s0:[K,V]."""

  def step(s, x):
    qt, kt, vt, gt, bt = x
    s = s * jnp.exp(gt)[:, None]
    # Delta rule: correct the state toward v_t along the k_t direction.
    s = s + (bt * kt)[:, None] * (vt - kt @ s)[None, :]
    return s, qt @ s

  s, o = jax.lax.scan(step, s0, (q, k, v, g, beta))
  return o, s


# heads are axis 1 of [T,H,D] and axis 0 of the [H,K,V] state.
_over_heads = jax.vmap(_seq_scan, in_axes=(1, 1, 1, 1, 1, 0),
                       out_axes=(1, 0))


def forward(case: kda_case.Case, q, k, v, g, beta, a_log, dt_bias,
            initial_state):
  """Canonical-layout KDA. All args [B,T,H,*] except state [N,H,K,V]."""
  qn = _l2norm(q) * (case.head_dim ** -0.5)
  kn = _l2norm(k)
  gg = _activate_gate(g, a_log, dt_bias, case.lower_bound)

  cu = case.cu_seqlens
  outs, finals = [], []
  for n in range(len(cu) - 1):
    lo, hi = int(cu[n]), int(cu[n + 1])
    # Fixed-length puts one sequence per batch row; varlen packs them into
    # row 0. `cu_seqlens` is static (it comes from the Case), so these are
    # ordinary Python slices, not dynamic_slice.
    bi = 0 if case.is_varlen else n
    t0, t1 = (lo, hi) if case.is_varlen else (0, hi - lo)
    o, s = _over_heads(qn[bi, t0:t1], kn[bi, t0:t1], v[bi, t0:t1],
                       gg[bi, t0:t1], beta[bi, t0:t1], initial_state[n])
    outs.append(o)
    finals.append(s)

  out = (jnp.concatenate(outs, 0)[None] if case.is_varlen
         else jnp.stack(outs, 0))
  return out, jnp.stack(finals, 0)


def run(case: kda_case.Case, inp: dict[str, np.ndarray]) -> dict:
  """Forward + VJP in float64. Returns canonical-layout fp64 numpy."""
  f64 = lambda x: jnp.asarray(x, jnp.float64)
  primals = tuple(f64(inp[n]) for n in kda_case.DIFFERENTIABLE)
  fn = lambda *p: forward(case, *p)

  (out, final_state), vjp = jax.vjp(fn, *primals)
  grads = vjp((f64(inp["do"]), f64(inp["dht"])))

  res = {"output": np.asarray(out), "final_state": np.asarray(final_state)}
  for name, gr in zip(kda_case.DIFFERENTIABLE, grads):
    res[f"d{name}"] = np.asarray(gr)
  return res


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *kda_case.CASES])
  p.add_argument("--dir", default="artifacts")
  args = p.parse_args()

  os.makedirs(args.dir, exist_ok=True)
  for name in (list(kda_case.CASES) if args.case == "all" else [args.case]):
    case = kda_case.CASES[name]
    inp = kda_case.load_inputs(os.path.join(args.dir, f"in_{name}.npz"))
    t0 = time.time()
    res = run(case, inp)
    # Stored float64. The obvious saving -- store fp32, since every backend
    # compared against it is fp32-or-worse -- is wrong: `npref vs ref` is
    # fp64-vs-fp64, and fp32 storage injects ~2e-09 that swamps it.
    np.savez(os.path.join(args.dir, f"ref_{name}.npz"), **res)
    print(f"ref_{name}: {len(res)} tensors ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
  main()
