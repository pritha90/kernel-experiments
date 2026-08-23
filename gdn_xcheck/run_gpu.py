# Copyright 2026. Apache-2.0.
"""GPU side of the cross-check: QwenLM/FlashQLA chunked gated delta rule.

FlashQLA implements only the delta-rule recurrence. tokamax's op is the *fused*
causal-conv1d + GDN used for serving, so everything ahead of the recurrence
(ragged depthwise conv with conv_state carry, silu, split, l2norm, the beta and
g parameterisations) is replicated here in torch and then handed to FlashQLA.

That split is deliberate: `--ref` runs the same preprocessed tensors through a
naive fp32 scan, so a mismatch tells you whether the preprocessing or the
kernel is at fault before you ever compare against the TPU.

Usage:
  python run_gpu.py --indir artifacts --outdir artifacts
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import gdn_case

_CAST = ("qkv", "conv_weight", "conv_bias", "conv_state")


def ragged_causal_conv1d(
    x: torch.Tensor,          # [T, C] pre-conv
    conv_state: torch.Tensor,  # [num_blocks, K-1, C]
    weight: torch.Tensor,      # [C, 1, K]
    bias: torch.Tensor | None,  # [C]
    cu: np.ndarray,            # [num_seqs+1]
    state_indices: np.ndarray,  # [max_reqs]
    has_initial_state: np.ndarray,  # [num_seqs] bool
    kernel_size: int,
):
  """Per-sequence depthwise causal conv1d + conv_state update.

  Mirrors `ragged_conv1d_mixed_prefill`: each sequence is convolved with its
  own conv_state as left context (zeroed when the slot has no live history),
  and the new state is the last K-1 *pre-conv* tokens.
  """
  c = x.shape[-1]
  out = torch.empty_like(x)
  new_conv_state = conv_state.clone()
  w = weight[:, 0, :].to(torch.float32)  # [C, K]

  for i in range(len(cu) - 1):
    s, e = int(cu[i]), int(cu[i + 1])
    length = e - s
    if length <= 0:
      continue
    slot = int(state_indices[i])
    left = conv_state[slot].to(torch.float32)
    if not bool(has_initial_state[i]):
      left = torch.zeros_like(left)

    seq = x[s:e].to(torch.float32)
    padded = torch.cat([left, seq], dim=0)            # [K-1+L, C]
    acc = torch.zeros((length, c), dtype=torch.float32, device=x.device)
    for j in range(kernel_size):
      acc += padded[j : j + length] * w[:, j]
    if bias is not None:
      acc += bias.to(torch.float32)
    out[s:e] = acc.to(x.dtype)

    # New state: tail of [left ; seq], last K-1 pre-conv tokens.
    tail = torch.cat([left.to(x.dtype), x[s:e]], dim=0)[-(kernel_size - 1):]
    new_conv_state[slot] = tail

  return out, new_conv_state


def preprocess(case: gdn_case.Case, inp: dict, device, dtype):
  """conv -> silu -> split -> l2norm-ready q/k/v, plus beta/g/initial_state."""
  t = lambda a, dt=None: torch.as_tensor(  # noqa: E731
      np.asarray(a), device=device, dtype=dt)

  cu_full = inp["query_start_loc"]
  n_seqs = case.num_seqs
  cu = cu_full[: n_seqs + 1]
  state_indices = inp["state_indices"]
  max_reqs = len(state_indices)
  # tokamax derives this over *all* max_reqs slots, padded ones included.
  query_lens_full = cu_full[1 : max_reqs + 1] - cu_full[:max_reqs]
  has_init_full = (inp["seq_lens"][:max_reqs] - query_lens_full) > 0
  has_init = has_init_full[:n_seqs]

  x = t(inp["qkv"], dtype)
  conv_state = t(inp["conv_state"], dtype)
  weight = t(inp["conv_weight"], dtype)
  bias = t(inp["conv_bias"], dtype)

  xc, new_conv_state = ragged_causal_conv1d(
      x, conv_state, weight, bias, cu, state_indices, has_init,
      case.kernel_size,
  )
  xc = F.silu(xc.to(torch.float32)).to(dtype)

  key_dim = case.n_kq * case.d_k
  q = xc[:, :key_dim].reshape(-1, case.n_kq, case.d_k)
  k = xc[:, key_dim : 2 * key_dim].reshape(-1, case.n_kq, case.d_k)
  v = xc[:, 2 * key_dim :].reshape(-1, case.n_v, case.d_v)

  # tokamax expands q/k with jnp.repeat(axis=head) == repeat_interleave.
  rep = case.n_v // case.n_kq
  if rep > 1:
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)

  b = t(inp["b"], torch.float32)
  a = t(inp["a"], torch.float32)
  a_log = t(inp["a_log"], torch.float32)
  dt_bias = t(inp["dt_bias"], torch.float32)

  beta = torch.sigmoid(b)
  g = -torch.exp(a_log) * F.softplus(a + dt_bias)

  rec = t(inp["recurrent_state"], torch.float32)
  h0 = rec[torch.as_tensor(state_indices[:n_seqs].astype(np.int64),
                           device=device)].clone()
  h0[~torch.as_tensor(has_init, device=device)] = 0.0

  # `ragged_gated_delta_rule_ref` writes the masked initial states back into
  # the slot table before the scan, so slots with no live history come out
  # zeroed even if they are padding the scan never visits. Replicate that or
  # the padded cases disagree on untouched slots.
  rec_base = rec.clone()
  stale = torch.as_tensor(
      state_indices[~has_init_full].astype(np.int64), device=device)
  rec_base[stale] = 0.0

  return dict(
      q=q[None], k=k[None], v=v[None],
      beta=beta[None], g=g[None],
      h0=h0,
      cu=torch.as_tensor(cu.astype(np.int64), device=device),
      new_conv_state=new_conv_state,
      full_recurrent_state=rec_base,
      state_indices=state_indices,
      n_seqs=n_seqs,
  )


def l2norm(x, eps=1e-6):
  xf = x.to(torch.float32)
  return (x * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + eps)).to(x.dtype)


def naive_gdr_fp32(q, k, v, g, beta, h0, cu, scale):
  """Token-by-token fp32 scan over the already-preprocessed tensors."""
  q, k, v = q[0].float(), k[0].float(), v[0].float()
  g, beta = g[0].float(), beta[0].float()
  q, k = l2norm(q), l2norm(k)
  q = q * scale
  hv, dk, dv = v.shape[1], q.shape[-1], v.shape[-1]

  o = torch.zeros_like(v)
  finals = torch.zeros((len(cu) - 1, hv, dk, dv), dtype=torch.float32,
                       device=q.device)
  for i in range(len(cu) - 1):
    s, e = int(cu[i]), int(cu[i + 1])
    S = h0[i].clone()  # [HV, K, V]
    for tk in range(s, e):
      eg = torch.exp(g[tk])[:, None]                       # [HV,1]
      kt, vt, qt = k[tk], v[tk], q[tk]                     # [HV,K],[HV,V]
      kS = torch.einsum("hd,hdm->hm", kt, S)
      v_new = beta[tk][:, None] * (vt - eg * kS)
      S = S * eg[:, :, None] + torch.einsum("hd,hm->hdm", kt, v_new)
      o[tk] = torch.einsum("hd,hdm->hm", qt, S)
    finals[i] = S
  return o[None], finals


def run_case(case: gdn_case.Case, inp: dict, device, dtype, *,
             want_ref: bool, auto_cp: bool, ref_only: bool = False):
  pre = preprocess(case, inp, device, dtype)
  scale = case.d_k ** -0.5

  idx = torch.as_tensor(
      pre["state_indices"][: pre["n_seqs"]].astype(np.int64), device=device)
  res = {"conv_state": pre["new_conv_state"].float().cpu().numpy()}

  if not ref_only:
    from flash_qla import chunk_gated_delta_rule  # noqa: PLC0415 - CUDA-only

    o, final_state = chunk_gated_delta_rule(
        q=pre["q"].contiguous(),
        k=pre["k"].contiguous(),
        v=pre["v"].contiguous(),
        g=pre["g"].contiguous(),
        beta=pre["beta"].contiguous(),
        scale=scale,
        initial_state=pre["h0"].contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=pre["cu"],
        auto_cp=auto_cp,
    )

    # Scatter the per-sequence final states back into the full slot table so
    # the shapes line up with what tokamax returns.
    rec_out = pre["full_recurrent_state"].clone()
    rec_out[idx] = final_state.to(torch.float32)
    res["output"] = o[0].reshape(case.num_tokens, -1).float().cpu().numpy()
    res["recurrent_state"] = rec_out.float().cpu().numpy()

  if want_ref:
    o_r, fs_r = naive_gdr_fp32(
        pre["q"], pre["k"], pre["v"], pre["g"], pre["beta"],
        pre["h0"], pre["cu"], scale,
    )
    rec_r = pre["full_recurrent_state"].clone()
    rec_r[idx] = fs_r
    res["ref_output"] = o_r[0].reshape(case.num_tokens, -1).cpu().numpy()
    res["ref_recurrent_state"] = rec_r.cpu().numpy()
    res["ref_conv_state"] = res["conv_state"]
  return res


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--case", default="all", choices=["all", *gdn_case.CASES])
  p.add_argument("--indir", default="artifacts")
  p.add_argument("--outdir", default="artifacts")
  p.add_argument("--dtype", default="bfloat16",
                 choices=["bfloat16", "float16", "float32"])
  p.add_argument("--ref", action="store_true",
                 help="also run a naive fp32 torch scan (slow, O(T) python)")
  p.add_argument("--auto-cp", action="store_true",
                 help="leave FlashQLA's intra-card CP on (off by default for "
                      "run-to-run determinism)")
  p.add_argument("--ref-only", action="store_true",
                 help="skip FlashQLA and emit only the fp32 scan. Runs on CPU "
                      "and is how you validate the torch preprocessing "
                      "against tokamax's reference without a GPU.")
  p.add_argument("--device", default=None,
                 help="torch device (default: cpu with --ref-only, else cuda)")
  args = p.parse_args()

  device = torch.device(
      args.device or ("cpu" if args.ref_only else "cuda"))
  dtype = getattr(torch, args.dtype)
  if device.type == "cuda":
    print(f"torch {torch.__version__} {torch.cuda.get_device_name()} "
          f"sm{''.join(map(str, torch.cuda.get_device_capability()))}")
  else:
    print(f"torch {torch.__version__} device={device}")
    args.ref = True

  os.makedirs(args.outdir, exist_ok=True)
  names = list(gdn_case.CASES) if args.case == "all" else [args.case]
  for name in names:
    case = gdn_case.CASES[name]
    inp = gdn_case.load_inputs(os.path.join(args.indir, f"in_{name}.npz"))
    try:
      res = run_case(case, inp, device, dtype, want_ref=args.ref,
                     auto_cp=args.auto_cp, ref_only=args.ref_only)
    except Exception as e:  # noqa: BLE001
      print(f"[{name}] FAILED: {type(e).__name__}: {e}")
      continue
    dst = os.path.join(
        args.outdir, f"{'gpuref' if args.ref_only else 'gpu'}_{name}.npz")
    np.savez(dst, **res)
    extra = ""
    if "output" in res and "ref_output" in res:
      d = np.abs(res["output"] - res["ref_output"]).max()
      extra = f"  (kernel vs fp32 ref, max|d| = {d:.3e})"
    print(f"[{name}] ok -> {dst}{extra}")


if __name__ == "__main__":
  main()
