#!/usr/bin/env bash
# Stage 1 end-to-end: generate inputs, both fp64 forwards, the fp64 VJP
# arbiter, then tokamax's XLA reference and FLA's naive reference (forward
# AND backward), compare, and run the negative control.
#
# CPU only -- no TPU and no GPU needed. Triton is not needed either; the FLA
# reference is loaded straight from the source tree if triton is missing.
#
#   TOKAMAX=~/src/tokamax-pr1103 FLA_ROOT=~/src/flash-linear-attention \
#     ./run_stage1.sh
set -euo pipefail

: "${TOKAMAX:?set TOKAMAX to a checkout of tokamax PR #1103}"
: "${FLA_ROOT:?set FLA_ROOT to a checkout of fla-org/flash-linear-attention}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")"
export FLA_ROOT
export PYTHONPATH="$TOKAMAX:$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "== generating inputs + numpy float64 forward"
"$PY" kda_case.py --case all --ref

echo && echo "== float64 arbiter (JAX, forward + VJP)"
"$PY" arbiter.py --case all

echo && echo "== tokamax implementation=xla, float32, forward + VJP"
"$PY" run_tpu.py --case all --impl xla --dtype float32 --backward

echo && echo "== FLA naive_recurrent_kda, float32, forward + autograd"
"$PY" run_gpu.py --case all --impl naive --dtype float32 --backward

echo && echo "== stage 1: every reference vs the arbiter"
"$PY" compare.py --stage semantics

echo && echo "== negative control: injected conversion bugs must be caught"
"$PY" test_conversions.py
