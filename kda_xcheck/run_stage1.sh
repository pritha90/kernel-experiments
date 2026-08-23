#!/usr/bin/env bash
# Stage 1 end-to-end: generate inputs + arbiter, run tokamax's fp32 XLA
# reference, compare, then run the negative control. CPU only -- no TPU and
# no GPU needed.
#
#   TOKAMAX=~/src/tokamax-pr1103 ./run_stage1.sh
set -euo pipefail

: "${TOKAMAX:?set TOKAMAX to a checkout of tokamax PR #1103}"
PY="${PYTHON:-python}"
cd "$(dirname "$0")"
export PYTHONPATH="$TOKAMAX${PYTHONPATH:+:$PYTHONPATH}"

echo "== generating inputs + float64 arbiter"
"$PY" kda_case.py --case all --ref

echo && echo "== tokamax implementation=xla, float32"
"$PY" run_tpu.py --case all --impl xla --dtype float32

echo && echo "== stage 1: tokamax xla vs arbiter"
"$PY" compare.py --stage semantics

echo && echo "== negative control: injected conversion bugs must be caught"
"$PY" test_conversions.py
