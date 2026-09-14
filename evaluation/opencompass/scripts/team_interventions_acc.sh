#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/focus/bin/python}"
DATASET="${DATASET:-humaneval}"
case "$DATASET" in humaneval|mbpp|gsm8k|math) ;; *) echo 'Invalid DATASET' >&2; exit 1;; esac
"$PYTHON_BIN" run.py "configs/sdar/team/interventions/eval_sdar_hf_${DATASET}_interventions.py" --debug
