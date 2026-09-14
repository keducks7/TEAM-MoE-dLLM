#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATASET="${DATASET:-humaneval}"
case "$DATASET" in humaneval|mbpp|gsm8k|math) ;; *) echo 'Invalid DATASET' >&2; exit 1;; esac
LOG_DIR="${LOG_DIR:-$PWD/outputs/console_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE=$(mktemp "$LOG_DIR/${DATASET}_coldR${COLD_REUSE_INTERVAL:-0}_hotP${HOT_PRUNE_COUNT:-0}_protect${HOT_PROTECT_TOP1:-1}_$(date +%Y%m%d_%H%M%S).XXXXXX.log")
echo "Console log: $LOG_FILE"
{
    echo "Python: $PYTHON_BIN"
    echo "Dataset: $DATASET; GPUs: $CUDA_VISIBLE_DEVICES"
    echo "Cold interval: ${COLD_REUSE_INTERVAL:-0}; hot prune: ${HOT_PRUNE_COUNT:-0}; top1 protection: ${HOT_PROTECT_TOP1:-1}"
    "$PYTHON_BIN" -u run.py "configs/sdar/team/interventions/eval_sdar_hf_${DATASET}_interventions.py" --debug
} 2>&1 | tee "$LOG_FILE"
