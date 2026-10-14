#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/human-eval${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_GPUS="${MODEL_GPUS:-2}"
export DATASET="${DATASET:-humaneval}"
export MODE="${MODE:-acc}"
export SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"
export SAMPLES="${SAMPLES:-all}" SAMPLE_START="${SAMPLE_START:-32}"
METHOD="${METHOD:-protected}"
case "$METHOD" in
    team|b0|none) export TEAM_POOL_STRATEGY=none ;;
    reference|protected|protected_hot) export TEAM_POOL_STRATEGY="$METHOD" ;;
    *) echo 'METHOD must be team, reference, protected or protected_hot' >&2; exit 2 ;;
esac
case "$DATASET" in humaneval|mbpp|gsm8k|math) ;; *) echo 'Invalid DATASET' >&2; exit 2;; esac
case "$MODE" in acc|metrics) ;; *) echo 'MODE must be acc or metrics' >&2; exit 2;; esac
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_ROOT="${RUN_ROOT:-$PWD/outputs/team_frontier_pool}"
mkdir -p "$RUN_ROOT"
TASK_RUN_DIR=$(mktemp -d "$RUN_ROOT/${DATASET}_${TEAM_POOL_STRATEGY}_b${TEAM_POOL_BUDGET:-36}_${MODE}_seed${SEED}.XXXXXX")
export TEAM_WORK_DIR="$TASK_RUN_DIR/opencompass"
export TEAM_RECORD_DIR="$TASK_RUN_DIR/records"
echo "Run directory: $TASK_RUN_DIR"
"$PYTHON_BIN" -u scripts/run_team_frontier_pool.py \
    --config "configs/sdar/team/frontier_pool/eval_sdar_hf_${DATASET}_frontier_pool.py" \
    --manifest "$TASK_RUN_DIR/manifest.json" 2>&1 | tee "$TASK_RUN_DIR/console.log"
if [[ "${COLLECT_LATENCY:-1}" == 1 || "$MODE" == metrics ]]; then
    "$PYTHON_BIN" analysis/tools/calculate_team_metrics.py --metrics-dir "$TEAM_RECORD_DIR" \
        2>&1 | tee -a "$TASK_RUN_DIR/console.log"
fi
