#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATASET="${DATASET:-humaneval}"
if [[ -z "${TRACE_DIR:-}" ]]; then
    export RUN_DIR="${RUN_DIR:-$PWD/analysis/records/dependency/${DATASET}_structure_$(date +%Y%m%d_%H%M%S)_$$}"
    export PROBE=0 HOOK_OFF=0 DRY_RUN=0
    export MODEL_GPUS="${MODEL_GPUS:-2}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export DEPENDENCY_LAYERS="${DEPENDENCY_LAYERS:-auto}"
    export SAMPLES="${SAMPLES:-32}"
    bash scripts/vanilla_dependency_motivation.sh
    TRACE_DIR="$RUN_DIR/traces"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/analysis/reports/decoding_expert_structure/${DATASET}_$(date +%Y%m%d_%H%M%S)_$$}"
"$PYTHON_BIN" analysis/tools/analyze_decoding_expert_structure.py \
    --trace-dir "$TRACE_DIR" --output-dir "$OUTPUT_DIR" \
    --budgets "${POOL_BUDGETS:-8,16,24,32}" \
    --edge-fraction "${EDGE_FRACTION:-0.2}" \
    --frontier-fraction "${FRONTIER_FRACTION:-0.25}" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS:-1000}" --seed "${SEED:-20261012}"
echo "Structure report: $OUTPUT_DIR/report.md"
