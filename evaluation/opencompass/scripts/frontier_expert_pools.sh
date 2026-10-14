#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATASET="${DATASET:-humaneval}"
: "${TRACE_DIR:?Set TRACE_DIR to an existing Vanilla dependency traces directory}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/analysis/reports/frontier_expert_pools/${DATASET}_$(date +%Y%m%d_%H%M%S)_$$}"
"$PYTHON_BIN" analysis/tools/analyze_frontier_expert_pools.py \
    --trace-dir "$TRACE_DIR" --output-dir "$OUTPUT_DIR" \
    --budgets "${POOL_BUDGETS:-8,16,24,32}" \
    --hot-coverage-target "${HOT_COVERAGE_TARGET:-0.9}" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS:-1000}" --seed "${SEED:-20261012}"
echo "Expert pool report: $OUTPUT_DIR/report.md"
