#!/usr/bin/env bash
# Explicit eval-only reuse: never launch inference.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 ]]; then
    echo "Usage: bash scripts/team_eval_existing.sh SAVED_CONFIG EXPERIMENT_DIR" >&2
    echo "EXPERIMENT_DIR must contain predictions/ and configs/. Use absolute paths." >&2
    exit 2
fi
CONFIG_PATH="$1"
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }
EXPERIMENT_DIR=$(cd "$2" && pwd)
[[ -d "$EXPERIMENT_DIR/predictions" ]] || { echo 'Missing predictions directory' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EXPERIMENT_DIR/logs"
LOG_FILE=$(mktemp "$EXPERIMENT_DIR/logs/eval_only_$(date +%Y%m%d_%H%M%S).XXXXXX.log")
echo "Eval-only log: $LOG_FILE"
"$PYTHON_BIN" -u run.py "$CONFIG_PATH" --mode eval \
    --work-dir "$(dirname "$EXPERIMENT_DIR")" \
    --reuse "$(basename "$EXPERIMENT_DIR")" --debug 2>&1 | tee "$LOG_FILE"
