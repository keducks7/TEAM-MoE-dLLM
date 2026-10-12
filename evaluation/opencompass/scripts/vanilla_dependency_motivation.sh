#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/human-eval${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED="${SEED:-20261012}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATASET="${DATASET:-humaneval}"
MODEL_GPUS="${MODEL_GPUS:-1}"
# Conservative default for 2 x 40GB; single 80GB retains the old default.
if [[ "$MODEL_GPUS" == 2 ]]; then
    DEFAULT_GPU_MEMORY=37GiB
else
    DEFAULT_GPU_MEMORY=74GiB
fi
RUN_DIR="${RUN_DIR:-$PWD/analysis/records/dependency/${DATASET}_$(date +%Y%m%d_%H%M%S)_$$}"
OPTIONS=()
if [[ "${HOOK_OFF:-0}" == 1 ]]; then OPTIONS+=(--hook-off); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then OPTIONS+=(--dry-run); fi
if [[ "${PROBE:-0}" == 1 ]]; then
    OPTIONS+=(--probe --probe-layer "${PROBE_LAYER:-24}" --probe-pool-size "${PROBE_POOL_SIZE:-12}"
        --probe-stride "${PROBE_STRIDE:-2}" --probe-max-steps "${PROBE_MAX_STEPS:-48}"
        --probe-mass-tolerance "${PROBE_MASS_TOLERANCE:-0.03}"
        --probe-confidence-tolerance "${PROBE_CONFIDENCE_TOLERANCE:-0.05}")
fi
"$PYTHON_BIN" scripts/run_dependency_motivation.py \
    --dataset "$DATASET" \
    --model-path "${MODEL_PATH:-/root/lkd/Models/SDAR-30B-A3B-Chat-b32}" \
    --output-dir "$RUN_DIR" --samples "${SAMPLES:-32}" \
    --sample-start "${SAMPLE_START:-0}" --layers "${DEPENDENCY_LAYERS:-auto}" \
    --max-blocks "${MAX_BLOCKS:-8}" --gen-length "${GEN_LENGTH:-4096}" \
    --seed "${SEED:-20261012}" --gpus "$MODEL_GPUS" \
    --memory "${GPU_MEMORY:-$DEFAULT_GPU_MEMORY}" "${OPTIONS[@]}"
if [[ "${HOOK_OFF:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
    "$PYTHON_BIN" analysis/tools/analyze_dependency_motivation.py \
        --trace-dir "$RUN_DIR/traces" --output-dir "$RUN_DIR/report"
    "$PYTHON_BIN" analysis/tools/analyze_dependency_drift.py \
        --trace-dir "$RUN_DIR/traces" --output-dir "$RUN_DIR/drift"
    if [[ "${PROBE:-0}" == 1 ]]; then
        "$PYTHON_BIN" analysis/tools/analyze_dependency_probes.py \
            --probe-dir "$RUN_DIR/probes" --output-dir "$RUN_DIR/probe_report"
    fi
fi
echo "Run directory: $RUN_DIR"
