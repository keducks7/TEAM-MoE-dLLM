#!/usr/bin/env bash
# Diagnostic only: five replay forwards per matched state, no speed/ACC claim.
set -euo pipefail
if [[ "${HOOK_OFF:-0}" == 1 ]]; then
    echo 'Frontier probes need observation; use vanilla_dependency_motivation.sh for Hook Off.' >&2
    exit 2
fi
export PROBE=1 PROBE_KIND=frontier
export DEPENDENCY_LAYERS="${DEPENDENCY_LAYERS:-24}"
export PROBE_LAYER="${PROBE_LAYER:-24}"
export SAMPLES="${SAMPLES:-8}"
export MODEL_GPUS="${MODEL_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
bash "$(dirname "$0")/vanilla_dependency_motivation.sh"
