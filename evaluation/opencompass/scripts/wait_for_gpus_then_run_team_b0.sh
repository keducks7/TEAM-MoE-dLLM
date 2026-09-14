#!/usr/bin/env bash

# Wait until the selected physical GPUs are stably idle, then run the paired
# TEAM B0 Hook-Off/Hook-On seed check.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/team_b0_hook_seed_check.sh"

GPU_IDS="${GPU_IDS:-0,1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-1024}"
MAX_IDLE_UTILIZATION="${MAX_IDLE_UTILIZATION:-5}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found." >&2
    exit 1
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "TEAM runner script was not found: ${RUN_SCRIPT}" >&2
    exit 1
fi

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ && \
        "${STABLE_CHECKS}" =~ ^[1-9][0-9]*$ && \
        "${MAX_IDLE_MEMORY_MIB}" =~ ^[0-9]+$ && \
        "${MAX_IDLE_UTILIZATION}" =~ ^[0-9]+$ ]]; then
    echo "Polling and threshold settings must be non-negative integers." >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU index." >&2
    exit 1
fi

for gpu_id in "${GPU_ARRAY[@]}"; do
    if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid physical GPU index in GPU_IDS: ${gpu_id}" >&2
        exit 1
    fi
    if ! nvidia-smi -i "${gpu_id}" --query-gpu=index \
            --format=csv,noheader,nounits >/dev/null 2>&1; then
        echo "GPU ${gpu_id} is not visible to nvidia-smi." >&2
        exit 1
    fi
done

gpu_is_idle() {
    local gpu_id="$1"
    local stats
    local memory_used
    local utilization
    local compute_pids

    stats="$(nvidia-smi -i "${gpu_id}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null)" || return 1

    IFS=',' read -r memory_used utilization <<< "${stats}"
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"

    if ! [[ "${memory_used}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    compute_pids="$(nvidia-smi -i "${gpu_id}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null || true)"
    compute_pids="$(printf '%s' "${compute_pids}" | tr -d '[:space:]')"

    printf 'GPU %s: memory=%s MiB, utilization=%s%%, compute_pids=%s\n' \
        "${gpu_id}" "${memory_used}" "${utilization}" \
        "${compute_pids:-none}"

    [[ -z "${compute_pids}" && \
       "${memory_used}" -le "${MAX_IDLE_MEMORY_MIB}" && \
       "${utilization}" -le "${MAX_IDLE_UTILIZATION}" ]]
}

consecutive_idle=0

echo "Waiting for physical GPUs: ${GPU_IDS}"
echo "Idle rule: no compute PID, memory <= ${MAX_IDLE_MEMORY_MIB} MiB, utilization <= ${MAX_IDLE_UTILIZATION}%"
echo "Required consecutive checks: ${STABLE_CHECKS}; polling interval: ${POLL_SECONDS}s"

while true; do
    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPUs"
    all_idle=true

    for gpu_id in "${GPU_ARRAY[@]}"; do
        if ! gpu_is_idle "${gpu_id}"; then
            all_idle=false
        fi
    done

    if [[ "${all_idle}" == true ]]; then
        consecutive_idle=$((consecutive_idle + 1))
        echo "Idle check passed (${consecutive_idle}/${STABLE_CHECKS})."
    else
        consecutive_idle=0
        echo "At least one requested GPU is busy; stable counter reset."
    fi

    if [[ "${consecutive_idle}" -ge "${STABLE_CHECKS}" ]]; then
        break
    fi

    sleep "${POLL_SECONDS}"
done

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs ${GPU_IDS} are stably idle. Starting TEAM B0 runs."

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
exec bash "${RUN_SCRIPT}"
