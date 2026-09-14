#!/usr/bin/env bash

# Paired determinism check for HumanEval with original TEAM (cold budget = 0).
# The two runs use the same process-level seed and differ only in whether the
# lightweight APF/TPF/APT metric hook is enabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENCOMPASS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/focus/bin/python}"
SEED="${SEED:-20260907}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

OFF_CONFIG="configs/sdar/team/cold_budget/eval_sdar_hf_humaneval_cold_budget.py"
ON_CONFIG="configs/sdar/team/cold_budget/metrics/eval_sdar_hf_humaneval_cold_budget_metrics.py"

OFF_WORK_ROOT="./outputs/seed_check/team_b0_hook_off_seed${SEED}"
ON_WORK_ROOT="./analysis/records/opencompass_runs/seed_check/team_b0_hook_on_seed${SEED}"
METRICS_DIR="./analysis/records/metrics/humaneval_team_cold_probability_sum_b0_metrics_v1"

cd "${OPENCOMPASS_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ -n "$(find "${METRICS_DIR}" -type f -name '*.jsonl' -print -quit 2>/dev/null || true)" ]]; then
    echo "Refusing to append duplicate records to: ${METRICS_DIR}" >&2
    echo "Move the existing B0 metric directory aside, then run this script again." >&2
    exit 2
fi

export PYTHONPATH="${OPENCOMPASS_DIR}:${OPENCOMPASS_DIR}/human-eval${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="${SEED}"
export CUDA_VISIBLE_DEVICES
export COLD_EXPERT_BUDGET=0
export COLD_EXPERT_SELECTION=probability_sum

run_seeded() {
    local config_path="$1"
    local work_root="$2"

    "${PYTHON_BIN}" - "${config_path}" "${work_root}" "${SEED}" <<'PY'
import random
import sys

import numpy as np
import torch

config_path, work_root, seed_text = sys.argv[1:]
seed = int(seed_text)

random.seed(seed)
np.random.seed(seed % (2**32))
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --debug is required here: LocalRunner then executes inference in this seeded
# Python process instead of starting a fresh, unseeded worker subprocess.
sys.argv = [
    'run.py',
    config_path,
    '--debug',
    '-w',
    work_root,
]

from opencompass.cli.main import main

main()
PY
}

echo "============================================================"
echo "TEAM B0 paired Hook check"
echo "Seed       : ${SEED}"
echo "CUDA GPUs  : ${CUDA_VISIBLE_DEVICES}"
echo "Python     : ${PYTHON_BIN}"
echo "============================================================"

echo
echo "[1/2] Running TEAM B0 with metric hook OFF"
run_seeded "${OFF_CONFIG}" "${OFF_WORK_ROOT}"

echo
echo "[2/2] Running TEAM B0 with metric hook ON"
run_seeded "${ON_CONFIG}" "${ON_WORK_ROOT}"

echo
echo "[Metrics] Aggregating Hook-On APF/TPF/APT"
"${PYTHON_BIN}" analysis/tools/calculate_team_metrics.py \
    --metrics-dir "${METRICS_DIR}"

OFF_RESULT="$(find "${OFF_WORK_ROOT}" -type f -path '*/results/*/openai_humaneval.json' -print | sort | tail -n 1)"
ON_RESULT="$(find "${ON_WORK_ROOT}" -type f -path '*/results/*/openai_humaneval.json' -print | sort | tail -n 1)"

if [[ -z "${OFF_RESULT}" || -z "${ON_RESULT}" ]]; then
    echo >&2
    echo "Both runs finished, but one or both result JSON files were not found." >&2
    echo "Hook-Off work root: ${OFF_WORK_ROOT}" >&2
    echo "Hook-On work root : ${ON_WORK_ROOT}" >&2
    exit 3
fi

echo
echo "[Comparison] Hook-Off versus Hook-On"
"${PYTHON_BIN}" - "${OFF_RESULT}" "${ON_RESULT}" <<'PY'
import json
import sys

off_path, on_path = sys.argv[1:]
with open(off_path, encoding='utf-8') as file:
    off = json.load(file)
with open(on_path, encoding='utf-8') as file:
    on = json.load(file)


def correctness_by_example(result):
    values = {}
    for index, detail in enumerate(result.get('details', [])):
        key = detail.get('example_abbr', str(index))
        correct = detail.get('is_correct', detail.get('passed'))
        if isinstance(correct, list):
            correct = correct[0]
        values[key] = bool(correct)
    return values


off_correct = correctness_by_example(off)
on_correct = correctness_by_example(on)
common = sorted(set(off_correct) & set(on_correct))
changed = [key for key in common if off_correct[key] != on_correct[key]]
off_score = off.get('humaneval_pass@1')
on_score = on.get('humaneval_pass@1')

print(f'Hook-Off ACC       : {off_score:.8f}')
print(f'Hook-On ACC        : {on_score:.8f}')
print(f'ACC difference     : {on_score - off_score:+.8f}')
print(f'Compared examples  : {len(common)}')
print(f'Correctness changed: {len(changed)}')
print(f'Hook-Off result    : {off_path}')
print(f'Hook-On result     : {on_path}')

if off_score == on_score and not changed:
    print('Conclusion          : PASS - ACC and per-example correctness match.')
else:
    print('Conclusion          : MISMATCH - inspect the paired predictions.')
PY

echo
echo "Done. These --debug runs are for determinism/ACC validation, not latency or speedup measurement."
