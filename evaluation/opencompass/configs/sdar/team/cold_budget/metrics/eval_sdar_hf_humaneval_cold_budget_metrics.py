"""Collect TEAM metrics with the fixed HumanEval cold-token budget."""

from mmengine.config import read_base

with read_base():
    from ..eval_sdar_hf_humaneval_cold_budget import *  # noqa: F401, F403

RUN_TAG = (
    f'humaneval_team_cold_{COLD_EXPERT_SELECTION}'
    f'_b{COLD_EXPERT_BUDGET}_metrics_v1')
for model in models:
    model['abbr'] = f"{model['abbr']}-metrics"
    model['collect_metrics'] = True
    model['metrics_output_dir'] = f'./analysis/records/metrics/{RUN_TAG}'

work_dir = f'./analysis/records/opencompass_runs/{RUN_TAG}'
