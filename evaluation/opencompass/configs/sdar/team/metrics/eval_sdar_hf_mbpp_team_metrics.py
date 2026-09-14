"""Evaluate TEAM on MBPP while collecting unified APF/TPF/APT metrics."""

from mmengine.config import read_base

with read_base():
    from ..eval_sdar_hf_mbpp import *  # noqa: F401, F403

RUN_TAG = 'mbpp_team_b32_thr0_95_metrics_v1'
for model in models:
    model['abbr'] = f"{model['abbr']}-team-metrics"
    model['collect_metrics'] = True
    model['metrics_output_dir'] = f'./analysis/records/metrics/{RUN_TAG}'

work_dir = f'./analysis/records/opencompass_runs/{RUN_TAG}'
