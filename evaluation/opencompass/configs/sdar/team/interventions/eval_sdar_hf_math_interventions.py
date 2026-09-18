"""ACC-only TEAM interventions; defaults reproduce the B0 control."""
from mmengine.config import read_base

with read_base():
    from ..eval_sdar_hf_math import *

COLD_REUSE_INTERVAL = int(__import__('os').environ.get('COLD_REUSE_INTERVAL', '0'))
HOT_PRUNE_COUNT = int(__import__('os').environ.get('HOT_PRUNE_COUNT', '0'))
HOT_PROTECT_TOP1 = __import__('os').environ.get('HOT_PROTECT_TOP1', '1') == '1'
RUN_TAG = f'coldR{COLD_REUSE_INTERVAL}_hotP{HOT_PRUNE_COUNT}_protect{int(HOT_PROTECT_TOP1)}'
COLD_STRATEGY = __import__('os').environ.get('COLD_STRATEGY', 'none')
HOT_PRUNE_SCHEDULE_TEXT = __import__('os').environ.get('HOT_PRUNE_SCHEDULE', '')
HOT_PRUNE_SCHEDULE = [int(v) for v in HOT_PRUNE_SCHEDULE_TEXT.split(',')] if HOT_PRUNE_SCHEDULE_TEXT else None
RUN_TAG += f'_{COLD_STRATEGY}_schedule' + (HOT_PRUNE_SCHEDULE_TEXT.replace(',', '-') or 'uniform')
for model in models:
    model['collect_metrics'] = False
    model['abbr'] += '-' + RUN_TAG
    model['generation_kwargs'].update(
        cold_expert_budget=0,
        cold_strategy=COLD_STRATEGY,
        hot_prune_schedule=HOT_PRUNE_SCHEDULE,
        cold_reuse_interval=COLD_REUSE_INTERVAL,
        hot_prune_count=HOT_PRUNE_COUNT,
        hot_protect_top1=HOT_PROTECT_TOP1)
work_dir = './outputs/team_interventions/math_' + RUN_TAG
