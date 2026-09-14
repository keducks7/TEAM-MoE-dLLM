"""ACC-only TEAM interventions; defaults reproduce the B0 control."""
from mmengine.config import read_base

with read_base():
    from ..eval_sdar_hf_mbpp import *

COLD_REUSE_INTERVAL = int(__import__('os').environ.get('COLD_REUSE_INTERVAL', '0'))
HOT_PRUNE_COUNT = int(__import__('os').environ.get('HOT_PRUNE_COUNT', '0'))
HOT_PROTECT_TOP1 = __import__('os').environ.get('HOT_PROTECT_TOP1', '1') == '1'
RUN_TAG = f'coldR{COLD_REUSE_INTERVAL}_hotP{HOT_PRUNE_COUNT}_protect{int(HOT_PROTECT_TOP1)}'
for model in models:
    model['collect_metrics'] = False
    model['abbr'] += '-' + RUN_TAG
    model['generation_kwargs'].update(
        cold_expert_budget=0,
        cold_reuse_interval=COLD_REUSE_INTERVAL,
        hot_prune_count=HOT_PRUNE_COUNT,
        hot_protect_top1=HOT_PROTECT_TOP1)
work_dir = './outputs/team_interventions/mbpp_' + RUN_TAG
