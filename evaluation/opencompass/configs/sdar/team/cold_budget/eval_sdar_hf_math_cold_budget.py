"""Evaluate TEAM with a fixed shared cold-token expert budget."""

from mmengine.config import read_base

with read_base():
    from ..eval_sdar_hf_math import *  # noqa: F401, F403

COLD_EXPERT_BUDGET = int(
    __import__('os').environ.get('COLD_EXPERT_BUDGET', '8'))
COLD_EXPERT_SELECTION = __import__('os').environ.get(
    'COLD_EXPERT_SELECTION', 'probability_sum')
for model in models:
    model['abbr'] = (
        f"{model['abbr']}-cold-{COLD_EXPERT_SELECTION}"
        f"-budget-{COLD_EXPERT_BUDGET}")
    model['generation_kwargs']['cold_expert_budget'] = COLD_EXPERT_BUDGET
    model['generation_kwargs'][
        'cold_expert_selection'] = COLD_EXPERT_SELECTION

work_dir = (
    f'./outputs/team_cold_budget/math_{COLD_EXPERT_SELECTION}'
    f'_b{COLD_EXPERT_BUDGET}')
