"""TEAM + shared pool; native metrics and generation timing, no hooks."""
from mmengine.config import read_base

with read_base():
    from ..eval_sdar_hf_humaneval import *

POOL_STRATEGY = __import__('os').environ.get('TEAM_POOL_STRATEGY', 'protected')
POOL_BUDGET = int(__import__('os').environ.get('TEAM_POOL_BUDGET', '36'))
COVERAGE_TARGET = float(__import__('os').environ.get('TEAM_POOL_COVERAGE_TARGET', '0.9'))
MODE = __import__('os').environ.get('MODE', 'acc')
SEED = int(__import__('os').environ.get('SEED', '20261014'))
MODEL_GPUS = int(__import__('os').environ.get('MODEL_GPUS', '2'))
GPU_MEMORY = __import__('os').environ.get('GPU_MEMORY', '37GiB' if MODEL_GPUS == 2 else '74GiB')
SAMPLES = __import__('os').environ.get('SAMPLES', '64')
SAMPLE_START = int(__import__('os').environ.get('SAMPLE_START', '32'))
COLLECT_LATENCY = __import__('os').environ.get('COLLECT_LATENCY', '1') == '1'
RUN_TAG = f'humaneval_{POOL_STRATEGY}_b{POOL_BUDGET}_seed{SEED}_{MODE}'
RUN_TAG += '_' + __import__('time').strftime('%Y%m%d_%H%M%S') + '_' + str(__import__('os').getpid())
assert MODE in ('acc', 'metrics')
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 1
    if SAMPLES == 'all':
        dataset['reader_cfg'].pop('test_range', None)
    else:
        dataset['reader_cfg']['test_range'] = f'[{SAMPLE_START}:{SAMPLE_START + int(SAMPLES)}]'
for model in models:
    model['abbr'] += '-' + POOL_STRATEGY
    model['path'] = __import__('os').environ.get('MODEL_PATH', model['path'])
    model['run_cfg']['num_gpus'] = MODEL_GPUS
    model['model_kwargs'].update(device_map='auto', max_memory={i: GPU_MEMORY for i in range(MODEL_GPUS)})
    model.update(collect_metrics=MODE == 'metrics', collect_latency=COLLECT_LATENCY,
                 metrics_backend='native', accepted_token_mode='mask_transitions', sample_seed=SEED,
                 metrics_output_dir=__import__('os').environ.get('TEAM_RECORD_DIR', './analysis/records/team_frontier_pool/' + RUN_TAG))
    model['generation_kwargs'].update(cold_expert_budget=0, cold_reuse_interval=0,
        cold_strategy='none', hot_prune_count=0, hot_prune_schedule=None,
        team_pool_strategy=POOL_STRATEGY, team_pool_budget=POOL_BUDGET,
        team_pool_coverage_target=COVERAGE_TARGET)
infer['runner']['retry'] = 0
eval['partitioner']['n'] = 1
work_dir = __import__('os').environ.get('TEAM_WORK_DIR', './outputs/team_frontier_pool/' + RUN_TAG)
