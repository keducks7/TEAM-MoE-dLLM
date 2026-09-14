"""Collect Vanilla SDAR APF/TPF/APT records while evaluating GSM8K."""

import torch
from mmengine.config import read_base
from opencompass.models import BD3withChatTemplate_original
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask


with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets


summarizer = dict(dataset_abbrs=[['gsm8k', 'accuracy']])

datasets = [*gsm8k_datasets]
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 1


# Must contain SDAR's official, unmodified modeling_sdar_moe.py.
MODEL_PATH = '/root/lkd/Models/SDAR-30B-A3B-Chat-b32'
MODEL_GPUS = 2
NUM_WORKERS = 1
RUN_TAG = 'gsm8k_vanilla_b32_thr0_95_metrics_v1'

models = [
    dict(
        type=BD3withChatTemplate_original,
        abbr='SDAR-30B-A3B-b32-thr0_95-vanilla-gsm8k-metrics',
        path=MODEL_PATH,
        run_cfg=dict(num_gpus=MODEL_GPUS),
        collect_metrics=True,
        metrics_output_dir=f'./analysis/records/metrics/{RUN_TAG}',
        generation_kwargs=dict(
            mask_id=151669,
            gen_length=4096,
            block_length=32,
            denoising_steps=32,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            threshold=0.95,
        ),
        model_kwargs=dict(
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='balanced',
            max_memory={0: '37GiB', 1: '37GiB'},
            low_cpu_mem_usage=True,
        ),
    ),
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=NUM_WORKERS,
        num_split=1,
        force_rebuild=True,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=NUM_WORKERS,
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        retry=0,
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=16),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

work_dir = f'./analysis/records/opencompass_runs/{RUN_TAG}'
