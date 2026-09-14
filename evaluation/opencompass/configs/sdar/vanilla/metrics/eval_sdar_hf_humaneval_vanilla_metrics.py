import torch
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import BD3withChatTemplate_original


with read_base():
    # Use the ordinary HumanEval inferencer.  The *_2 dataset configuration is
    # reserved for the repository's TEAM mark pipeline.
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets

    # summarizer
    from opencompass.configs.summarizers.internlm2_keyset import summarizer
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import (
        mathbench_2024_summary_groups,
    )
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups


summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], [])

summary_groups.append(
    {
        'name': 'Mathbench',
        'subsets': ['mathbench-a (average)', 'mathbench-t (average)'],
    },
)

summarizer = dict(
    dataset_abbrs=[
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
    ],
    summary_groups=summary_groups,
)


datasets = [*humaneval_datasets]
for dataset in datasets:
    # SDAR's current block-diffusion wrapper only supports batch size 1.
    dataset['infer_cfg']['inferencer']['batch_size'] = 1


# CHANGE ME: this directory must contain the official, unmodified SDAR
# modeling_sdar_moe.py.  Do not copy TEAM's repository-root model file here.
MODEL_GPUS = 2
NUM_WORKERS = 1
model_configs = [
    (
        'SDAR-30B-A3B-Chat-b32-thr0_95-vanilla-metrics',
        "/root/lkd/Models/SDAR-30B-A3B-Chat-b32",
        32,
        0.95,
        MODEL_GPUS,
    ),
]

models = []
for abbr, path, block_length, threshold, num_gpus in model_configs:
    models.append(
        dict(
            type=BD3withChatTemplate_original,
            abbr=abbr,
            path=path,
            run_cfg=dict(num_gpus=num_gpus),

            # Lightweight Vanilla APF/TPF/APT collection.  Use a fresh output
            # directory for every run so stale JSONL records are not pooled.
            collect_metrics=True,
            metrics_output_dir=(
                './analysis/records/metrics/'
                'humaneval_vanilla_b32_thr0_95_metrics_v2'),

            generation_kwargs=dict(
                mask_id=151669,
                gen_length=4096,
                block_length=block_length,
                denoising_steps=block_length,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                cfg_scale=0.0,
                remasking='low_confidence',
                threshold=threshold,
            ),
            model_kwargs=dict(
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # Load one SDAR-30B model across both visible A800 GPUs.
                device_map='balanced',
                # Reserve memory on each card for activations and KV cache.
                max_memory={0: '37GiB', 1: '37GiB'},
                low_cpu_mem_usage=True,
            ),
        ),
    )


infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        # One inference process owns both GPUs.  Setting this to 2 would split
        # HumanEval into two tasks and reload the 30B model for each task.
        num_worker=NUM_WORKERS,
        num_split=1,
        # Avoid reusing predictions from a TEAM or previous Vanilla run.
        force_rebuild=True,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=NUM_WORKERS,
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        # Retrying a partially completed task can append duplicate metric
        # records.  Rerun with a fresh metrics_output_dir after a failure.
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

work_dir = (
    './analysis/records/opencompass_runs/'
    'humaneval_vanilla_b32_thr0_95_metrics_v2')
