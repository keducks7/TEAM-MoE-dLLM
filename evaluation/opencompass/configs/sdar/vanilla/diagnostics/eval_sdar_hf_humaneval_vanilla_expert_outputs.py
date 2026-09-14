import torch
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import BD3withChatTemplate_original


with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets


summarizer = dict(
    dataset_abbrs=[
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
    ],
)


# First run: 1 sample with the hook off. Second run: keep 1 and turn it on.
# After the two outputs match, set PILOT_SAMPLES=10 and keep the hook on.
PILOT_SAMPLES = 10
COLLECT_EXPERT_OUTPUT_STATS = True
RUN_TAG = (
    f'humaneval_vanilla_expert_outputs_{PILOT_SAMPLES}samples_'
    f'hook_{"on" if COLLECT_EXPERT_OUTPUT_STATS else "off"}_v1'
)
datasets = [*humaneval_datasets]
for dataset in datasets:
    dataset['reader_cfg']['test_range'] = f'[0:{PILOT_SAMPLES}]'
    dataset['infer_cfg']['inferencer']['batch_size'] = 1


# This is one model process sharded over two A800 40GB GPUs, not two workers.
MODEL_GPUS = 2
NUM_WORKERS = 1
model_configs = [
    (
        f'SDAR-30B-A3B-b32-thr0_95-{RUN_TAG}',
        '/root/lkd/Models/SDAR-30B-A3B-Chat-b32',
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

            collect_metrics=True,
            metrics_output_dir=f'./analysis/records/metrics/{RUN_TAG}',
            collect_expert_output_stats=COLLECT_EXPERT_OUTPUT_STATS,
            expert_output_stats_dir=(
                f'./analysis/records/expert_output_stats/{RUN_TAG}'),

            # Keep route tracing off for this run: the expert-output file
            # already contains counts by step/layer/state/expert.
            collect_routing_trace=False,

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
                device_map='balanced',
                max_memory={0: '36GiB', 1: '36GiB'},
            ),
        ),
    )


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
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)


work_dir = f'./analysis/records/opencompass_runs/{RUN_TAG}'
