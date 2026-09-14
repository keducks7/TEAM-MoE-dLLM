# SDAR evaluation configs

Run all commands from `evaluation/opencompass`.  Every Vanilla model path must
point to a model directory containing SDAR's official, unmodified
`modeling_sdar_moe.py`.

## Layout

- `team/`: clean TEAM accuracy/latency configs and legacy TEAM Mark configs.
- `team/metrics/`: TEAM APF/TPF/APT configs using the unified lightweight
  metric hooks. These use TEAM's ordinary `modeling_sdar_moe.py`, not the
  legacy `modeling_sdar_moe_mark.py`.
- `team/cold_budget/`: TEAM accuracy/latency and metric configs with a shared
  cold-token expert budget. Extra experts can be selected by probability sum,
  Top-1 Frequency, or Top-8 Frequency.
- `vanilla/accuracy/`: clean Vanilla accuracy/latency configs with metric hooks
  disabled.
- `vanilla/metrics/`: Vanilla APF/TPF/APT collection configs.  These runs are
  not valid latency or speedup measurements.
- `vanilla/diagnostics/`: routing and expert-output pilot configs.

## Vanilla accuracy, without metric hooks

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/accuracy/eval_sdar_hf_gsm8k_vanilla.py
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/accuracy/eval_sdar_hf_math_vanilla.py
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/accuracy/eval_sdar_hf_humaneval_vanilla.py
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/accuracy/eval_sdar_hf_mbpp_vanilla.py
```

## Vanilla accuracy plus APF/TPF/APT records

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/metrics/eval_sdar_hf_gsm8k_vanilla_metrics.py
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/metrics/eval_sdar_hf_math_vanilla_metrics.py
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/metrics/eval_sdar_hf_humaneval_vanilla_metrics.py
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/vanilla/metrics/eval_sdar_hf_mbpp_vanilla_metrics.py
```

## TEAM accuracy plus APF/TPF/APT records

Before running these configs, the model directory must contain the repository
root `modeling_sdar_moe.py`. Do not use `modeling_sdar_moe_mark.py` with the
unified hook configs.

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/team/metrics/eval_sdar_hf_gsm8k_team_metrics.py --debug
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/team/metrics/eval_sdar_hf_math_team_metrics.py --debug
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/team/metrics/eval_sdar_hf_humaneval_team_metrics.py --debug
CUDA_VISIBLE_DEVICES=0,1 python run.py configs/sdar/team/metrics/eval_sdar_hf_mbpp_team_metrics.py --debug
```

`--debug` keeps the inference task in the foreground so the final metric
summary is visible directly in the terminal. Without it, the same summary is
written to the inference `.out` log.

Aggregate a completed metric run with:

```bash
python analysis/tools/calculate_team_metrics.py \
  --metrics-dir analysis/records/metrics/<RUN_TAG>
```

Metric files are append-only.  Change `RUN_TAG` or select a new empty output
directory before repeating an experiment.

## TEAM fixed cold-token expert budget

Copy the repository-root `modeling_sdar_moe.py` into the TEAM model directory
before running. Set `COLD_EXPERT_BUDGET` and `COLD_EXPERT_SELECTION` as
environment variables; budget `0` is the original TEAM baseline. For example:

```bash
COLD_EXPERT_SELECTION=top1_frequency COLD_EXPERT_BUDGET=2 \
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  configs/sdar/team/cold_budget/eval_sdar_hf_humaneval_cold_budget.py
```

See `team/cold_budget/README.md` for both frequency strategies and metric
commands.
