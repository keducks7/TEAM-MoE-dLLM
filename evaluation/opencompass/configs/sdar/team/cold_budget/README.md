# TEAM cold-token expert budget

These configs keep TEAM's hot/cold classification unchanged and give all cold
tokens in each MoE layer and speculative forward a shared budget of additional
unique experts. Each token still routes to the model's original top-8 experts;
the budget expands only the shared expert candidate set.

The configs read two optional environment variables:

- `COLD_EXPERT_BUDGET`: maximum number of extra shared experts. `0` reproduces
  original TEAM; useful pilot values are `2` and `4`.
- `COLD_EXPERT_SELECTION`: extra-expert selection strategy:
  - `probability_sum`: sum every cold token's full router probabilities
    (backward-compatible default).
  - `top1_frequency`: each cold token votes for its highest-probability expert
    outside the hot expert set.
  - `top8_frequency`: each cold token votes for the experts in its original
    unrestricted top-8; votes for experts already in the hot set are excluded.

The defaults remain `COLD_EXPERT_BUDGET=8` and
`COLD_EXPERT_SELECTION=probability_sum` so existing commands retain their old
behavior. Before running, copy the repository-root `modeling_sdar_moe.py` into
the Hugging Face model directory used by the config.

Run from `evaluation/opencompass`. For example, HumanEval accuracy with Top-1
Frequency and budget 2:

```bash
COLD_EXPERT_SELECTION=top1_frequency COLD_EXPERT_BUDGET=2 \
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  configs/sdar/team/cold_budget/eval_sdar_hf_humaneval_cold_budget.py
```

HumanEval accuracy with Top-8 Frequency and budget 2:

```bash
COLD_EXPERT_SELECTION=top8_frequency COLD_EXPERT_BUDGET=2 \
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  configs/sdar/team/cold_budget/eval_sdar_hf_humaneval_cold_budget.py
```

Add unified APF/TPF/APT collection by using the matching config under
`metrics/`, for example:

```bash
COLD_EXPERT_SELECTION=top1_frequency COLD_EXPERT_BUDGET=2 \
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  configs/sdar/team/cold_budget/metrics/eval_sdar_hf_humaneval_cold_budget_metrics.py \
  --debug
```

Replace `humaneval` with `mbpp`, `gsm8k`, or `math` for the other datasets.
Use non-metrics configs for latency and speedup measurements.
