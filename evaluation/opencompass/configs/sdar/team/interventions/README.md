# TEAM ACC interventions

## Cold strategies and depth-dependent candidate pruning

`COLD_STRATEGY` is one of `none`, `reuse`, `moe_zero`, `full_skip`.
For compatibility, `none` with a nonzero `COLD_REUSE_INTERVAL` still enables
the earlier reuse experiment. Explicit `reuse` requires interval >=2.
`moe_zero` and `full_skip` require interval 0 and cannot combine with reuse.

- `moe_zero`: Cold positions retain attention/residual, but do not enter the
  router or experts. Their aggregate MoE output is zero, not a stale cache.
- `full_skip`: In speculative forwards, remove Cold positions before model
  embedding/attention/MoE/final projection. They do not supply current K/V.
  Keep original RoPE position IDs, cached-prefix keys and branch boundaries.
  Scatter per-layer MLP caches back to full coordinates (skipped entries zero).
  Do not accept skipped positions: retain mask and previous confidence, then
  reclassify on the next iteration. A newly hot/filled position computes again.
  There is no periodic forced refresh for this mode. Initial block forwards
  and completed-block KV prefill remain unchanged. Supports batch=1, existing
  four-branch, 32-token TEAM path and threshold<1 only. Confidence can be stale
  while a position stays cold; test termination/quality on the server.

`HOT_PRUNE_SCHEDULE=8,4,0` overrides `HOT_PRUNE_COUNT`: at zero-based layer i,
use segment `min(2, i*3//num_layers)`. It deletes up to 8/4/0 candidates in
early/middle/late layers; actual set sizes are not fixed. Top-1 protection,
at least 8 surviving experts, and the original Cold pool remain unchanged.
This is candidate-pool pruning, not per-token top-k reduction or Cold Budget.

```bash
# Set DATASET=mbpp/gsm8k/math as needed; defaults to HumanEval.
COLD_STRATEGY=none COLD_REUSE_INTERVAL=0 HOT_PRUNE_COUNT=0 HOT_PRUNE_SCHEDULE='' bash scripts/team_interventions_acc.sh
COLD_STRATEGY=reuse COLD_REUSE_INTERVAL=2 HOT_PRUNE_COUNT=0 HOT_PRUNE_SCHEDULE='' bash scripts/team_interventions_acc.sh
COLD_STRATEGY=moe_zero COLD_REUSE_INTERVAL=0 HOT_PRUNE_COUNT=0 HOT_PRUNE_SCHEDULE='' bash scripts/team_interventions_acc.sh
COLD_STRATEGY=full_skip COLD_REUSE_INTERVAL=0 HOT_PRUNE_COUNT=0 HOT_PRUNE_SCHEDULE='' bash scripts/team_interventions_acc.sh
# Layer-dependent Hot-only pruning; Cold computes normally.
COLD_STRATEGY=none COLD_REUSE_INTERVAL=0 HOT_PRUNE_SCHEDULE=8,4,0 HOT_PROTECT_TOP1=1 bash scripts/team_interventions_acc.sh
# Uniform comparator
COLD_STRATEGY=none COLD_REUSE_INTERVAL=0 HOT_PRUNE_SCHEDULE=4,4,4 HOT_PROTECT_TOP1=1 bash scripts/team_interventions_acc.sh
```

Tags include Cold strategy and layer schedule to avoid mixing output directories.
Sync the wrapper, root model file, four intervention configs and test script.
Also replace the model file in the checkpoint directory. Mark is unchanged.

Use the repository's ordinary `modeling_sdar_moe.py` in your TEAM checkpoint
directory. Do not use Mark. These configs disable metric hooks and set Cold
Budget to zero. Changes to the model wrapper must also be synced to the server.

- `COLD_REUSE_INTERVAL=0`: off. Values >=2 reuse cached aggregate MoE outputs
  only for positions cold in both the current branch and the previous selected
  branch. First forwards compute fully; every R-th denoising step refreshes.
  Newly filled positions and hot positions recompute. Attention and residuals
  remain current. The existing DCD cache carries the outputs; no KV cache change.
- `HOT_PRUNE_COUNT=0`: off. Positive values remove at most that many low-frequency
  experts from the hot-mask-token top-8 union per layer/forward, aggregating the
  four speculative branches. Ties use expert ID order. At least eight experts
  remain. Hot tokens reroute to top-8 within this reduced pool using current
  router probabilities and the existing normalization rule.
- `HOT_PROTECT_TOP1=1`: protect every hot token's original top-1 expert. `0`
  disables protection. Protection may mean fewer than P experts can be removed.

Cold's pool is constructed before hot pruning. Decoded DCD handling is unchanged.
Consequently an expert removed from hot can still be used by cold: this experiment
isolates hot routing, and does not guarantee fewer globally activated experts.
The first full forward in each block remains unchanged for both interventions.

From `evaluation/opencompass`:

```bash
# TEAM B0 control
COLD_REUSE_INTERVAL=0 HOT_PRUNE_COUNT=0 bash scripts/team_interventions_acc.sh
# Cold reuse only
COLD_REUSE_INTERVAL=2 HOT_PRUNE_COUNT=0 bash scripts/team_interventions_acc.sh
# Hot pruning only, top-1 protected
COLD_REUSE_INTERVAL=0 HOT_PRUNE_COUNT=2 HOT_PROTECT_TOP1=1 bash scripts/team_interventions_acc.sh
# CPU smoke tests (requires torch)
python scripts/test_team_interventions.py
```

Set `DATASET=mbpp`, `gsm8k`, or `math` for other datasets (default: humaneval).
Set `CUDA_VISIBLE_DEVICES` and `PYTHON_BIN` for your server. Outputs are separated
by parameter tag. These commands do not guarantee per-sample deterministic seeds;
repeat paired comparisons before attributing small ACC differences to a method.
Tests cover local MoE behavior, not multi-GPU branch/cache integration. Run a
small server smoke test before full evaluation. New experiments are not proven
to accelerate generation or preserve accuracy.
