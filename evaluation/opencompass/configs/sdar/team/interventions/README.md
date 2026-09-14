# TEAM ACC interventions

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
