# Vanilla dependency–expert motivation experiment

Read-only observation on **BD3withChatTemplate_original**, not TEAM or Mark.
No LAC/DCD/SEH, no expert pruning or changes to Top-8/acceptance. Native block
attention and prefix KV cache remain unchanged. Existing configs and metric
hooks are not enabled by this pilot. New observer flags default to off.

## Server runs (A800 80GB, one visible GPU)

Use the existing environment containing PyTorch/Transformers/OpenCompass and
HumanEval dependencies. Use the ordinary model checkpoint, **not Mark**. No
replacement of checkpoint `modeling_sdar_moe.py` is needed for this observer.

```bash
cd /root/lkd/TEAM-MoE-dLLM/evaluation/opencompass
export PYTHON_BIN=/root/miniconda3/envs/team/bin/python
export MODEL_PATH=/root/lkd/Models/SDAR-30B-A3B-Chat-b32
export CUDA_VISIBLE_DEVICES=0

# NumPy tests + CPU hook/output/RNG/cache parity test; run before GPU experiment.
"$PYTHON_BIN" scripts/test_dependency_motivation.py

# Smoke/config check: no model loading. RUN_DIR must NOT already exist.
DATASET=humaneval DRY_RUN=1 RUN_DIR="$PWD/analysis/records/dependency/config_check" \
    bash scripts/vanilla_dependency_motivation.sh

# First test one sample, two observed blocks. Generation still follows baseline.
DATASET=humaneval SAMPLES=1 MAX_BLOCKS=2 \
    bash scripts/vanilla_dependency_motivation.sh

# Main pilots: use separate directories for each dataset/run.
DATASET=humaneval SAMPLES=32 MAX_BLOCKS=8 \
    bash scripts/vanilla_dependency_motivation.sh
DATASET=mbpp SAMPLES=32 MAX_BLOCKS=8 \
    bash scripts/vanilla_dependency_motivation.sh
```

Default generation length/threshold are inherited from Vanilla accuracy configs
(4096 / .95), and dataset imports are identical: HumanEval and sanitized MBPP.
Samples are first 32 by default (`SAMPLE_START` changes the slice). Do not tune
on these samples and then report them as untouched final evaluation.

Default layers are physical layer IDs `0, num_layers//2, num_layers-1`.
Use `DEPENDENCY_LAYERS=all` or `DEPENDENCY_LAYERS=0,10,20,30` to override.
Record more layers only after smoke testing memory/storage. `MAX_BLOCKS` limits
observations, **not decoding**. No query renormalization over just masked keys.

For two GPUs use `CUDA_VISIBLE_DEVICES=0,1 MODEL_GPUS=2 GPU_MEMORY=37GiB`.
`GEN_LENGTH`, `SEED` and `GPU_MEMORY` can also be overridden; changing length
changes the experiment and must not be mixed with another setting.

## On/Off prediction check

Run these one-sample jobs separately (not concurrently on the same GPU), with
the same environment, checkpoint and seed. Compare generated prediction strings
under `on/opencompass/.../predictions` and `off/opencompass/.../predictions`, not
just ACC. Hook-Off still uses the same seed/config/dataset slice.

```bash
DATASET=humaneval SAMPLES=1 MAX_BLOCKS=2 SEED=20261012 \
RUN_DIR="$PWD/analysis/records/dependency/on" \
    bash scripts/vanilla_dependency_motivation.sh
DATASET=humaneval SAMPLES=1 MAX_BLOCKS=2 SEED=20261012 HOOK_OFF=1 \
RUN_DIR="$PWD/analysis/records/dependency/off" \
    bash scripts/vanilla_dependency_motivation.sh
"$PYTHON_BIN" scripts/compare_dependency_predictions.py \
    --on "$PWD/analysis/records/dependency/on" \
    --off "$PWD/analysis/records/dependency/off"
```

The observer never replaces outputs, changes RNG, updates KV or requests eager
attention. It hooks normalized Q/K and gate output, reproduces RoPE and prefix
keys, and reconstructs probabilities in FP32 **beside** the original SDPA.
This is not bit-identical to fused attention probabilities. It only supports
the inspected SDAR SDPA layout; unsupported remote model code fails explicitly.
Attention dropout must be zero and model must be in eval mode. Attention/kernel
reconstruction and CPU copies add overhead: **not a speed benchmark**.

## Outputs and interpretation

- `manifest.json`, `resolved_config.py`: seed, model path, settings, dataset slice.
- `traces/dependency_*.npz`: one prompt per file, physical layer IDs, block/step,
  global positions, token IDs, mask, predictions, confidence, accepted flags,
  head-mean current-block attention with full-prefix denominator, full router
  probabilities, raw Top-8 probabilities/IDs, same-layer hidden cosine matrix.
  Prompt hash and actual remote model source SHA256 are included.
- `report/history_summary.csv`: Q1 correlation, strong-edge recall **and precision**
  for previous same-layer attention, router cosine, expert overlap, hidden cosine,
  confidence similarity/change similarity, and position proximity. Large tie
  sets can exceed the chosen edge fraction; constants are undefined, not zero.
- `report/expert_relation_summary.csv`: Q2 Top-8 overlap/Jaccard, Top-1 equality,
  full router cosine, shared mass, private Top-8 mass, union size and masked-token
  expert use frequency, stratified by layer/mask ratio/distance/confidence/role.
- `report/matched_contrast_summary.csv`: strong-minus-weak contrasts within the
  same forward/layer/distance/confidence/accepted-role stratum. Unmatched sparse
  strata are excluded rather than extrapolated. This is not causal adjustment.
- `report/report.md`, `summary.json`, forward-level CSVs: overview/provenance.

Vanilla has no original TEAM Hot/Cold labels. `one_accepted` etc. are retrospective
roles for diagnostics, not labels available before forward. Cross-step analysis
uses only positions masked in both steps, same block and consecutive steps;
confidence changes require three consecutive states. No block-boundary history.
Bootstrap clusters by prompt; sample count does not become millions of independent
observations just because there are many pairs. Interpretation remains exploratory.

Reanalyze without rerunning inference:

```bash
"$PYTHON_BIN" analysis/tools/analyze_dependency_motivation.py \
    --trace-dir /absolute/run/traces --output-dir /absolute/run/report_reanalysis
```

Files to upload: `opencompass/models/huggingface_bd3_original.py`,
`opencompass/models/sdar_dependency_observer.py`,
`analysis/tools/analyze_dependency_motivation.py`,
`scripts/run_dependency_motivation.py`, `scripts/vanilla_dependency_motivation.sh`,
`scripts/test_dependency_motivation.py`, `scripts/compare_dependency_predictions.py`,
and this documentation. The analysis
**tool** is required; old analysis results are not required.
