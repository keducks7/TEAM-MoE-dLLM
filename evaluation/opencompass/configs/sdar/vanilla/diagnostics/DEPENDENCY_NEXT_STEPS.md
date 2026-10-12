# GSM8K、历史失效与第 24 层配对干预

三项实验独立。GSM8K 为观测；历史失效使用现有 NPZ，不需 GPU；配对干预包含额外 forward，仅做机制诊断。普通 Vanilla 与 TEAM 配置默认不启用干预。

## 环境与服务器更新

```bash
cd /root/lkd/TEAM-MoE-dLLM
git pull --ff-only https://github.com/keducks7/TEAM-MoE-dLLM.git main
cd evaluation/opencompass
export PYTHON_BIN=/root/miniconda3/envs/team/bin/python
export MODEL_PATH=/root/lkd/Models/SDAR-30B-A3B-Chat-b32
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_GPUS=2
export GPU_MEMORY=37GiB
export SEED=20261012

"$PYTHON_BIN" scripts/test_dependency_motivation.py
"$PYTHON_BIN" scripts/test_dependency_next_steps.py
```

使用可运行普通 Vanilla 的环境和 checkpoint，不用 Mark，不需要替换 checkpoint 的模型源码。单卡 A800 可留给其他 eval。两卡每张预算 37GiB，若卡容量更小需调整。GPU 逻辑编号由 CUDA_VISIBLE_DEVICES 映射。

本地验证包含 NumPy/语法/报告测试。PyTorch hook/replay 测试必须在有 PyTorch 的服务器完成，本地跳过不等于通过。两个脚本任何测试失败时，不进行正式干预。

## 1. GSM8K 观测

```bash
# 先一条检查；沿用原 generation length，不改变观测协议。
DATASET=gsm8k SAMPLES=1 MAX_BLOCKS=2 \
bash scripts/vanilla_dependency_motivation.sh

# 正式探索：与 HumanEval/MBPP 同样设置。
DATASET=gsm8k SAMPLES=32 MAX_BLOCKS=8 \
bash scripts/vanilla_dependency_motivation.sh
```

GSM8K 继承 `configs/sdar/vanilla/accuracy/eval_sdar_hf_gsm8k_vanilla.py` 的 0-shot 数据集和阈值 .95。观察层默认 0/24/47、block=32、gen_length=4096。MAX_BLOCKS 限制记录而非生成，避免生成截断混淆。

每次运行使用独立目录。自动生成 `report/`（原动机报告）与 `drift/`（新增历史失效报告）。若 evaluator 缺依赖，不要重复 GPU 推理；已有 traces 可独立分析。

## 2. 现有记录的历史失效分析（不用重新推理）

把 TRACE_DIR 换成服务器现有数据路径；REPORT_DIR 必须是新目录。

```bash
"$PYTHON_BIN" analysis/tools/analyze_dependency_drift.py \
  --trace-dir /absolute/humaneval_run/traces \
  --output-dir /absolute/humaneval_run/drift_v1

"$PYTHON_BIN" analysis/tools/analyze_dependency_drift.py \
  --trace-dir /absolute/mbpp_run/traces \
  --output-dir /absolute/mbpp_run/drift_v1
```

只跨同块相邻步骤、持续 masked 位置。使用上一轮接受数量、接受比例及历史关联质量预测 `1-Spearman(current, previous)`。控制 mask 比例、pair 数、接受比例和上一轮置信度的 partial rank 是描述性分析，不是训练预测器或因果调整，也不自动拟合刷新阈值。输出 `states.jsonl`、`summary.json` 和 `report.md`。

## 3. 干预一致性 smoke check

先单样本、短生成检查工程正确性。下列两次 gen_length 都为 256；它们不是与 4096 正式结果混合的实验。

```bash
DATASET=humaneval SAMPLES=1 GEN_LENGTH=256 MAX_BLOCKS=2 \
DEPENDENCY_LAYERS=24 PROBE=1 PROBE_MAX_STEPS=4 \
RUN_DIR="$PWD/analysis/records/dependency/probe_smoke_on" \
bash scripts/vanilla_dependency_motivation.sh

DATASET=humaneval SAMPLES=1 GEN_LENGTH=256 MAX_BLOCKS=2 \
DEPENDENCY_LAYERS=24 PROBE=0 HOOK_OFF=1 \
RUN_DIR="$PWD/analysis/records/dependency/probe_smoke_off" \
bash scripts/vanilla_dependency_motivation.sh

"$PYTHON_BIN" scripts/compare_dependency_predictions.py \
  --on "$PWD/analysis/records/dependency/probe_smoke_on" \
  --off "$PWD/analysis/records/dependency/probe_smoke_off"
```

目录必须不存在。一个样本可能没有合适配对：查 `probes/` 和 `probe_report/summary.json` 的匹配覆盖，不能把 no_match 当作零影响。没有匹配时 prediction parity 只能检查旁路/观测部分；CPU replay 测试及有真实匹配的 GPU smoke 才验证干预重放。

## 4. 小规模配对干预

```bash
# 先 8 样本检查覆盖及错误。
DATASET=humaneval SAMPLES=8 MAX_BLOCKS=8 \
DEPENDENCY_LAYERS=24 PROBE=1 \
bash scripts/vanilla_dependency_motivation.sh

DATASET=mbpp SAMPLES=8 MAX_BLOCKS=8 \
DEPENDENCY_LAYERS=24 PROBE=1 \
bash scripts/vanilla_dependency_motivation.sh

# 检查后扩到 32 样本；严格匹配覆盖可能仍然较低。
for dataset in humaneval mbpp; do
  DATASET="$dataset" SAMPLES=32 MAX_BLOCKS=8 \
  DEPENDENCY_LAYERS=24 PROBE=1 \
  bash scripts/vanilla_dependency_motivation.sh
done
```

参数含义：

- `PROBE_LAYER=24`：物理第 24 层，按 0 开始编号。
- `PROBE_POOL_SIZE=12`：两个目标 token 共用 12 个候选专家，每个仍使用 Top-8，不是每 token Top-12。
- `PROBE_STRIDE=2`：检查块内 step 0/2/4/...。
- `PROBE_MAX_STEPS=48`：每样本最多尝试 48 个观测状态；未匹配不执行额外 forward。
- `PROBE_MASS_TOLERANCE=0.03`：逐 token 被排除的原 Top-8 权重比例匹配误差最多 3 个百分点。
- `PROBE_CONFIDENCE_TOLERANCE=0.05`：逐 token 置信度匹配误差最多 .05。

配对同时控制精确距离、原始 Top-8 重叠数、置信度分组和接受角色；逐 token 的置信度/接受/损失共同对齐，不独立排序后混淆。强弱依赖为本步 masked pairs 的 80%/20% 分位，两对 token 不重合。各自从原 Top-8 并集构建 saliency pool 并保护两位置 Top-1。并集不大于预算时不计为有效干预。未匹配不自动放松。

同一状态依次运行 sham（无干预重放）、strong pair 限制、weak pair 限制。额外 forward 只修改目标层两个位置的 MoE 输出；其他位置本层输出不变，下游 attention 正常传播。它是在已计算原始专家输出后重新计算目标行的诊断，不是高效 dispatch 实现。

继续生成只使用原始 forward 的预测/接受，诊断结束恢复 RNG，KV 的身份/形状/版本检查必须不变。sham 若改变预测/接受则报错；目标 gate 与原始记录不一致也报错。带干预的 logits 和接受决策不写入真实生成序列。

重点看 `probe_report/report.md` 的其他位置预测翻转、原先接受位置的预测翻转、接受集合变化，以及 strong-minus-weak 配对差值。置信度变化不是正确性或质量损伤标签。没有完整生成干预，因此不能用最终 baseline ACC 评价干预质量；额外耗时也不能报告成 speedup。

新增代码：`opencompass/models/sdar_dependency_probe.py`、`analysis/tools/analyze_dependency_drift.py`、`analysis/tools/analyze_dependency_probes.py`、`scripts/test_dependency_next_steps.py`；同时更新 wrapper/observer/launcher/bash。同步整个提交即可，不需要上传历史分析结果。
