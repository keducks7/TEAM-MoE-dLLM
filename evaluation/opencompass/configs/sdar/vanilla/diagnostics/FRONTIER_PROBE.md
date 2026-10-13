# 单位置专家替换：谁在支撑本轮解码

本实验检验：自身路由损失相近时，与当前接受位置关联更强的 masked token，其专家替换是否更容易改变当前解码。使用 Vanilla、bs=1、默认第 24 层，属于离线诊断，不是加速算法，也不输出“干预后 ACC”。

## 协议

1. 正常 forward，冻结原始接受集合 F。候选 source 是本轮不接受的 masked token。
2. source j 的关联分数是 mean attention[F, j]，保留 attention 的方向。它是本层关联代理，不是已证实的后续层因果影响。
3. 最高/最低 20% 分位为强/弱组。优先在同状态匹配最近 F 距离、置信度区间、置信度差 <=0.05、移除和替补专家的归一化原始 Top-8 权重差各 <=0.03。
4. 严格配对不可得时，不放松容差，改为各组按 prompt/block/step 的固定哈希随机抽一个 source，标为 `stratified_unmatched`。只用于描述及受控预测分析，绝不混入严格配对差值。
5. source 原始 Top-8 的第 8 名被排除，补入集合外最高 router 分数专家（无并列时即第 9 名）。仍保留 8 个专家、原始 Top-1 和前 7 名。边界同分记录 `boundary_tie`，沿实际 Top-8 ID 顺序处理。
6. 同一状态额外做 5 次 forward：一次 sham；强 source 原始 Top-8 重算、专家替换；弱 source 原始 Top-8 重算、专家替换。每次恢复 RNG，禁止 store_kv，检查 gate 和 KV。
7. 正常生成只沿原始 forward 前进。Probe 不改变接受集合或下一步输入。

原始 Top-8 重算与替换都经同一个单位置重算函数，区分路由变化与数值路径变化。若原始 Top-8 重算改变任一 masked 位置预测或任一接受决策，该状态标为 `control_failed`，排除有效效果和预测分析。报告仍保留失败计数。未跨越决策边界的数值误差也单独报告，连续指标同时给出相对原始结果及相对重算对照两种口径。

冻结 F 后测量：预测内容翻转、接受资格丢失、原始预测 token 的 log probability 变化、原始预测相对最强竞争 token 的 logit margin 变化。原预测翻转时 margin 可以为负，不改成新 winner 的正 margin。保存逐接收位置变化，避免平均抵消。

## 历史代理与预测比较

同时保存：

- 当前 attention 对当前真实 F 的关联：离线参考，不是在线可用信号。
- 上一步 attention 对当前真实 F 的关联：只检验旧图，仍使用当前 oracle F。
- 上一步 attention 对历史预测 F 的关联：在上轮仍未接受的位置中，按上轮置信度取最高四分之一，排除 source 自身。仅使用历史信息。这是固定启发式，不是经过验证的接受预测器。
- 同状态内打乱 source 分数得到的负对照：独立固定随机种子；不用于选择干预位置。

历史只取同块、相同位置、紧邻上一迭代。Probe stride=2 也仍使用上一迭代，不是上一次 probe。缺少历史记录为 null，不补零，不跨块延续。

主要连续分析目标预先固定为：F 的原始预测 logit margin 绝对变化，相对原始 Top-8 重算对照。

对比模型：router/置信度等控制量；增加位置距离；再分别增加当前依赖、打乱依赖、历史依赖。每次留出完整一道题，训练折内标准化，固定 ridge alpha=1，训练按 prompt 加权。数学拟合仅用于分析解释力，不更新 dLLM 权重或训练部署策略。历史比较用相同完整记录子集。少于 8 个有效 prompt 不拟合，该门槛不代表统计功效充分。

较小的留出 MSE 表示更好地预测局部变化，不代表更高 ACC。按 prompt 重采样的 CI 为探索性描述，没有修正交叉验证模型拟合的不确定性或多重比较。强弱分层采样会限制可外推人群。单次打乱是负对照，不是完整置换检验。

## 已有轨迹离线覆盖检查（2026-10-13）

以下仅复算 source 选择，不包含新干预效果，亦未验证服务器执行：

| 已有运行 | 尝试状态 | 严格配对 | 非配对分层抽样 | source 不足 | 有计划干预的题目 |
|---|---:|---:|---:|---:|---:|
| HumanEval 8 题 | 247 | 2 | 216 | 29 | 8/8 |
| MBPP 32 题 | 509 | 6 | 429 | 74 | 32/32 |
| GSM8K 32 题 | 1025 | 24 | 847 | 154 | 32/32 |

这也是保留非配对观测的原因：严格配对覆盖很低，不能靠放松容差把样本伪装成可比。最终有效数量还取决于原始 Top-8 重算对照是否通过。

## 前置条件

- 使用原生 Vanilla checkpoint 的 `modeling_sdar_moe.py`，不替换为 TEAM 或 Mark 模型。
- 更新本仓库普通 Vanilla wrapper、observer、两个 probe 模块、runner、脚本和分析工具。无需改 checkpoint 源码。
- 保持与之前 Vanilla dependency 实验相同的 Python 环境及模型。
- 默认双卡 `CUDA_VISIBLE_DEVICES=0,1`、`MODEL_GPUS=2`、每卡 `37GiB`。实际 GPU 编号可改。
- 输出目录必须是新的。探针额外运行最多 5 × PROBE_MAX_STEPS 次 forward/题，加上正常生成和观测成本，请先小规模检查。

```bash
cd /root/lkd/TEAM-MoE-dLLM/evaluation/opencompass
export PYTHON_BIN=/root/miniconda3/envs/team/bin/python
export MODEL_PATH=/root/lkd/Models/SDAR-30B-A3B-Chat-b32
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_GPUS=2
export GPU_MEMORY=37GiB
export SEED=20261012

"$PYTHON_BIN" scripts/test_frontier_probe.py
"$PYTHON_BIN" scripts/test_dependency_motivation.py
"$PYTHON_BIN" scripts/test_dependency_next_steps.py
```

先检查 1 道题。步数/块数限制可能没有有效干预，不能把 `no_match` 当作零影响。

```bash
DATASET=humaneval SAMPLES=1 GEN_LENGTH=256 MAX_BLOCKS=2 \
PROBE_MAX_STEPS=4 bash scripts/vanilla_frontier_probe.sh
```

然后 HumanEval 8 道覆盖检查：

```bash
DATASET=humaneval SAMPLES=8 MAX_BLOCKS=8 \
bash scripts/vanilla_frontier_probe.sh
```

检查输出 `probe_report/summary.json` 的 `observed_states`、`prompts_observed`、`matched_states` 和 `status_counts.control_failed`。不要将非配对数据的组间差值当作严格配对效应。如果原始 Top-8 对照改变了决策，应先排查数值重算路径。

完成小规模检查后，依次运行三个任务，每个 32 道：

```bash
for dataset in humaneval mbpp gsm8k; do
    DATASET="$dataset" SAMPLES=32 MAX_BLOCKS=8 \
    bash scripts/vanilla_frontier_probe.sh || break
done
```

该循环每个数据集独立生成一个新目录；HumanEval 32 道包含此前 8 道，不得将两次运行当成 40 个独立样本合并。若只补剩余 24 道，可设 `SAMPLE_START=8 SAMPLES=24`，合并时必须先检查 prompt 和状态不重复。

## 基线预测一致性

在全新目录运行同样参数，关闭 probe 但保留相同 observation 层；比较 `predictions` 中完整文本。两边模型、seed、样本起止、generation length、GPU 配置均要相同。

```bash
DATASET=humaneval SAMPLES=1 GEN_LENGTH=256 MAX_BLOCKS=2 \
DEPENDENCY_LAYERS=24 PROBE=0 \
RUN_DIR="$PWD/analysis/records/dependency/frontier_baseline_check" \
bash scripts/vanilla_dependency_motivation.sh

# 将 PROBE_RUN_DIR 改成同参数 probe 运行实际输出的新目录。
"$PYTHON_BIN" scripts/compare_dependency_predictions.py \
    --on "$PROBE_RUN_DIR" \
    --off "$PWD/analysis/records/dependency/frontier_baseline_check"
```

## 仅重做分析

```bash
"$PYTHON_BIN" analysis/tools/analyze_frontier_probes.py \
    --probe-dir /实际运行目录/probes \
    --output-dir /实际运行目录/frontier_report_recomputed
```

结果保存在 `probes/frontier_sequence*.jsonl` 和 `probe_report/{summary.json,report.md}`。原有双 token 共池干预仍可用 `PROBE=1 PROBE_KIND=pair` 运行；新入口默认 frontier，二者 schema 和分析分开。

后续是否实现全局共享池，取决于本实验是否支持依赖相对 router/距离的增量信息。没有在本次修改中加入部署裁剪策略，也没有据局部结果声称最终 ACC 或 speedup。
