# 解码前沿专家需求保护：离线四版本

预算 B 是当前层/当前 forward 的共享候选集合容量，即唯一专家数。
每 token 仍在候选集合中选 Top-8；B=16 不表示每 token 使用 16 个专家。
该原型只分析已有 Vanilla traces，不修改 modeling_sdar_moe.py，不执行干预 forward，
不产生新的 ACC 或速度结果。需要 Python + NumPy，无需 GPU。

## 四个版本

1. position_confidence_reference：按参考 Hot 的归一化原始 Top-8 权重聚合选 B 个专家。
2. protected_shared：保留参考 Hot 的 Top-1，逐次照顾覆盖最低的 Hot，达到 90% 目标或预算耗尽；剩余空间按 Cold 原始 Top-8 权重和分配。
3. protected_dependency：完全相同的保护阶段；剩余空间用历史依赖加权 Cold 投票。
4. protected_shuffled_dependency：在当前 Cold 集合内打乱历史依赖，保持权重分布，检验结构是否有增益。

Hot 使用前一迭代置信度、当前已填位置和历史置信度前 2 个位置的距离规则，
与前一轮 position_confidence_reference 相同。它是 Vanilla 单轨迹参考，不能称为完整 TEAM。
历史依赖来自前一迭代参考 Hot 接收位置到当前 Cold 源位置的 attention；去除自连接。
这与前一轮“历史高置信前 25% 接收集合”的定义有区别，本轮明确使用参考 Hot。

剩余空间中的依赖权重为 `1 + d/max(d)`，范围 [1,2]；全零则为 1。
Cold 为空时共享投票退回全体 mask；此状态不算依赖生效机会。
若保护阶段占满预算，三个 protected 版本相同；必须报告这一比例。

覆盖定义为原始归一化 Top-8 权重被集合保留的比例，不能等同于专家输出贡献或正确率。
逐次保护时挑覆盖最小的 Hot，补入它权重最高的未保留原始专家。
覆盖相同按物理位置顺序，专家分数相同按频次/编号稳定排序。
90% 是预设路由覆盖目标，不是质量保证；各层各任务统一，未按 ACC 搜索。

## 预算不可行

若必保 Top-1 数超过请求预算，记录 budget_relaxed=1。所有四个版本在该状态统一使用
`max(min(B, original_union), protected_top1_count)` 的实际容量；没有哪一侧额外占预算。

- strict 主表只包含原请求预算能容纳必保 Top-1 且实际压缩并集的状态。
- aligned 表额外包含提高预算的状态，逐状态容量仍完全相同；不可称为严格固定请求预算实验。
- 原并集小于预算时所有规则容量截断到并集大小，记录但不纳入压缩主表。
- state_forward.csv 保存不可行标志、实际容量、保护阶段开销、剩余空间和依赖生效机会。

## 指标

保留前一轮的本步/下一步接受覆盖、全部/剩余覆盖、Top-1 丢失、覆盖第 10 百分位、
实际激活专家并集；新增参考 Hot/Cold 覆盖、最差 Hot/最差接受位置覆盖、
Hot Top-1 丢失、达到 90% 的 Hot 比例、低频专家路由权重保留比例。
policy_forward.csv 同时保存集合专家 ID 与相对 protected_shared 是否改变集合。

观察单位先 forward、再题目；题目等权 bootstrap。下一步标签仅评估使用，缺失时删失。
主表覆盖全部可行压缩状态，opportunity_strict 仅用于解释依赖有机会生效的子集。
不要用后者代替主表或仅凭最差 Hot 改善判断算法有效，必须同时看接受位置及 Cold 损失。

## 运行

从 evaluation/opencompass 执行：

```bash
export PYTHON_BIN=/root/miniconda3/envs/team/bin/python
DATASET=humaneval TRACE_DIR=analysis/<已有 HumanEval 目录>/traces bash scripts/frontier_expert_pools.sh
DATASET=mbpp TRACE_DIR=analysis/<已有 MBPP 目录>/traces bash scripts/frontier_expert_pools.sh
DATASET=gsm8k TRACE_DIR=analysis/<已有 GSM8K 目录>/traces bash scripts/frontier_expert_pools.sh
```

可用 POOL_BUDGETS=8,16,24,32、HOT_COVERAGE_TARGET=0.9、OUTPUT_DIR 显式配置。
默认输出 analysis/reports/frontier_expert_pools/<任务>_<时间>_<pid>/。
核心结果为 report.md、summary.json、strict_summary.csv、strict_delta_vs_position_confidence_reference.csv、
strict_delta_vs_protected_shared.csv、feasibility_summary.csv；aligned 和 opportunity_strict 另存。

## 创新性定位

“缩小共享专家集合”已有 DES；“按 token 的解码状态分配计算”已有 TEAM、REFLEX。
Top-1 保护、覆盖目标和贪心补足本身应作为实验基线，不宜单独主张强创新。
潜在贡献是验证公共/私有专家需求如何随解码变化，以及历史 token 关联是否在保护前沿之后
仍能改善共享选择。如果依赖没有额外收益，应报告原型失败或贡献收缩，不能换个命名宣称创新。
最终还需独立样本上的在线 ACC、端到端延迟及保护/依赖消融，才有论文证据。

原始论文：

- DES: https://arxiv.org/abs/2602.00879
- TEAM: https://arxiv.org/abs/2602.08404
- REFLEX: https://arxiv.org/abs/2608.01784
