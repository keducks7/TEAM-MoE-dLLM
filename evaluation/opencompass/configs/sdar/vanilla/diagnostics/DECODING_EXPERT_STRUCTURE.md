# 解码进程与 MoE 专家共享：三个观察剖析

目标：先寻找跨任务规律，再判断是否值得设计历史依赖加权的共享专家集合。
模型路线为 Vanilla、bs=1，保持原 Top-8 与解码过程。记录用现有只读 observer，
分析只需要 NumPy，不需要 GPU。不启用干预 probe、Mark 或 APF/TPF metric hook。

## 数据与控制

复用已有 HumanEval、MBPP、GSM8K 各 32 题的原始 traces。每题前 8 个 block；
已有 auto 层为物理层 0、24、47。先看三层跨任务趋势，必要时再补全部层。
8 题是代码检查规模，32 题仍是探索规模；确定规则后应换独立样本验证。
正常预测不由分析修改。前 8 个 block 的结论不可推广成整条长序列规律。

实验单位依次为同状态 token 对或 token、forward、题目；最终题目等权。
区间按题目 bootstrap，不把大量 token 对当成独立样本。
剩余 mask 比例区间为 0–.25、.25–.5、.5–.75、.75–1；比例越大越接近块的早期。
每个 block 单独定义阶段，不能用整条回答中的步骤编号代替块内进展。

## 剖析 1：专家共享关系是否随解码阶段与接受角色变化

对当前仍为 mask 的 token 对，取对称 attention `(Aij+Aji)/2`；
同一状态内最高/最低 20% 分为强/弱关联，分位点重合时不构造比较。
测原始 Top-8 身份重叠，以及共享身份承载的归一化 Top-8 权重。

分前/中/后层、剩余 mask 比例、对中本步接受数 0/1/2。
另对本步都未接受的位置，按下一步实际接受数 0/1/2 分组。
下一步只有同 block、相邻 step、相同位置映射的记录才有效；末步与缺失步按删失处理。
接受角色是事后标签，不能进入在线候选集合规则。

强减弱的主比较在同 forward、同 layer、同距离分组、同置信度分组、同接受角色内完成。
距离与置信度只是分组控制，不能声称已排除全部混杂。缺少一侧的分组不外推。
先平均同状态控制分组，再平均题目内 forward，最后平均题目。

可支持的发现：关联与路由共享的联系在接近接受的位置或某些去噪阶段更突出。
若所有角色和阶段几乎相同，该联系可能更多是一般表征规律，需降低解码特异性主张。
稀疏分组不能只看点估计，要同时看题目覆盖与区间。

## 剖析 2：共享的是身份，还是主要路由权重

归一化原始 Top-8 权重 `w_i,e = p_i,e / sum_{e in S_i} p_i,e`。
对一对 token，分别计算其共享身份承载的权重，再平均两侧；余下为私有权重。
这区分“重叠很多但主要权重不共享”和“主要权重也共享”。它仍不等于专家输出贡献。

另外统计本状态仅 1–2 个 masked token 使用的低频专家：
它们承载多少 Top-8 权重、多少 token 的 Top-1，以及活跃专家中的占比。
按本步接受、下一步接受、剩余位置与阶段分组，并保存 mask_count。
低频是当前状态属性，不能当成专家永久不重要。
若低频专家对少量将接受的位置承载高权重，说明简单按频次裁剪可能误伤关键需求。

## 剖析 3：相同集合预算，依赖信息能否改善选择

每个有相邻历史的状态、每层，只在当前 masked token 原始 Top-8 并集中选共享集合。
请求预算固定为 8/16/24/32 个唯一专家；并集不足时实际预算取并集大小。
所有规则共享同一状态、同一候选专家、同一实际预算。
预算控制集合大小，每 token 仍在集合内选原始数量的 Top-k，SDAR 为 8。

评分规则：

| 规则 | 专家得分 | 用途 |
|---|---|---|
| frequency | 原始 Top-8 的票数 | 频次基线 |
| probability_sum | 全体 masked 的完整 router 概率和 | 既有聚合思路 |
| topk_mass | 归一化原始 Top-8 权重和 | 不加 token 权重的直接基线 |
| confidence_weighted | `(1+上一迭代置信度)` 加权 Top-8 权重 | 解码置信度基线 |
| position_confidence_reference | 位置/历史置信度参考 Hot 的 Top-8 权重和 | TEAM 分类思想参考 |
| historical_dependency | 历史关联强度加权 Top-8 权重 | 单独看关联信号 |
| confidence_plus_dependency | 历史置信度与历史关联权重相乘 | 检验置信度之外的增益 |
| confidence_plus_shuffled_dependency | 同状态打乱历史关联权重 | 关联结构负对照 |
| oracle_current_dependency | 当前实际接受位置的当前 attention 加权 | 事后诊断，不能在线使用 |

历史预测接收位置：当前仍为 mask 的位置中，按上一迭代置信度选前 25%。
历史关联强度：上一迭代这些接收位置到各源位置的 attention 均值，去除自连接。
权重固定为 `1 + score/max(score)`，范围 [1,2]；全零时全部为 1。
该强度和下一步是否实际接受没有信息泄漏；前者是历史预测，后者仅用于评估。
置信度、前 25% 与权重公式是预设的简单剖析规则，不按测试 ACC 搜参数。

位置/置信度参考用当前已填位置和历史置信度前 2 个位置作距离锚点；
距离 <=3 或历史置信度 >=.7 的位置为参考 Hot，前 2 个位置总被保护。
它在 Vanilla 单轨迹上实现，缺少 TEAM 四候选及待 DCD 更新行，**不是原 TEAM 的准确复现**。
参考规则以有限预算选择专家时，先按 Hot 得分，零票候选用全体频次补足。
并列按全体频次、专家编号稳定排序，所有规则一致。

主要指标：

- 原始 Top-8 权重覆盖（全部 mask、本步接受、下一步接受、其余位置）。
- 原始 Top-1 丢失率与 token 覆盖率的第 10 百分位，避免只改善平均覆盖。
- 对当前接受集合关联最高 20% 的源位置的覆盖；该分组是事后评估。
- 强制在集合内重新选 Top-8 后的实际激活专家并集及减少数。
- 保存所有未压缩状态，主比较只包含请求预算小于原始并集的状态。

最重要的比较是逐状态 `confidence_plus_dependency - confidence_weighted`，
同时对照 shuffled、概率和、频次及位置/置信度规则。相同预算下成本代理可能相同，
此时应看覆盖收益；实际专家数是成本代理，不能宣称实测速率。

同一正常路由得到的“全体覆盖”天然有利于 topk_mass，故应同时报告接受位置与尾部覆盖，
不能只挑有利的子群。允许 trade-off：关键位置覆盖增加，其他位置可能下降。
如果该规则只对关联源子群有利，却不改善本步/下一步接受覆盖，动机链还不完整。

## 决策与下一步

先对现有三组数据完成分析，无需新模型运行。
若阶段/接受角色的专家结构有稳定差异，且历史依赖在相同预算下改善接受位置覆盖，
又优于 shuffled、置信度基线及位置/置信度参考，才有充分理由进入该共享集合规则的在线验证。
如果仅优于置信度基线，却落后于位置/置信度参考，说明关联信号可用，但该选择规则还不够好。
如果 oracle 有增益、历史无增益，优先改历史预测；两者都无增益时不强行设计该方法。
若不同任务方向不一致，明确报告任务依赖并补独立样本，不挑结果解释。

## 运行

从 `evaluation/opencompass` 执行，已有 trace 分析只需要 Python + NumPy：

```bash
PYTHON_BIN=/root/miniconda3/envs/team/bin/python \
DATASET=humaneval TRACE_DIR=analysis/records/dependency/<已有运行目录>/traces \
bash scripts/decoding_expert_structure.sh
```

本地现有三个目录：

```bash
DATASET=humaneval TRACE_DIR=analysis/humaneval_20261012_084323_47902/traces bash scripts/decoding_expert_structure.sh
DATASET=mbpp TRACE_DIR=analysis/mbpp_20261012_094425_62942/traces bash scripts/decoding_expert_structure.sh
DATASET=gsm8k TRACE_DIR=analysis/gsm8k_20261012_154006_144488/traces bash scripts/decoding_expert_structure.sh
```

服务器需要新增独立观察样本时，省略 TRACE_DIR，使用双卡：

```bash
export PYTHON_BIN=/root/miniconda3/envs/team/bin/python
export MODEL_PATH=/root/lkd/Models/SDAR-30B-A3B-Chat-b32
export CUDA_VISIBLE_DEVICES=0,1 MODEL_GPUS=2 GPU_MEMORY=37GiB
for dataset in humaneval mbpp gsm8k; do
    DATASET="$dataset" SAMPLE_START=32 SAMPLES=32 MAX_BLOCKS=8 \
    DEPENDENCY_LAYERS=auto bash scripts/decoding_expert_structure.sh
done
```

新增原始记录在 `analysis/records/dependency/`，派生报告在
`analysis/reports/decoding_expert_structure/<dataset>_<时间>_<pid>/`。
核心文件：report.md、summary.json、matched_contrast_summary.csv、rare_summary.csv、
pool_summary.csv、pool_delta_summary.csv；forward 文件保留状态级数据。
pool_overall_summary.csv 汇总各层/预算/规则；pool_overall_delta_summary.csv 与
pool_overall_reference_delta_summary.csv 分别给出相对置信度及位置/置信度参考的配对差值。
