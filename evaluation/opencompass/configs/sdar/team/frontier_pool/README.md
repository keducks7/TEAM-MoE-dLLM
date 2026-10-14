# TEAM 中的前沿专家集合选择与生成计时

普通 modeling_sdar_moe.py 执行新算法，普通 TEAM wrapper 保持 DCD 和 SEH。
本目录不使用 Mark wrapper、metric forward hook、attention observer 或干预 probe。
历史依赖/打乱依赖目前仍是离线剖析。在线方法包括 TEAM、reference、protected，
以及只改变 protected 余量投票者的机制对照 protected_hot。

## 方法总览

| METHOD | 专家集合策略 | DCD / SEH |
|---|---|---|
| team | 原始 TEAM LAC，未启用新集合策略 | 保留 |
| reference | 原始 Hot 的 Top-8 权重聚合，选择共享集合 | 保留 |
| protected | Hot Top-1 保护，逐次补足覆盖最低的 Hot，剩余容量按 Cold 权重聚合 | 保留 |
| protected_hot | 与 protected 完全相同的保护步骤，剩余容量改按 Hot 权重聚合 | 保留 |

候选来自四个实际执行分支中所有需要计算行的原始 Top-8 并集；每 token 仍为 Top-8。
共享池跨四分支选择，不会只看最终分支，也不把四个分支误当成独立 bs=4。
稳定 decoded 行沿用 DCD 的选中父分支缓存。仍需计算的非 mask 行保留完整原始 Top-8 路由，
包括上一步刚接受、等待 DCD 延迟更新的位置，以及本轮 SEH 分支新填入的位置。
代码中的 pending 是这两类行的统称，不是只指 DCD，也不是旧块前缀。
Hot/Cold 在新选择器中仅指仍为 mask 的位置；原 TEAM 的非 Cold 必要集合同时包含 Hot 与 pending。
这些非 mask 行的专家并集占用共享池容量并计入 APF，与 Hot/Cold 所选专家可以重叠，不能相加计数。
旧块前缀提供 KV，稳定已缓存的行不参与本次 MoE，不会占用本次专家池。
输入状态不同引起的后续预测/接受轨迹变化属于策略效应。

TEAM_POOL_BUDGET 是每层每次 speculative forward 的容量上限，默认 36；不是目标 APF。
为避免原 TEAM 已经只需要少量专家的层反而增加开销，容量还被当前原 TEAM 的
Hot＋待更新行专家并集大小截断。候选身份仍可从全部需要计算行中选择。
如果“仍需计算的非 mask 行完整 Top-8＋Hot Top-1”超过请求容量，各实验规则用同一可行性约束
扩大实际容量，并记录 pool_budget_relaxations、pool_mean_actual_budget。
不将扩容静默当成固定预算节省。实际容量始终不超过该输入状态原 TEAM 的必要集合大小。

reference 在容量足够覆盖原 TEAM 集合时恢复原 LAC 集合；这项 CPU 源码对照通过。
protected 的覆盖目标是归一化原始 Top-8 路由权重，默认 0.9，不能当作 90% 质量保证。
第一轮 forward、prompt prefill 与最后 KV 写回保持原始完整计算，故全局 APF 不直接等于池容量。
普通算法与旧 Cold Budget/Cold 复用/Hot 裁剪不能同时开启；配置已将这些旧实验开关关闭。
贪心选择本身有开销，本原型不预先承诺端到端加速，计时会包含它。

## protected_hot 机制对照

只改变最后一个步骤：达到 Hot 覆盖目标后，余量由 Hot 而非 Cold 的原始 Top-8 权重聚合填充。
预算 36、覆盖目标 0.9、Top-1 保护、非 mask 完整 Top-8 保护、DCD/SEH 均保持不变。
Cold 仍参与计算并在同一个共享池中选 Top-8，不是跳过 Cold，也不是 Hot-only 计算。
没有 Hot 时沿用原 fallback；没有余量时，两个 protected 规则在同一输入上选择相同集合。
对照回答“过早停止补足 Hot、转而为 Cold 分配余量是否造成退化”，不预先承诺恢复质量或速度。

新 JSONL 记录 pool_remainder_layer_forwards 与 pool_remainder_slots_sum，检查该差异实际作用的空间。
同时在生成计时结束后记录 generation_stop_reason（stop_token / length_limit / incomplete）、
generation_capacity_tokens 和 generation_capacity_filled；不改变接受和停止规则。
capacity 是块对齐后输出缓冲区的真实长度，可能略大于 gen_length，例如 4123 而非 4096。
这些是终止结果的诊断信息，不是逐层逐步 logit 轨迹。

## 两种运行模式

- MODE=acc：真实生成并评估 ACC；默认同时记录生成耗时，不注册统计 hook、不统计专家数。
  token 数在计时结束后从原始生成结果统计，避免每个 forward 为统计做 GPU 同步。
- MODE=metrics：同样生成并评估 ACC，同时在实际 MoE 路由中统计 APF/TPF/APT 和耗时。
  使用 native counters，不注册统计 hook；耗时包含这些专家统计的开销。
- COLLECT_LATENCY=0 可关闭计时。MODE=acc 且此开关为 0 时不写统计 JSONL。

计时以 perf_counter＋所有模型 CUDA 分片的前后同步实现：

| 包含 | 不包含 |
|---|---|
| prompt prefill、去噪 forward、DCD/SEH、集合选择、KV 写回 | 模型加载、分词、答案解码/后处理、JSON 写盘、ACC evaluator |

汇总输出总生成秒数、每样本平均/中位/p95、接受 token/s。
它是 generation wall-clock，不是整个 OpenCompass 命令耗时，也不是纯 MoE kernel latency。
不同题目长度不同，比较时使用相同样本，硬件保持空闲；首样本可能含 kernel 初始化开销。
严格计时用 MODE=acc；不要把 metrics 计时与 acc 计时混算 speedup。

新配置 TPF 使用实际 mask→token 转换数，包含被 SEH 验证的预填 token。
另外保留 transfer_accepted_tokens 供指标运行核对旧口径。历史 hook 默认只数 transfer_index，
不能把其旧 TPF 与新 TPF 不加说明地混在一起。accepted_token_mode 已写入每条记录。
原 Mark JSON 的 APF/TPF 仍可由 calculate_team_metrics.py 读取；Mark 文件没有新增生成计时。
Mark 路由未同步新策略，不能用未修改 Mark 测新方法。本次新方法的指标直接由普通路径得到。

## 服务器准备

上传/更新这些源码：

- modeling_sdar_moe.py
- evaluation/opencompass/opencompass/models/huggingface_bd3_decoded_jump_expert_limit_speculative.py
- evaluation/opencompass/configs/sdar/team/frontier_pool/ （本目录）
- evaluation/opencompass/scripts/team_frontier_pool.sh
- evaluation/opencompass/scripts/run_team_frontier_pool.py
- evaluation/opencompass/scripts/test_team_frontier_runtime.py
- evaluation/opencompass/analysis/tools/calculate_team_metrics.py
- evaluation/opencompass/analysis/tools/compare_team_latency.py

必须同时更新 checkpoint 中的普通模型源码。不要替换 Mark 文件：

```bash
cd /root/lkd/TEAM-MoE-dLLM
cp -n /root/lkd/Models/SDAR-30B-A3B-Chat-b32/modeling_sdar_moe.py \
      /root/lkd/Models/SDAR-30B-A3B-Chat-b32/modeling_sdar_moe.before_frontier.py
cp modeling_sdar_moe.py /root/lkd/Models/SDAR-30B-A3B-Chat-b32/modeling_sdar_moe.py
cd evaluation/opencompass
export PYTHON_BIN=/root/miniconda3/envs/team/bin/python
export MODEL_PATH=/root/lkd/Models/SDAR-30B-A3B-Chat-b32
export CUDA_VISIBLE_DEVICES=0,1 MODEL_GPUS=2 GPU_MEMORY=37GiB
export SEED=42
"$PYTHON_BIN" scripts/test_team_frontier_runtime.py
```

新配置也检查实际加载的 MoE 是否包含新接口，避免旧 HF 模块或 Mark 代码被误用。
所有对照用同一更新后的源码。新开关默认 none 时的普通 MoE 输出已有 CPU 验证。
逐样本 RNG 由固定 seed 和 prompt token ID 哈希决定，并在结束后恢复 RNG；
一个方法使用更多 forward 不会改变下一道题的起始随机数状态。

## 运行检查与 ACC 实验

默认 SEED=42、SAMPLES=all。先运行上面的 CPU 源码测试，再做全量 HumanEval。
当前服务器 human-eval-clean 对完整题库执行覆盖检查，8/64 题子集会触发
“Some problems are not attempted”；不要直接删除该断言。未修复子集评估前使用 SAMPLES=all。

只运行本次新增组：

```bash
SEED=42 TEAM_POOL_BUDGET=36 TEAM_POOL_COVERAGE_TARGET=0.9 \
DATASET=humaneval METHOD=protected_hot MODE=acc SAMPLES=all \
bash scripts/team_frontier_pool.sh
```

旧结果可作探索性参考；严格速度对照应在本次同一更新源码、同一环境下运行。
compare_team_latency.py 会拒绝新旧模型源码哈希不同的记录，不应绕过检查。
同一更新源码下运行原 TEAM 与两组 protected（reference 已有，不是本次机制对照的唯一差异）：

```bash
export TEAM_POOL_BUDGET=36 TEAM_POOL_COVERAGE_TARGET=0.9
(
set -e
for method in team protected protected_hot; do
    DATASET=humaneval METHOD="$method" MODE=acc SAMPLES=all \
    bash scripts/team_frontier_pool.sh
done
)
```

单独带 APF/TPF/APT＋时间：

```bash
DATASET=humaneval METHOD=protected_hot MODE=metrics SAMPLES=all bash scripts/team_frontier_pool.sh
```

DATASET 可换为 mbpp、gsm8k 或 math；SAMPLES=all 使用完整配置数据集。
单张 A800 80G 可把设备配置改为 CUDA_VISIBLE_DEVICES=0 MODEL_GPUS=1 GPU_MEMORY=74GiB。
每次创建独立运行目录，防止追加旧指标或复用旧预测。

## 输出与比较

输出在 outputs/team_frontier_pool/<任务_方法_预算_模式_seed.随机后缀>/：

- console.log：整个流程的日志，末尾追加指标/时间/终止诊断汇总。
- manifest.json / resolved_config.py：样本范围、seed、源码哈希、配置。
- opencompass/：预测、ACC 结果和 OpenCompass 汇总。
- records/team_metrics_pid*.jsonl：逐样本统计（acc 模式为耗时/TPF，APF 不可用）。

每个脚本完成后自动汇总已有记录。也可执行：

```bash
"$PYTHON_BIN" analysis/tools/calculate_team_metrics.py --metrics-dir <本次运行目录>/records
"$PYTHON_BIN" analysis/tools/compare_team_latency.py \
    --baseline-dir <同样本同模式TEAM目录>/records \
    --candidate-dir <同样本同模式protected目录>/records
```

speedup 为同样本 TEAM 总生成耗时 / 新方法总生成耗时。
比较工具检查 prompt、seed、模型源码、生成参数和统计模式；不能把旧 Mark JSON 当计时基线。
新增 protected_hot 已在本地完成 CPU 源码级检查，没有该策略的 30B 模型/GPU 实测。
本地测试同时验证旧 reference/protected 选择器与更新前源码输出一致。
真实生成效果必须以服务器结果为准，不根据 CPU 玩具模型推断 ACC 或加速。
