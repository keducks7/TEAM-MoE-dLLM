# TEAM 中的前沿专家集合选择与生成计时

普通 modeling_sdar_moe.py 执行新算法，普通 TEAM wrapper 保持 DCD 和 SEH。
本目录不使用 Mark wrapper、metric forward hook、attention observer 或干预 probe。
历史依赖/打乱依赖目前仍是离线剖析，本次上线验证先比较 TEAM、reference、protected。

## 方法总览

| METHOD | 专家集合策略 | DCD / SEH |
|---|---|---|
| team | 原始 TEAM LAC，未启用新集合策略 | 保留 |
| reference | 原始 Hot 的 Top-8 权重聚合，选择共享集合 | 保留 |
| protected | Hot Top-1 保护，逐次补足覆盖最低的 Hot，剩余容量按 Cold 权重聚合 | 保留 |

候选来自四个实际执行分支中所有需要计算行的原始 Top-8 并集；每 token 仍为 Top-8。
共享池跨四分支选择，不会只看最终分支，也不把四个分支误当成独立 bs=4。
稳定 decoded 行沿用 DCD 的选中父分支缓存。待 DCD 更新的行保留其完整原始 Top-8 路由，
这些专家同时占用共享池容量并计入 APF，不能只统计 mask 行。
输入状态不同引起的后续预测/接受轨迹变化属于策略效应。

TEAM_POOL_BUDGET 是每层每次 speculative forward 的容量上限，默认 36；不是目标 APF。
为避免原 TEAM 已经只需要少量专家的层反而增加开销，容量还被当前原 TEAM 的
Hot＋待更新行专家并集大小截断。候选身份仍可从全部需要计算行中选择。
如果“待更新行完整 Top-8＋Hot Top-1”超过请求容量，两种实验规则用同一可行性约束
扩大实际容量，并记录 pool_budget_relaxations、pool_mean_actual_budget。
不将扩容静默当成固定预算节省。实际容量始终不超过该输入状态原 TEAM 的必要集合大小。

reference 在容量足够覆盖原 TEAM 集合时恢复原 LAC 集合；这项 CPU 源码对照通过。
protected 的覆盖目标是归一化原始 Top-8 路由权重，默认 0.9，不能当作 90% 质量保证。
第一轮 forward、prompt prefill 与最后 KV 写回保持原始完整计算，故全局 APF 不直接等于池容量。
普通算法与旧 Cold Budget/Cold 复用/Hot 裁剪不能同时开启；配置已将这些旧实验开关关闭。
贪心选择本身有开销，本原型不预先承诺端到端加速，计时会包含它。

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
export SEED=20261014
"$PYTHON_BIN" scripts/test_team_frontier_runtime.py
```

新配置也检查实际加载的 MoE 是否包含新接口，避免旧 HF 模块或 Mark 代码被误用。
所有对照用同一更新后的源码。新开关默认 none 时的普通 MoE 输出已有 CPU 验证。
逐样本 RNG 由固定 seed 和 prompt token ID 哈希决定，并在结束后恢复 RNG；
一个方法使用更多 forward 不会改变下一道题的起始随机数状态。

## 小规模检查与 ACC 实验

先跑 8 题检查服务器模型/配置加载：

```bash
DATASET=humaneval METHOD=team MODE=acc SAMPLES=8 SAMPLE_START=32 bash scripts/team_frontier_pool.sh
DATASET=humaneval METHOD=reference MODE=acc SAMPLES=8 SAMPLE_START=32 bash scripts/team_frontier_pool.sh
DATASET=humaneval METHOD=protected MODE=acc SAMPLES=8 SAMPLE_START=32 bash scripts/team_frontier_pool.sh
```

确认完成后，同一 64 题做 HumanEval/MBPP 三组对照：

```bash
export TEAM_POOL_BUDGET=36 TEAM_POOL_COVERAGE_TARGET=0.9
for dataset in humaneval mbpp; do
    for method in team reference protected; do
        DATASET="$dataset" METHOD="$method" MODE=acc SAMPLES=64 SAMPLE_START=32 \
        bash scripts/team_frontier_pool.sh
    done
done
```

单独带 APF/TPF/APT＋时间：

```bash
DATASET=humaneval METHOD=team MODE=metrics SAMPLES=64 SAMPLE_START=32 bash scripts/team_frontier_pool.sh
DATASET=humaneval METHOD=reference MODE=metrics SAMPLES=64 SAMPLE_START=32 bash scripts/team_frontier_pool.sh
DATASET=humaneval METHOD=protected MODE=metrics SAMPLES=64 SAMPLE_START=32 bash scripts/team_frontier_pool.sh
```

DATASET 可换为 mbpp、gsm8k 或 math；SAMPLES=all 使用完整配置数据集。
单张 A800 80G 可把设备配置改为 CUDA_VISIBLE_DEVICES=0 MODEL_GPUS=1 GPU_MEMORY=74GiB。
每次创建独立运行目录，防止追加旧指标或复用旧预测。

## 输出与比较

输出在 outputs/team_frontier_pool/<任务_方法_预算_模式_seed.随机后缀>/：

- console.log：整个流程的日志。
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
本地完成 CPU 源码级检查，没有 30B 模型/GPU 实测；服务器 8 题检查是进入完整评估前的必要验证。
