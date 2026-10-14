#!/usr/bin/env python3
"""Three observational analyses on unmodified Vanilla dependency traces.

No replay forwards. Oracle acceptance labels are evaluation targets, never
inputs to historical policies. Coverage measures routing demand, not accuracy.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_dependency_motivation import (
    confidence_bin, distance_bin, load_trace, stage, write_csv,
)


PAIR_METRICS = ['overlap', 'shared_topk_mass', 'private_topk_mass', 'top1_equal']
POOL_METRICS = ['mass_all', 'mass_accept_now', 'mass_accept_next',
                'mass_remaining', 'mass_dependency_sources', 'top1_loss_all',
                'top1_loss_accept_now', 'mass_p10_all', 'changed_topk_fraction',
                'active_experts_after', 'active_expert_reduction']


def adjacent(d, previous, current):
    return (0 <= previous < current < len(d['mask'])
            and d['block_id'][previous] == d['block_id'][current]
            and d['step'][current] == d['step'][previous] + 1
            and np.array_equal(d['positions'][previous], d['positions'][current]))


def next_accepted(d, f):
    # None means censored, not "will not be accepted".
    return d['accepted'][f + 1] & d['mask'][f] if adjacent(d, f, f + 1) else None


def routing_mass(ids, probs):
    """Dense normalized ORIGINAL top-k weights; all other weights are zero."""
    ids = np.asarray(ids, dtype=int)
    selected = np.take_along_axis(probs, ids, axis=1).astype(np.float64)
    selected /= np.maximum(selected.sum(1, keepdims=True), 1e-15)
    result = np.zeros(probs.shape, dtype=np.float64)
    np.put_along_axis(result, ids, selected, axis=1)
    return result


def pair_features(ids, mass, i, j):
    membership = np.zeros(mass.shape, dtype=bool)
    np.put_along_axis(membership, ids, True, axis=1)
    common = membership[i] & membership[j]
    shared_mass = ((mass[i] * common).sum(1) + (mass[j] * common).sum(1)) / 2
    return dict(overlap=common.sum(1) / ids.shape[1],
                shared_topk_mass=shared_mass, private_topk_mass=1 - shared_mass,
                top1_equal=(ids[i, 0] == ids[j, 0]).astype(float))


def dependency_scores(attention, receivers, sources):
    """Receiver -> source attention, excluding self edges per source."""
    receivers, sources = np.asarray(receivers, dtype=int), np.asarray(sources, dtype=int)
    if not len(receivers):
        return np.zeros(len(sources), dtype=float)
    values = np.asarray(attention[np.ix_(receivers, sources)], dtype=float)
    valid = receivers[:, None] != sources[None, :]
    return (values * valid).sum(0) / np.maximum(valid.sum(0), 1)


def high_group(values, fraction=.2):
    values = np.asarray(values)
    if len(values) < 2 or np.ptp(values) == 0:
        return np.zeros(len(values), dtype=bool)
    return values >= np.quantile(values, 1 - fraction)


def bounded_weights(values):
    """Fixed [1,2] weighting; no fitted thresholds or test-ACC tuning."""
    values = np.maximum(np.asarray(values, dtype=float), 0)
    return 1 + values / max(float(values.max(initial=0)), 1e-15)


def historical_features(d, f, layer_idx, frontier_fraction=.25):
    if not adjacent(d, f - 1, f):
        return None
    sources = np.flatnonzero(d['mask'][f])
    if not np.all(d['mask'][f - 1, sources]):
        raise ValueError('A position became masked again within a block')
    confidence = d['confidence'][f - 1, sources].astype(float)
    # Only remaining positions; newly accepted previous-step tokens are known
    # decoded positions in the current input, not predicted next receivers.
    count = max(1, int(np.ceil(len(sources) * frontier_fraction)))
    order = np.lexsort((d['positions'][f, sources], -confidence))
    predicted_receivers = sources[order[:count]]
    dependency = dependency_scores(d['attention'][f - 1, layer_idx],
                                   predicted_receivers, sources)
    decoded = np.flatnonzero(~d['mask'][f])
    anchors = np.union1d(decoded, sources[order[:min(2, len(sources))]])
    distance = np.abs(d['positions'][f, sources, None]
                      - d['positions'][f, anchors][None, :]).min(1)
    # Single-trajectory analogue of TEAM's >3, <.7, non-top2 cold rule.
    hot = (distance <= 3) | (confidence >= .7)
    hot[order[:min(2, len(sources))]] = True
    return dict(confidence=confidence, dependency=dependency, hot=hot,
                predicted_receivers=predicted_receivers)


def choose_pool(score, active, budget, tie_score):
    """Exactly budget unique active candidates; stable expert-ID tie break."""
    candidates = np.flatnonzero(active)
    if budget > len(candidates):
        raise ValueError('Budget exceeds active candidate universe')
    order = np.lexsort((candidates, -tie_score[candidates], -score[candidates]))
    return candidates[order[:budget]]


def pool_metrics(ids, probs, mass, pool, groups):
    covered = mass[:, pool].sum(1)
    top1_lost = ~np.isin(ids[:, 0], pool)
    selected = pool[np.argsort(-probs[:, pool], axis=1, kind='stable')[:, :ids.shape[1]]]
    actual = len(np.unique(selected))
    union = len(np.unique(ids))
    result = {}
    for name, rows in groups.items():
        result['mass_' + name] = float(covered[rows].mean()) if rows.any() else None
    result.update(top1_loss_all=float(top1_lost.mean()),
                  top1_loss_accept_now=float(top1_lost[groups['accept_now']].mean())
                  if groups['accept_now'].any() else None,
                  mass_p10_all=float(np.quantile(covered, .1)),
                  changed_topk_fraction=float(np.mean(np.any(~np.isin(ids, pool), axis=1))),
                  active_experts_after=actual,
                  active_expert_reduction=union - actual)
    return result


def analyze_trace(path, budgets=(8, 16, 24, 32), edge_fraction=.2,
                  frontier_fraction=.25, seed=20261012):
    metadata, d = load_trace(path)
    rows_pair, rows_contrast, rows_rare, rows_pool = [], [], [], []
    counters = defaultdict(int)
    sample = metadata['prompt_sha256']
    for f, mask in enumerate(d['mask']):
        sources = np.flatnonzero(mask)
        counters['frames'] += 1
        if not len(sources):
            continue
        future = next_accepted(d, f)
        counters['frames_next_labels_available' if future is not None
                 else 'frames_next_labels_censored'] += 1
        li, lj = np.triu_indices(len(sources), 1)
        i, j = sources[li], sources[lj]
        distance = np.abs(d['positions'][f, i] - d['positions'][f, j])
        conf = (d['confidence'][f, i] + d['confidence'][f, j]) / 2
        now_role = d['accepted'][f, i].astype(int) + d['accepted'][f, j].astype(int)
        next_role = (future[i].astype(int) + future[j].astype(int)) if future is not None else None
        strata = [(distance_bin(int(a)), confidence_bin(float(b))) for a, b in zip(distance, conf)]
        for layer_idx, layer_id in enumerate(d['layer_ids']):
            base = dict(sample=sample, trace=path.name, block=int(d['block_id'][f]),
                        step=int(d['step'][f]), layer=int(layer_id),
                        mask_stage=stage(float(mask.mean())), mask_count=len(sources))
            ids = d['expert_ids'][f, layer_idx, sources].astype(int)
            probs = d['router_probs'][f, layer_idx, sources].astype(float)
            mass = routing_mass(ids, probs)
            frequency = np.bincount(ids.ravel(), minlength=probs.shape[1])
            features = pair_features(ids, mass, li, lj)
            dep = (d['attention'][f, layer_idx, i, j]
                   + d['attention'][f, layer_idx, j, i]) / 2
            if len(dep) >= 2 and np.ptp(dep) > 0:
                lo, hi = np.quantile(dep, [edge_fraction, 1 - edge_fraction])
                # Tied quantiles have no separable strong/weak groups.
                if lo < hi:
                    strong, weak = dep >= hi, dep <= lo
                    for axis, role in [('accept_now', now_role), ('accept_next', next_role)]:
                        if role is None:
                            continue
                        for value in (0, 1, 2):
                            # Future labels apply only to positions that both
                            # remain masked after the current acceptance.
                            eligible = role == value
                            if axis == 'accept_next':
                                eligible &= ~d['accepted'][f, i] & ~d['accepted'][f, j]
                            for group, chosen in [('strong', strong), ('weak', weak)]:
                                selected = eligible & chosen
                                if selected.any():
                                    rows_pair.append(dict(base, role_axis=axis, role_count=value,
                                        dependency_group=group, pairs=int(selected.sum()),
                                        **{m: float(v[selected].mean()) for m, v in features.items()}))
                            for stratum in sorted(set(strata)):
                                chosen = eligible & np.array([s == stratum for s in strata])
                                a, b = chosen & strong, chosen & weak
                                if not a.any() or not b.any():
                                    continue
                                rows_contrast.append(dict(base, role_axis=axis, role_count=value,
                                    distance=stratum[0], confidence_bin=stratum[1],
                                    strong_pairs=int(a.sum()), weak_pairs=int(b.sum()),
                                    **{m: float(v[a].mean() - v[b].mean()) for m, v in features.items()}))
            # Rare means <=2 CURRENT masked tokens route to an expert. Keep
            # mask_count/stage so rarity is not mistaken for a fixed property.
            rare = (frequency <= 2) & (frequency > 0)
            roles = {'accept_now': d['accepted'][f, sources],
                     'remaining_after_now': ~d['accepted'][f, sources]}
            if future is not None:
                roles.update(accept_next=future[sources],
                    remaining_after_next=~(future[sources] | d['accepted'][f, sources]))
            for role_name, selected in roles.items():
                if not selected.any():
                    continue
                rows_rare.append(dict(base, token_role=role_name, tokens=int(selected.sum()),
                    rare_topk_mass=float(mass[selected][:, rare].sum(1).mean()),
                    rare_top1_fraction=float(rare[ids[selected, 0]].mean()),
                    rare_expert_fraction=float(rare.sum() / max((frequency > 0).sum(), 1)),
                    active_experts=int((frequency > 0).sum())))
            history = historical_features(d, f, layer_idx, frontier_fraction)
            if history is None:
                counters['layer_frames_without_history'] += 1
                continue
            counters['layer_frames_with_history'] += 1
            raw_scores = dependency_scores(d['attention'][f, layer_idx],
                np.flatnonzero(d['accepted'][f]), sources)
            dependency = history['dependency']
            rng_seed = int.from_bytes(hashlib.sha256(
                f'{seed}/{sample}/{base["block"]}/{base["step"]}/{layer_id}'.encode()).digest()[:8], 'little')
            shuffled = np.random.default_rng(rng_seed).permutation(dependency)
            confidence_weights = 1 + np.clip(history['confidence'], 0, 1)
            scores = {
                'frequency': frequency.astype(float),
                'probability_sum': probs.sum(0),
                'topk_mass': mass.sum(0),
                'confidence_weighted': confidence_weights @ mass,
                'position_confidence_reference': mass[history['hot']].sum(0),
                'historical_dependency': bounded_weights(dependency) @ mass,
                'confidence_plus_dependency': (confidence_weights * bounded_weights(dependency)) @ mass,
                'confidence_plus_shuffled_dependency': (confidence_weights * bounded_weights(shuffled)) @ mass,
                'oracle_current_dependency': bounded_weights(raw_scores) @ mass,
            }
            active = frequency > 0
            union = int(active.sum())
            k = ids.shape[1]
            groups = dict(all=np.ones(len(sources), dtype=bool),
                accept_now=d['accepted'][f, sources],
                accept_next=future[sources] if future is not None else np.zeros(len(sources), dtype=bool),
                remaining=~d['accepted'][f, sources],
                dependency_sources=high_group(raw_scores, edge_fraction))
            for requested in budgets:
                budget = min(requested, union)
                if budget < k:
                    raise ValueError(f'Budget {requested} is below model top-k={k}')
                for policy, score in scores.items():
                    pool = choose_pool(score, active, budget, frequency)
                    metrics = pool_metrics(ids, probs, mass, pool, groups)
                    rows_pool.append(dict(base, budget_requested=requested,
                        budget_actual=budget, original_union=union,
                        compressed=int(budget < union), policy=policy,
                        predicted_receiver_count=len(history['predicted_receivers']),
                        hot_reference_count=int(history['hot'].sum()),
                        next_labels_available=int(future is not None), **metrics))
    return metadata, dict(counters), rows_pair, rows_contrast, rows_rare, rows_pool


def macro_summary(rows, keys, metrics, draws=1000, seed=20261012):
    """Strata -> frame -> prompt -> equal-prompt average and bootstrap CI."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in rows:
        group = tuple(row[k] for k in keys)
        frame = (row['trace'], row['block'], row['step'])
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                grouped[group][row['sample']][frame][metric].append(value)
    result = []
    rng = np.random.default_rng(seed)
    for group, samples in sorted(grouped.items()):
        row = dict(zip(keys, group), prompts=len(samples),
                   frames=sum(len(frames) for frames in samples.values()))
        for metric in metrics:
            values = []
            for frames in samples.values():
                frame_values = [np.mean(ms[metric]) for ms in frames.values() if ms[metric]]
                if frame_values:
                    values.append(float(np.mean(frame_values)))
            values = np.asarray(values)
            row[metric] = float(values.mean()) if len(values) else None
            row[metric + '_prompts'] = len(values)
            if len(values) > 1:
                bootstrap = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
                low, high = np.quantile(bootstrap, [.025, .975])
                row[metric + '_ci_low'], row[metric + '_ci_high'] = float(low), float(high)
            else:
                row[metric + '_ci_low'] = row[metric + '_ci_high'] = None
        result.append(row)
    return result


def policy_deltas(rows, baseline='confidence_weighted'):
    groups = defaultdict(dict)
    for row in rows:
        key = (row['sample'], row['trace'], row['block'], row['step'],
               row['layer'], row['budget_requested'])
        groups[key][row['policy']] = row
    result = []
    for policies in groups.values():
        control = policies[baseline]
        for name, row in policies.items():
            values = {m: None if row[m] is None or control[m] is None
                      else row[m] - control[m] for m in POOL_METRICS}
            result.append(dict(row, **values, comparison_baseline=baseline))
    return result


def fmt(value):
    return 'NA' if value is None else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budgets', default='8,16,24,32')
    parser.add_argument('--edge-fraction', type=float, default=.2)
    parser.add_argument('--frontier-fraction', type=float, default=.25)
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    budgets = sorted(set(map(int, args.budgets.split(','))))
    if (not budgets or min(budgets) < 1 or not 0 < args.edge_fraction < .5
            or not 0 < args.frontier_fraction <= 1 or args.bootstrap_draws < 1):
        parser.error('Invalid budgets/fractions/bootstrap count')
    files = sorted(args.trace_dir.glob('dependency_*.npz'))
    if not files:
        parser.error('No dependency traces found')
    # Avoid accidentally overwriting an existing experiment report.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata, counts, seen = [], defaultdict(int), set()
    pair, contrasts, rare, pools = [], [], [], []
    signatures = set()
    for path in files:
        info, counter, p, c, r, b = analyze_trace(path, budgets, args.edge_fraction,
                                                args.frontier_fraction, args.seed)
        if info['prompt_sha256'] in seen:
            raise ValueError('Duplicate prompt traces; analyze each dataset/run separately')
        seen.add(info['prompt_sha256'])
        signatures.add((info['model_source_sha256'], info['threshold'], tuple(info['layer_ids']),
            info.get('seed'), json.dumps(info.get('generation_settings', {}), sort_keys=True)))
        if len(signatures) > 1:
            raise ValueError('Mixed model/settings in the trace directory')
        metadata.append(info)
        for key, value in counter.items():
            counts[key] += value
        pair.extend(p); contrasts.extend(c); rare.extend(r); pools.extend(b)
    summary = lambda rows, keys, metrics: macro_summary(rows, keys, metrics,
                                                       args.bootstrap_draws, args.seed)
    pair_keys = ['layer', 'mask_stage', 'role_axis', 'role_count']
    c_summary = summary(contrasts, pair_keys, PAIR_METRICS)
    rare_summary = summary(rare, ['layer', 'mask_stage', 'token_role'],
                          ['rare_topk_mass', 'rare_top1_fraction', 'rare_expert_fraction', 'active_experts'])
    # Exclude noncompressing frames from headline pool comparison. Still save
    # all rows, making saturation and differing requested budgets inspectable.
    compressed = [r for r in pools if r['compressed']]
    pool_keys = ['layer', 'mask_stage', 'budget_requested', 'policy']
    pool_summary = summary(compressed, pool_keys, POOL_METRICS)
    deltas = policy_deltas(compressed)
    delta_summary = summary(deltas, pool_keys, POOL_METRICS)
    reference_deltas = policy_deltas(compressed, 'position_confidence_reference')
    reference_delta_summary = summary(reference_deltas, pool_keys, POOL_METRICS)
    overall_keys = ['layer', 'budget_requested', 'policy']
    overall_pool = summary(compressed, overall_keys, POOL_METRICS)
    overall_delta = summary(deltas, overall_keys, POOL_METRICS)
    overall_reference_delta = summary(reference_deltas, overall_keys, POOL_METRICS)
    for name, rows in [('pair_forward', pair), ('matched_contrast_forward', contrasts),
                       ('matched_contrast_summary', c_summary), ('rare_forward', rare),
                       ('rare_summary', rare_summary), ('pool_forward', pools),
                       ('pool_summary', pool_summary), ('pool_delta_forward', deltas),
                       ('pool_delta_summary', delta_summary),
                       ('pool_reference_delta_summary', reference_delta_summary),
                       ('pool_overall_summary', overall_pool),
                       ('pool_overall_delta_summary', overall_delta),
                       ('pool_overall_reference_delta_summary', overall_reference_delta)]:
        write_csv(args.output_dir / (name + '.csv'), rows)
    report = dict(schema='decoding_expert_structure_v1', prompts=len(seen),
        trace_dir=str(args.trace_dir.resolve()), budgets=budgets,
        edge_fraction=args.edge_fraction, frontier_fraction=args.frontier_fraction,
        seed=args.seed, bootstrap_draws=args.bootstrap_draws, counters=dict(counts),
        pair_contrasts=c_summary, rare_summary=rare_summary, pool_summary=pool_summary,
        pool_delta_summary=delta_summary, pool_reference_delta_summary=reference_delta_summary,
        pool_overall_summary=overall_pool, pool_overall_delta_summary=overall_delta,
        pool_overall_reference_delta_summary=overall_reference_delta, metadata=metadata,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        notes=['Vanilla teacher trajectory; no replay or intervention',
               'Attention is an association proxy, not causal dependence',
               'Next acceptance requires adjacent same-block observed frames; otherwise censored',
               'Normalized original top-k router mass is a demand proxy, not expert output importance',
               'Pool policy comparisons share the same frames/candidate universe/budget',
               'Historical policies use only previous observations plus known current mask/position',
               'Oracle policy/receiver groups use current actual acceptance for retrospective evaluation only',
               'Position-confidence reference is not an exact replay of TEAM four-branch LAC or DCD',
               'All pool evaluations cover current masked rows only; pending decoded rows omitted',
               'One deterministic shuffled feature is a negative control, not a permutation test',
               'Effective expert-count reduction is not measured latency/speedup',
               'CI resamples prompts; small strata are exploratory; multiplicity not corrected'])
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 解码进程与专家共享：三个观察剖析', '',
        f'题目数：{len(seen)}。仅分析 Vanilla 原始轨迹。', '',
        '## 1. 阶段与接受角色：同状态、距离、置信度分组内强关联减弱关联', '',
        '| 层 | 剩余 mask 比例 | 接受角色 | 对中接受数 | 题目数 | 重叠差值 | 共享权重差值 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in c_summary:
        lines.append(f'| {r["layer"]} | {r["mask_stage"]} | {r["role_axis"]} | {r["role_count"]} | '
                     f'{r["prompts"]} | {fmt(r["overlap"])} | {fmt(r["shared_topk_mass"])} |')
    lines += ['', '## 2. 低频专家是否承载关键路由权重', '',
        'rare_summary.csv：低频 = 本状态仅 1–2 个 masked token 使用；按阶段与接受角色报告其权重与 Top-1 比例。',
        'shared_topk_mass 是归一化 Top-k 中共享身份承载的平均权重；不等于专家输出的实际贡献。', '',
        '## 3. 同集合预算的共享规则比较', '',
        'pool_summary.csv：只汇总发生实际集合压缩的状态。pool_forward.csv 同时保存未压缩状态。',
        'pool_delta_summary.csv：逐状态相对 confidence_weighted 的配对差值，再按题目等权。',
        '质量代理：原 Top-k 权重覆盖、本步/下一步接受位置覆盖、Top-1 丢失与低覆盖尾部。',
        '成本代理：强制在集合内重新选 Top-k 后，实际激活的唯一专家数。', '',
        '关键比较：confidence_plus_dependency 对 confidence_weighted、confidence_plus_shuffled_dependency；',
        '另与 probability_sum、frequency、topk_mass、position_confidence_reference 比较。',
        'pool_overall_reference_delta_summary.csv 保存逐状态相对位置/置信度参考的差值，避免只选弱基线。',
        'oracle_current_dependency 只用于诊断当前信号与历史信号差距。', '',
        '| 层 | 预算 | 规则 | 题目数 | 本步接受覆盖 | 下一步接受覆盖 | 全体覆盖 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in overall_pool:
        lines.append(f'| {r["layer"]} | {r["budget_requested"]} | {r["policy"]} | {r["prompts"]} | '
                     f'{fmt(r["mass_accept_now"])} | {fmt(r["mass_accept_next"])} | {fmt(r["mass_all"])} |')
    lines += ['', '## 解释边界', '',
        '历史信息没有跨 block 或跳步；下一步缺失不会当作未接受。',
        '稀疏分组同时查看题目数和置信区间，不以 token 对数量替代独立题目数。',
        '路由覆盖改善支持进入在线方法验证；本报告不能证明 ACC 或速度改善。', '',
        '所有配置、输入元数据、计数与分析代码哈希见 summary.json。']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(prompts=len(seen), counters=dict(counts),
                         output_dir=str(args.output_dir.resolve()))))


if __name__ == '__main__':
    main()
