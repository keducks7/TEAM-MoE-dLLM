#!/usr/bin/env python3
"""Offline fixed-top-k expert pools: frontier protection and history ablations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_decoding_expert_structure as structure
from analyze_dependency_motivation import load_trace, stage, write_csv


POLICIES = ('position_confidence_reference', 'protected_shared',
            'protected_dependency', 'protected_shuffled_dependency')
METRICS = structure.POOL_METRICS + [
    'mass_hot', 'mass_cold', 'mass_min_all', 'mass_min_hot',
    'mass_min_accept_now', 'top1_loss_hot', 'hot_target_met_fraction',
    'hot_all_target_met', 'rare_mass_coverage_hot', 'pool_changed_vs_shared',
]


def aligned_budget(requested, union, protected_count, topk):
    if requested < topk:
        raise ValueError('Requested pool budget is smaller than per-token top-k')
    if protected_count > union:
        raise ValueError('Protected experts are outside candidate universe')
    capped = min(requested, union)
    actual = max(capped, protected_count)
    return actual, int(protected_count > capped)


def protected_prefix(ids, mass, hot, budget, target=.9):
    """Protect top-1, then highest missing weight of lowest-covered Hot row.

    Deterministic ties: input row order (physical position), expert ID.
    Stops at target or budget, never expands budget implicitly.
    """
    if not 0 < target <= 1:
        raise ValueError('Target must be in (0,1]')
    rows = np.flatnonzero(hot)
    selected = np.zeros(mass.shape[1], dtype=bool)
    if len(rows):
        selected[np.unique(ids[rows, 0])] = True
    protected_count = int(selected.sum())
    if protected_count > budget:
        raise ValueError('Top-1 protection is infeasible under this budget')
    while len(rows) and selected.sum() < budget:
        coverage = mass[rows][:, selected].sum(1)
        worst = int(np.argmin(coverage))
        if coverage[worst] >= target - 1e-12:
            break
        row = rows[worst]
        missing = ids[row][~selected[ids[row]]]
        if not len(missing):
            raise RuntimeError('Below-target Hot row has no missing original experts')
        order = np.lexsort((missing, -mass[row, missing]))
        selected[missing[order[0]]] = True
    coverage = mass[rows][:, selected].sum(1) if len(rows) else np.array([])
    return selected, dict(protected_top1_count=protected_count,
        protected_prefix_size=int(selected.sum()),
        hot_fill_count=int(selected.sum()) - protected_count,
        remaining_after_protection=budget - int(selected.sum()),
        hot_target_reached_before_fill=int(not len(rows) or np.all(coverage >= target - 1e-12)))


def fill_pool(prefix, score, active, budget, frequency):
    missing = active & ~prefix
    count = budget - int(prefix.sum())
    chosen = structure.choose_pool(score, missing, count, frequency)
    return np.sort(np.concatenate([np.flatnonzero(prefix), chosen]))


def construct_pools(ids, mass, hot, dependency, shuffled, budget, target=.9):
    frequency = np.bincount(ids.ravel(), minlength=mass.shape[1])
    active = frequency > 0
    if budget < ids.shape[1] or budget > active.sum():
        raise ValueError('Invalid pool budget for current candidate universe')
    prefix, info = protected_prefix(ids, mass, hot, budget, target)
    cold = ~hot
    # Shared demand is from remaining Cold rows. If none remain, all rows vote.
    voters = cold if cold.any() else np.ones(len(hot), dtype=bool)
    shared_score = mass[voters].sum(0)
    weights = structure.bounded_weights(dependency[voters])
    shuffled_weights = structure.bounded_weights(shuffled[voters])
    reference_score = mass[hot].sum(0) if hot.any() else mass.sum(0)
    pools = dict(position_confidence_reference=structure.choose_pool(
        reference_score, active, budget, frequency),
        protected_shared=fill_pool(prefix, shared_score, active, budget, frequency),
        protected_dependency=fill_pool(prefix, weights @ mass[voters], active, budget, frequency),
        protected_shuffled_dependency=fill_pool(prefix, shuffled_weights @ mass[voters], active, budget, frequency))
    info['dependency_opportunity'] = int(
        cold.any() and info['remaining_after_protection'] > 0
        and np.ptp(dependency[cold]) > 0)
    return pools, info


def extra_metrics(ids, mass, pool, hot, now, target, shared_pool):
    covered = mass[:, pool].sum(1)
    frequency = np.bincount(ids.ravel(), minlength=mass.shape[1])
    rare = (frequency > 0) & (frequency <= 2)
    rare_mass = mass[:, rare].sum(1)
    covered_rare = mass[:, pool][:, rare[pool]].sum(1)
    eligible = hot & (rare_mass > 0)
    return dict(
        mass_hot=float(covered[hot].mean()) if hot.any() else None,
        mass_cold=float(covered[~hot].mean()) if (~hot).any() else None,
        mass_min_all=float(covered.min()),
        mass_min_hot=float(covered[hot].min()) if hot.any() else None,
        mass_min_accept_now=float(covered[now].min()) if now.any() else None,
        top1_loss_hot=float((~np.isin(ids[hot, 0], pool)).mean()) if hot.any() else None,
        hot_target_met_fraction=float((covered[hot] >= target - 1e-12).mean()) if hot.any() else None,
        hot_all_target_met=int(np.all(covered[hot] >= target - 1e-12)),
        rare_mass_coverage_hot=float((covered_rare[eligible] / rare_mass[eligible]).mean())
        if eligible.any() else None,
        pool_changed_vs_shared=int(set(pool) != set(shared_pool)),
    )


def analyze_trace(path, budgets=(8, 16, 24, 32), target=.9, seed=20261012):
    metadata, d = load_trace(path)
    rows, states = [], []
    counters = defaultdict(int)
    sample = metadata['prompt_sha256']
    for f, mask in enumerate(d['mask']):
        counters['frames'] += 1
        sources = np.flatnonzero(mask)
        if not len(sources):
            continue
        if not structure.adjacent(d, f - 1, f):
            counters['frames_without_history'] += 1
            continue
        future = structure.next_accepted(d, f)
        for li, lid in enumerate(d['layer_ids']):
            history = structure.historical_features(d, f, li)
            hot = history['hot']
            # Current reference Hot is known from prior confidence/current input.
            # Dependency reads ONLY previous attention, to these Hot receivers.
            dependency = structure.dependency_scores(d['attention'][f - 1, li],
                                                      sources[hot], sources)
            random_seed = int.from_bytes(hashlib.sha256(
                f'{seed}/{sample}/{int(d["block_id"][f])}/{int(d["step"][f])}/{int(lid)}'.encode()
            ).digest()[:8], 'little')
            shuffled = dependency.copy()
            # Preserve the Cold dependency distribution in the negative control.
            shuffled[~hot] = np.random.default_rng(random_seed).permutation(dependency[~hot])
            ids = d['expert_ids'][f, li, sources].astype(int)
            probs = d['router_probs'][f, li, sources].astype(float)
            mass = structure.routing_mass(ids, probs)
            union = len(np.unique(ids))
            protected_count = len(np.unique(ids[hot, 0]))
            now = d['accepted'][f, sources]
            raw_dependency = structure.dependency_scores(d['attention'][f, li],
                np.flatnonzero(d['accepted'][f]), sources)
            groups = dict(all=np.ones(len(sources), dtype=bool), accept_now=now,
                accept_next=future[sources] if future is not None else np.zeros(len(sources), dtype=bool),
                remaining=~now, dependency_sources=structure.high_group(raw_dependency))
            for requested in budgets:
                budget, relaxed = aligned_budget(requested, union, protected_count, ids.shape[1])
                pools, info = construct_pools(ids, mass, hot, dependency, shuffled, budget, target)
                base = dict(sample=sample, trace=path.name, block=int(d['block_id'][f]),
                    step=int(d['step'][f]), layer=int(lid), mask_stage=stage(float(mask.mean())),
                    mask_count=len(sources), hot_count=int(hot.sum()), cold_count=int((~hot).sum()),
                    budget_requested=requested, budget_actual=budget, budget_relaxed=relaxed,
                    original_union=union, compressed=int(budget < union),
                    next_labels_available=int(future is not None), **info)
                states.append(base)
                counters['layer_budget_states'] += 1
                counters['budget_infeasible_states'] += relaxed
                counters['dependency_opportunity_states'] += info['dependency_opportunity']
                for policy in POLICIES:
                    pool = np.sort(pools[policy])
                    metrics = structure.pool_metrics(ids, probs, mass, pool, groups)
                    metrics.update(extra_metrics(ids, mass, pool, hot, now, target, pools['protected_shared']))
                    rows.append(dict(base, policy=policy,
                                     pool_ids=' '.join(map(str, pool)), **metrics))
    return metadata, dict(counters), states, rows


def deltas(rows, baseline):
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ['sample', 'trace', 'block', 'step', 'layer', 'budget_requested'])
        grouped[key][row['policy']] = row
    result = []
    for policies in grouped.values():
        control = policies[baseline]
        for row in policies.values():
            if row['budget_actual'] != control['budget_actual']:
                raise ValueError('Policies compared under unequal budgets')
            values = {m: None if row[m] is None or control[m] is None
                      else row[m] - control[m] for m in METRICS}
            result.append(dict(row, comparison_baseline=baseline, **values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budgets', default='8,16,24,32')
    parser.add_argument('--hot-coverage-target', type=float, default=.9)
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    budgets = sorted(set(map(int, args.budgets.split(','))))
    if not budgets or min(budgets) < 1 or not 0 < args.hot_coverage_target <= 1 or args.bootstrap_draws < 1:
        parser.error('Invalid budget/target/bootstrap count')
    files = sorted(args.trace_dir.glob('dependency_*.npz'))
    if not files:
        parser.error('No dependency traces found')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, states, metadata = [], [], []
    seen, signatures, counters = set(), set(), defaultdict(int)
    for path in files:
        info, counts, ss, rr = analyze_trace(path, budgets, args.hot_coverage_target, args.seed)
        if info['prompt_sha256'] in seen:
            raise ValueError('Duplicate prompt traces')
        seen.add(info['prompt_sha256'])
        signatures.add((info['model_source_sha256'], info['threshold'], tuple(info['layer_ids']),
            info.get('seed'), json.dumps(info.get('generation_settings', {}), sort_keys=True)))
        if len(signatures) > 1:
            raise ValueError('Mixed model/settings')
        metadata.append(info)
        records.extend(rr); states.extend(ss)
        for key, value in counts.items():
            counters[key] += value
    summary = lambda rr, keys, metrics: structure.macro_summary(
        rr, keys, metrics, args.bootstrap_draws, args.seed)
    products = {}
    for label, subset in [
        ('strict', [r for r in records if r['compressed'] and not r['budget_relaxed']]),
        ('aligned', [r for r in records if r['compressed']]),
        ('opportunity_strict', [r for r in records if r['compressed'] and not r['budget_relaxed']
                                 and r['dependency_opportunity']]),
    ]:
        keys = ['layer', 'budget_requested', 'policy']
        products[label + '_summary'] = summary(subset, keys, METRICS)
        if label != 'opportunity_strict':
            products[label + '_stage_summary'] = summary(subset, ['layer', 'mask_stage', 'budget_requested', 'policy'], METRICS)
        for baseline in ['position_confidence_reference', 'protected_shared']:
            products[label + '_delta_vs_' + baseline] = summary(deltas(subset, baseline), keys, METRICS)
    products['feasibility_summary'] = summary(states, ['layer', 'budget_requested'], [
        'budget_relaxed', 'budget_actual', 'compressed', 'protected_top1_count',
        'remaining_after_protection', 'hot_target_reached_before_fill', 'dependency_opportunity',
    ])
    for name, rr in [('state_forward', states), ('policy_forward', records), *products.items()]:
        write_csv(args.output_dir / (name + '.csv'), rr)
    report = dict(schema='frontier_expert_pool_v1', prompts=len(seen),
        budgets=budgets, hot_coverage_target=args.hot_coverage_target, seed=args.seed,
        bootstrap_draws=args.bootstrap_draws, trace_dir=str(args.trace_dir.resolve()),
        counters=dict(counters), metadata=metadata, products=products,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        structure_source_sha256=hashlib.sha256(Path(structure.__file__).read_bytes()).hexdigest(),
        notes=['Offline Vanilla trajectory, not a TEAM/online ACC or latency run',
            'Hot definition is previous-confidence/current-position single-trajectory reference',
            'Historical attention goes from reference Hot receivers to Cold sources; self edges excluded',
            'Candidate universe is original top-k union of current masked rows only',
            'Per-token top-k is preserved; original routing mass is a demand proxy',
            'Strict summaries exclude infeasible protection states and noncompressing states',
            'Aligned summaries use the same raised actual budget for every policy in infeasible states',
            'One fixed Cold permutation is a negative control, not a significance test',
            'Threshold and budgets are preset; multiple comparisons not corrected'])
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 前沿需求保护与历史依赖：四版本离线比较', '',
        f'题目数 {len(seen)}；Hot 覆盖目标 {args.hot_coverage_target:.0%}；请求集合容量 {budgets}。', '',
        '主表 strict：仅原请求预算可容纳必保 Top-1、且实际压缩集合的状态。',
        'aligned：预算不可行时，所有规则统一提高到必保数量，单独汇总，避免容量增加被误认作算法收益。', '',
        '| 层 | 请求预算 | 规则 | 题目数 | 本步接受覆盖 | 下一步接受覆盖 | 最低 Hot 覆盖 | Hot Top-1 丢失 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in products['strict_summary']:
        lines.append(f'| {r["layer"]} | {r["budget_requested"]} | {r["policy"]} | {r["prompts"]} | '
            + ' | '.join(structure.fmt(r[m]) for m in ['mass_accept_now', 'mass_accept_next', 'mass_min_hot', 'top1_loss_hot']) + ' |')
    lines += ['', '## 预算可行性和依赖生效空间', '',
        'feasibility_summary.csv 同时报告不可行率、实际容量、Hot 覆盖目标满足率及保护后剩余空间。',
        'opportunity_strict 表只描述依赖有机会生效的子集，不能替代全状态主表。',
        'policy_forward.csv 保存每个候选集合的专家 ID，便于检查变化究竟来自哪里。', '',
        '## 判断', '',
        '首先比较 protected_shared 与位置/置信度参考；再比较 protected_dependency 与 protected_shared 和 shuffled。',
        '同时检查全部/Cold 覆盖损失，不能只挑最低 Hot 覆盖的收益。',
        '若依赖剩余空间很少或集合很少改变，应报告该方案中依赖没有足够发挥空间。',
        '所有结论限于路由覆盖，最终 ACC 和干净运行时间需要后续在线验证。']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(prompts=len(seen), counters=dict(counters), output_dir=str(args.output_dir.resolve()))))


if __name__ == '__main__':
    main()
