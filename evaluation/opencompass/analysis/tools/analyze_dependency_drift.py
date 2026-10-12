#!/usr/bin/env python3
"""Describe history failure around acceptance events; no online threshold fitting."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_dependency_motivation import load_trace, ranks, spearman, stage, strong_edge_recall


def adjacent(data, f):
    return (f > 0 and data['block_id'][f] == data['block_id'][f - 1]
            and data['step'][f] == data['step'][f - 1] + 1
            and np.array_equal(data['positions'][f], data['positions'][f - 1]))


def partial_rank_correlation(x, y, controls):
    """Descriptive rank residuals, not a predictive model or causal adjustment."""
    x, y = np.asarray(x), np.asarray(y)
    c = np.asarray(controls)
    if len(x) < max(12, c.shape[1] + 5):
        return None
    design = np.column_stack([np.ones(len(x)), *[ranks(c[:, i]) for i in range(c.shape[1])]])
    rx, ry = ranks(x), ranks(y)
    rx -= design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry -= design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    norm = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / norm) if norm > 1e-8 else None


def drift_records(path):
    meta, d = load_trace(path)
    result = []
    for f in range(1, len(d['mask'])):
        if not adjacent(d, f):
            continue
        surviving = d['mask'][f] & d['mask'][f - 1]
        indices = np.flatnonzero(surviving)
        if len(indices) < 4:
            continue
        li, lj = np.triu_indices(len(indices), 1)
        i, j = indices[li], indices[lj]
        accepted = d['accepted'][f - 1]
        old_mask = d['mask'][f - 1]
        if np.any(accepted & ~old_mask):
            raise ValueError(f'Invalid accepted positions: {path}')
        for layer_index, layer in enumerate(d['layer_ids']):
            current = d['attention'][f, layer_index]
            old = d['attention'][f - 1, layer_index]
            s = (current[i, j] + current[j, i]) / 2
            previous = (old[i, j] + old[j, i]) / 2
            rho = spearman(s, previous)
            recall = strong_edge_recall(s, previous)
            if rho is None or recall is None:
                continue
            # All signals below are available AFTER previous acceptance, BEFORE
            # current forward. Never use the current acceptance set as a feature.
            context_out = old[np.ix_(indices, np.flatnonzero(accepted))].sum(-1)
            context_in = old[np.ix_(np.flatnonzero(accepted), indices)].sum(0)
            support = (context_out + context_in) / 2
            pair_support = (support[li] + support[lj]) / 2
            rank_error = np.abs(ranks(s) - ranks(previous)) / max(1, len(s) - 1)
            result.append(dict(
                sample=meta['prompt_sha256'], trace=path.name,
                layer=int(layer), block=int(d['block_id'][f]), step=int(d['step'][f]),
                surviving_masks=int(len(indices)), mask_ratio=float(surviving.mean()),
                mask_stage=stage(float(surviving.mean())), pairs=int(len(i)),
                newly_accepted=int(accepted.sum()),
                accepted_fraction=float(accepted.sum() / old_mask.sum()),
                historical_accepted_context_mass=float(support.mean()),
                historical_accepted_context_mass_max=float(support.max()),
                prior_confidence_mean=float(d['confidence'][f - 1, indices].mean()),
                history_rho=rho, strong_edge_recall=recall,
                rank_drift=float(rank_error.mean()),
                context_vs_pair_drift_rho=spearman(pair_support, rank_error)))
    return meta, result


def estimate(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return dict(prompts=0, mean=None, ci95=None)
    ci = None
    if len(v) >= 2:
        rng = np.random.default_rng(20261012)
        ci = np.quantile(rng.choice(v, (2000, len(v))).mean(1), [.025, .975]).tolist()
    return dict(prompts=len(v), mean=float(v.mean()), ci95=ci)


def summarize(rows):
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[(row['layer'], row['sample'])].append(row)
    features = ['newly_accepted', 'accepted_fraction', 'historical_accepted_context_mass',
                'historical_accepted_context_mass_max']
    per_layer = defaultdict(lambda: defaultdict(list))
    for (layer, _sample), records in by_prompt.items():
        y = [1 - r['history_rho'] for r in records]
        for feature in features:
            x = [r[feature] for r in records]
            per_layer[layer][feature + '_vs_history_failure_rho'].append(spearman(x, y))
        controls = [[r['mask_ratio'], r['pairs'], r['accepted_fraction'],
                     r['prior_confidence_mean']] for r in records]
        per_layer[layer]['context_mass_partial_rho'].append(partial_rank_correlation(
            [r['historical_accepted_context_mass'] for r in records], y, controls))
        valid = [r['context_vs_pair_drift_rho'] for r in records
                 if r['context_vs_pair_drift_rho'] is not None]
        per_layer[layer]['context_vs_pair_drift_rho'].append(float(np.mean(valid)) if valid else None)
    correlations = {str(layer): {name: estimate(values) for name, values in metrics.items()}
                    for layer, metrics in sorted(per_layer.items())}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        n = row['newly_accepted']
        bucket = '0' if n == 0 else '1' if n == 1 else '2-3' if n <= 3 else '4-7' if n <= 7 else '8+'
        groups[(row['layer'], row['mask_stage'], bucket)][row['sample']].append(row)
    event_table = []
    for (layer, mask_stage, bucket), prompts in sorted(groups.items()):
        event_table.append(dict(layer=layer, mask_stage=mask_stage, accepted_bucket=bucket,
            states=sum(map(len, prompts.values())),
            history_rho=estimate([np.mean([r['history_rho'] for r in rs]) for rs in prompts.values()]),
            strong_edge_recall=estimate([np.mean([r['strong_edge_recall'] for r in rs]) for rs in prompts.values()])))
    return correlations, event_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    files = sorted(args.trace_dir.glob('dependency_*.npz'))
    if not files:
        p.error('No dependency traces')
    rows, sources, seen, signatures = [], [], set(), set()
    for file in files:
        meta, records = drift_records(file)
        if meta['prompt_sha256'] in seen:
            raise ValueError('Duplicate prompt; do not mix repeated experiments')
        seen.add(meta['prompt_sha256'])
        signatures.add((meta['model_source_sha256'], meta['threshold'], meta.get('seed'),
                        tuple(meta['layer_ids']),
                        json.dumps(meta.get('generation_settings', {}), sort_keys=True)))
        rows.extend(records)
        sources.append(dict(path=str(file.resolve()), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    if len(signatures) != 1:
        raise ValueError('Do not mix model source/seed/generation settings in one drift report')
    correlations, event_table = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(prompts=len(seen), adjacent_states_by_layer=len(rows),
                  correlations=correlations, event_table=event_table, sources=sources,
                  interpretation=['history_failure=1-Spearman, not task accuracy loss',
                      'positive feature correlation means more history drift, not causality',
                      'confidence/mask controls and all signals are from recorded baseline only',
                      'partial ranks are exploratory descriptive residual correlations',
                      'no online refresh threshold is fitted',
                      'no future steps or block boundaries are used as history'])
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output_dir / 'states.jsonl').open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row) + '\n')
    lines = ['# 解码事件与历史依赖失效', '', f'样本数：{len(seen)}。正相关表示历史关系更不稳定，不表示 ACC 下降。', '',
             '| 层 | 历史信号 | 有效样本 | 与历史失效的 rho | 95% CI |',
             '|---|---|---:|---:|---|']
    for layer, metrics in correlations.items():
        for feature, result in metrics.items():
            mean = 'NA' if result['mean'] is None else f"{result['mean']:.4f}"
            ci = 'NA' if result['ci95'] is None else f"[{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}]"
            lines.append(f"| {layer} | {feature} | {result['prompts']} | {mean} | {ci} |")
    lines += ['', '条件分组表、原始状态记录及来源哈希见 summary.json / states.jsonl。',
              '不能凭相关性直接设置更新阈值；应结合配对干预、独立验证和实际开销。']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(prompts=len(seen), records=len(rows), correlations=correlations)))


if __name__ == '__main__':
    main()
