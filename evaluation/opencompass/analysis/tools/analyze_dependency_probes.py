#!/usr/bin/env python3
"""Paired strong/weak diagnostic effects; never report probe time as speedup."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def estimate(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(prompts=0, mean=None, ci95=None)
    ci = None
    if len(values) > 1:
        rng = np.random.default_rng(20261012)
        ci = np.quantile(rng.choice(values, (2000, len(values))).mean(1), [.025, .975]).tolist()
    return dict(prompts=len(values), mean=float(values.mean()), ci95=ci)


def summarize(records):
    attempted = [r for r in records if r['status'] in ('matched', 'no_match')]
    matched = [r for r in attempted if r['status'] == 'matched']
    group = defaultdict(list)
    for row in matched:
        group[row['sample']].append(row)
    effects = ['own_prediction_flip_rate', 'other_prediction_flip_rate',
               'own_confidence_delta', 'other_confidence_abs_delta',
               'own_accepted_lost', 'other_accepted_lost', 'own_accepted_gained',
               'other_accepted_gained', 'baseline_accepted_prediction_flips', 'accepted_set_changes']
    result = {}
    for metric in effects:
        strong, weak, delta = [], [], []
        for rows in group.values():
            usable = [r for r in rows if r['strong']['effects'][metric] is not None
                      and r['weak']['effects'][metric] is not None]
            if not usable:
                continue
            a = np.mean([r['strong']['effects'][metric] for r in usable])
            b = np.mean([r['weak']['effects'][metric] for r in usable])
            strong.append(a); weak.append(b); delta.append(a - b)
        result[metric] = dict(strong=estimate(strong), weak=estimate(weak),
                              strong_minus_weak=estimate(delta))
    return dict(attempted_states=len(attempted), matched_states=len(matched),
                match_rate=len(matched) / len(attempted) if attempted else None,
                prompts_attempted=len({r['sample'] for r in attempted}),
                prompts_matched=len(group), effects=result,
                max_sham_logit_error=max([r['sham_logit_max_abs'] for r in matched], default=None),
                max_per_token_removed_mass_gap=max([
                    max(abs(x['removed_mass'] - y['removed_mass']) for x, y in zip(r['strong']['token_controls'], r['weak']['token_controls']))
                    for r in matched], default=None),
                max_per_token_confidence_gap=max([
                    max(abs(x['confidence'] - y['confidence']) for x, y in zip(r['strong']['token_controls'], r['weak']['token_controls']))
                    for r in matched], default=None),
                notes=['Extra forwards diagnose route restriction; do not claim speedup or task ACC',
                       'Positive strong-minus-weak flip/change effects mean strong pairs are more sensitive',
                       'Confidence delta is signed, not a correctness or harm metric',
                       'Current attention and acceptance are used for offline matching only',
                       'Only matched subset is analyzed; inspect match coverage before generalizing',
                       'Both tokens keep original Top-k; shared pool is fixed-size and protects Top-1',
                       'Original baseline generation uses no intervention outputs',
                       'CI is by prompt; multiple trials are not independent samples'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    files = sorted(args.probe_dir.glob('probe_*.jsonl'))
    if not files:
        p.error('No probes found: check observation range/stride')
    rows, sources, seen = [], [], set()
    for path in files:
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['schema'] != 'sdar_dependency_probe_v1':
                raise ValueError('Unknown probe schema')
            key = (r['sample'], r['block'], r['step'], r['layer'])
            if key in seen:
                raise ValueError('Duplicate probe state')
            seen.add(key)
            if r['status'] == 'matched':
                if r['strong']['stratum'] != r['weak']['stratum']:
                    raise ValueError('Unmatched distance/confidence/overlap/role')
                if set(r['strong']['tokens']) & set(r['weak']['tokens']):
                    raise ValueError('Strong/weak pairs must be disjoint')
                controls = list(zip(r['strong']['token_controls'], r['weak']['token_controls']))
                if any(x['accepted'] != y['accepted'] for x, y in controls):
                    raise ValueError('Per-token acceptance-role mismatch')
                if max(abs(x['removed_mass'] - y['removed_mass']) for x, y in controls) > r['mass_tolerance'] + 1e-8:
                    raise ValueError('Removed-mass mismatch')
                if max(abs(x['confidence'] - y['confidence']) for x, y in controls) > r['confidence_tolerance'] + 1e-8:
                    raise ValueError('Confidence mismatch')
            rows.append(r)
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    settings = {(r['source_sha256'], r['layer'], r['pool_size'], r['mass_tolerance'],
                 r['confidence_tolerance'], r.get('seed'), r.get('threshold')) for r in rows}
    if len(settings) != 1:
        raise ValueError('Do not mix source or probe settings in one report')
    report = summarize(rows)
    report['sources'] = sources
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def fmt(x):
        return 'NA' if x is None else f'{x:.5f}'
    lines = ['# 第 24 层专家共享配对干预', '',
             f"尝试状态：{report['attempted_states']}；匹配状态：{report['matched_states']}；"
             f"有效样本：{report['prompts_matched']}/{report['prompts_attempted']}。", '',
             '| 指标 | 强依赖 | 弱依赖 | 强减弱 | 差值 95% CI |',
             '|---|---:|---:|---:|---|']
    for metric, values in report['effects'].items():
        d = values['strong_minus_weak']
        ci = 'NA' if d['ci95'] is None else f"[{d['ci95'][0]:.5f}, {d['ci95'][1]:.5f}]"
        lines.append(f"| {metric} | {fmt(values['strong']['mean'])} | {fmt(values['weak']['mean'])} | {fmt(d['mean'])} | {ci} |")
    lines += ['', '没有匹配的状态保留为 no_match；不会自动放松匹配条件。',
              '额外 forward 是诊断成本，不是加速方法；结果仅说明路由限制的局部影响。',
              '依赖自身的因果作用仍未识别。稀疏匹配子集不能外推到所有 masked token。']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['attempted_states', 'matched_states', 'prompts_matched', 'match_rate']}))


if __name__ == '__main__':
    main()
