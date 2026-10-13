#!/usr/bin/env python3
"""Prompt-level effects and held-out predictive checks for frontier probes.

Strict matches and stratified unmatched observations are reported separately.
Current frontier is an offline oracle.
Ridge fits are analysis diagnostics, not training the language model/policy.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_dependency_probes import estimate


METRICS = ('frontier_prediction_flip_rate', 'frontier_acceptance_lost_rate',
           'frontier_logprob_delta', 'frontier_logprob_abs_delta',
           'frontier_margin_delta', 'frontier_margin_abs_delta')
ROUTER = ['source_confidence', 'removed_mass', 'replacement_mass',
          'frontier_count', 'mask_count', 'baseline_frontier_margin_mean']
GEOMETRY = ['nearest_frontier_distance', 'mean_frontier_distance']
HISTORY = ['historical_dependency_to_current_frontier',
           'historical_dependency_to_predicted_frontier']


def validate(row):
    if row.get('schema') != 'sdar_frontier_probe_v1':
        raise ValueError('Unsupported frontier schema')
    if row['status'] not in ('matched', 'observed_unmatched', 'no_match', 'control_failed'):
        raise ValueError('Unknown status')
    if row['status'] == 'no_match':
        return
    if not row['frontier'] or len(set(row['frontier'])) != len(row['frontier']):
        raise ValueError('Invalid frozen frontier')
    a, b = row['strong'], row['weak']
    if a['token'] == b['token']:
        raise ValueError('Source identities must differ')
    if a['dependency'] <= b['dependency']:
        raise ValueError('Strong dependency must exceed weak')
    if row['selection'] == 'strict_matched':
        if a['nearest_frontier_distance'] != b['nearest_frontier_distance']:
            raise ValueError('Unmatched distances')
        if max(abs(a[k] - b[k]) for k in ('removed_mass', 'replacement_mass')) > row['mass_tolerance'] + 1e-8:
            raise ValueError('Unmatched router losses')
        if abs(a['source_confidence'] - b['source_confidence']) > row['confidence_tolerance'] + 1e-8:
            raise ValueError('Unmatched confidence')
        if tuple(a['source_confidence'] < t for t in (.5, .9)) != tuple(b['source_confidence'] < t for t in (.5, .9)):
            raise ValueError('Unmatched confidence bins')
    elif row['selection'] != 'stratified_unmatched':
        raise ValueError('Unknown selection protocol')
    if row['status'] == 'matched' and row['selection'] != 'strict_matched':
        raise ValueError('Unmatched observations must not be reported as matched')
    if row['status'] == 'observed_unmatched' and row['selection'] != 'stratified_unmatched':
        raise ValueError('Inconsistent selection/status')
    for side in ('strong', 'weak'):
        source = row[side]
        if source['group'] != side:
            raise ValueError('Incorrect dependency group')
        old, new = source['original_pool'], source['replacement_pool']
        if source['token'] in row['frontier']:
            raise ValueError('Source must not be in the original frontier')
        if (len(old) != 8 or len(new) != 8 or len(set(old)) != 8 or len(set(new)) != 8
                or old[:-1] != new[:-1] or old[-1] in new or new[-1] in old):
            raise ValueError('Expected exactly one rank-8 replacement, protecting Top-1')
        if row['status'] != 'control_failed' and not source['control_decisions_equal']:
            raise ValueError('Failed reconstruction control cannot be included')
        for kind in ('effects', 'control_effects', 'relative_to_control'):
            effect = source[kind]
            if any(not np.isfinite(effect[m]) for m in METRICS):
                raise ValueError('Non-finite effect')
            for key in ('receiver_prediction_flips', 'receiver_acceptance_lost',
                        'receiver_logprob_delta', 'receiver_margin_delta'):
                if len(effect[key]) != len(row['frontier']):
                    raise ValueError('Receiver alignment mismatch')


def load_records(directory):
    files = sorted(directory.glob('frontier_*.jsonl'))
    if not files:
        raise ValueError('No frontier probe files')
    records, sources, seen, settings = [], [], set(), set()
    for file in files:
        for line in file.read_text().splitlines():
            row = json.loads(line)
            validate(row)
            key = (row['sample'], row['block'], row['step'], row['layer'])
            if key in seen:
                raise ValueError('Duplicate frontier state')
            seen.add(key)
            settings.add((*[row.get(k) for k in ('layer', 'source_sha256', 'protocol_sha256', 'replay_sha256',
                         'seed', 'threshold', 'mass_tolerance', 'confidence_tolerance')],
                         json.dumps(row.get('generation_settings', {}), sort_keys=True)))
            records.append(row)
        sources.append(dict(path=str(file.resolve()), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    if len(settings) != 1:
        raise ValueError('Mixed protocol/model/settings: analyze runs separately')
    return records, sources


def paired_effects(records, kind):
    grouped = defaultdict(list)
    for row in records:
        grouped[row['sample']].append(row)
    result = {}
    for metric in METRICS:
        a = [np.mean([r['strong'][kind][metric] for r in rows]) for rows in grouped.values()]
        b = [np.mean([r['weak'][kind][metric] for r in rows]) for rows in grouped.values()]
        result[metric] = dict(strong=estimate(a), weak=estimate(b),
                              strong_minus_weak=estimate(np.asarray(a) - b))
    return result


def held_out_mse(rows, features, outcome):
    """Leave one entire prompt out; training standardization and weights only."""
    samples = sorted({r['sample'] for r in rows})
    result = {}
    x = np.asarray([[r[k] for k in features] for r in rows], dtype=float)
    y = np.asarray([r[outcome] for r in rows], dtype=float)
    ids = np.asarray([r['sample'] for r in rows])
    for sample in samples:
        train, test = ids != sample, ids == sample
        train_counts = Counter(ids[train])
        weights = np.asarray([1 / train_counts[s] for s in ids[train]])
        # Mean-one weights keep fixed ridge penalty comparable across folds.
        weights *= len(weights) / weights.sum()
        mean = np.average(x[train], axis=0, weights=weights)
        scale = np.sqrt(np.average((x[train] - mean)**2, axis=0, weights=weights))
        scale = np.maximum(scale, 1e-8)
        design = np.c_[np.ones(train.sum()), (x[train] - mean) / scale]
        penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
        coef = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                               design.T @ (weights * y[train]))
        pred = np.c_[np.ones(test.sum()), (x[test] - mean) / scale] @ coef
        result[sample] = float(np.mean((pred - y[test])**2))
    return result


def predictive_checks(records):
    rows = []
    for record in records:
        for side in ('strong', 'weak'):
            row = dict(record[side], sample=record['sample'],
                       baseline_frontier_margin_mean=float(np.mean(record['baseline_frontier_margin'])))
            row.update(row['relative_to_control'])
            rows.append(row)
    outcome = 'frontier_margin_abs_delta'
    panels = {}
    for label, usable in [('current', rows), ('history_complete_cases',
            [r for r in rows if all(r.get(k) is not None for k in HISTORY)])]:
        n = len({r['sample'] for r in usable})
        if n < 8:
            panels[label] = dict(prompts=n, observations=len(usable), status='insufficient_prompts',
                                 reason='At least 8 independent prompts required; this is not a power guarantee')
            continue
        models = {'router_controls': ROUTER, 'router_distance': ROUTER + GEOMETRY,
                  'plus_current_dependency': ROUTER + GEOMETRY + ['dependency'],
                  'plus_shuffled_dependency': ROUTER + GEOMETRY + ['shuffled_dependency']}
        if label == 'history_complete_cases':
            models.update(plus_old_graph_current_frontier=ROUTER + GEOMETRY + [HISTORY[0]],
                          plus_old_graph_predicted_frontier=ROUTER + GEOMETRY + [HISTORY[1]])
        mse = {name: held_out_mse(usable, features, outcome) for name, features in models.items()}
        baseline = mse['router_distance']
        panels[label] = dict(prompts=n, observations=len(usable), status='exploratory', outcome=outcome,
            models={name: dict(features=models[name], prompt_macro_mse=estimate(list(values.values())),
                mse_difference_vs_router_distance=estimate([values[s] - baseline[s] for s in baseline]))
                for name, values in mse.items()})
    return panels


def summarize(records):
    valid = [r for r in records if r['status'] == 'matched']
    observed = [r for r in records if r['status'] in ('matched', 'observed_unmatched')]
    replayed = [r for r in records if r['status'] != 'no_match']
    return dict(attempted_states=len(records), matched_states=len(valid), observed_states=len(observed),
        prompts_attempted=len({r['sample'] for r in records}),
        prompts_matched=len({r['sample'] for r in valid}),
        prompts_observed=len({r['sample'] for r in observed}),
        status_counts=dict(Counter(r['status'] for r in records)),
        no_match_reasons=dict(Counter(r['no_match_reason'] for r in records if r['status'] == 'no_match')),
        effects_vs_original=paired_effects(valid, 'effects'),
        effects_vs_same_route_control=paired_effects(valid, 'relative_to_control'),
        reconstruction_effects=paired_effects(valid, 'control_effects'),
        descriptive_all_observations=paired_effects(observed, 'relative_to_control'),
        max_sham_logit_error=max((r['sham_logit_max_abs'] for r in replayed), default=None),
        max_control_logit_error=max((r[s]['control_logit_max_abs'] for r in replayed for s in ('strong', 'weak')), default=None),
        predictive_checks=predictive_checks(observed),
        notes=['Frontier is frozen original acceptance, an offline oracle, not an online method',
               'Paired effects use strict matches only; descriptive pooled effects are confounded and not matched comparisons',
               'Predictive checks use strict plus unmatched stratified observations, conditional on this sampling protocol',
               'Prediction flips/acceptance loss are not final ACC or causal identification of attention',
               'Same-route controls may have numeric drift; inspect both control and intervention effects',
               'Primary continuous outcome: absolute original-token logit-margin change, relative to same-route control',
               'Predictive analysis uses leave-one-prompt-out ridge (alpha=1), training-only scaling and prompt weights',
               'Negative held-out MSE difference means better prediction, not faster decoding or better ACC',
               'Bootstrap CIs are descriptive prompt resampling; fitted-fold uncertainty and multiple testing are not corrected',
               'Shuffled feature permutes source scores within each state; it is a single negative control, not a permutation significance test',
               'Old graph to current frontier still uses oracle receivers; only predicted-frontier feature is fully historical'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    records, sources = load_records(args.probe_dir)
    result = summarize(records); result['sources'] = sources
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# 单位置专家替换与当前接受集合', '',
             f"尝试 {result['attempted_states']} 个状态，有效 {result['matched_states']} 个，"
             f"覆盖 {result['prompts_matched']}/{result['prompts_attempted']} 个 prompt。", '',
             f"包含非配对观测共 {result['observed_states']} 个有效状态，覆盖 {result['prompts_observed']} 个 prompt。", '',
             '以下以原始接受集合为固定接收位置。差值基于相同重算路径的原始 Top-8 对照，按 prompt 等权。', '',
             '| 指标 | 强关联 | 弱关联 | 强减弱 | 95% CI |', '|---|---:|---:|---:|---|']
    for key, values in result['effects_vs_same_route_control'].items():
        d = values['strong_minus_weak']
        lines.append(f"| {key} | {values['strong']['mean']} | {values['weak']['mean']} | {d['mean']} | {d['ci95']} |")
    lines += ['', '## 全部有效观测（描述性，含非配对样本）', '',
              '本表的强弱差异未保证控制量匹配，不可作为依赖的独立效应。', '',
              '| 指标 | 强关联 | 弱关联 |', '|---|---:|---:|']
    for key, values in result['descriptive_all_observations'].items():
        lines.append(f"| {key} | {values['strong']['mean']} | {values['weak']['mean']} |")
    lines += ['', '## 预测比较', '',
              '比较目标为接受位置原始预测的 logit margin 绝对变化；更小的留出 MSE 表示预测更好。',
              '每次留出完整 prompt，历史比较仅使用共同非缺失记录。少于 8 个有效 prompt 不拟合。', '']
    for panel, result_panel in result['predictive_checks'].items():
        lines.append(f"### {panel}: {result_panel['status']} ({result_panel['prompts']} prompts)")
        lines.append('')
        for name, model in result_panel.get('models', {}).items():
            lines.append(f"- {name}: MSE={model['prompt_macro_mse']['mean']}; "
                         f"相对 router+distance 差值={model['mse_difference_vs_router_distance']}")
        lines.append('')
    lines += ['## 限制与检查', '', *['- ' + note for note in result['notes']], '',
              f"状态计数：{result['status_counts']}", f"未匹配原因：{result['no_match_reasons']}",
              f"最大 sham logit 误差：{result['max_sham_logit_error']}",
              f"最大原始 Top-8 重算 logit 误差：{result['max_control_logit_error']}"]
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('attempted_states', 'matched_states', 'observed_states', 'prompts_observed', 'prompts_matched', 'status_counts')}))


if __name__ == '__main__':
    main()
