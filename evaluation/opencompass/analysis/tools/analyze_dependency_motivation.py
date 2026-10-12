#!/usr/bin/env python3
"""Observational Q1/Q2 analysis. No causal or pruning-safety claims.

Stratify by physical layer, mask fraction, distance, confidence, and proximity
to actually accepted positions. Summary uncertainty is bootstrapped by prompt,
not by token pair. No history crosses a block boundary or a skipped step.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def ranks(x):
    order = np.argsort(x, kind='stable')
    sorted_x = np.asarray(x)[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    result = np.empty(len(x), dtype=float)
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start + end - 1) / 2
    return result


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 6:
        return None
    a, b = ranks(x[valid]), ranks(y[valid])
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else None


def strong_edge_recall(target, proxy, fraction=0.2):
    """Threshold-defined true strong edges; include ties, avoiding fake rankings."""
    if len(target) < 6 or np.ptp(target) == 0 or np.ptp(proxy) == 0:
        return None
    strong = target >= np.quantile(target, 1 - fraction)
    predicted = proxy >= np.quantile(proxy, 1 - fraction)
    return float((strong & predicted).sum() / strong.sum())


def strong_edge_precision(target, proxy, fraction=0.2):
    if len(target) < 6 or np.ptp(target) == 0 or np.ptp(proxy) == 0:
        return None
    strong = target >= np.quantile(target, 1 - fraction)
    predicted = proxy >= np.quantile(proxy, 1 - fraction)
    return float((strong & predicted).sum() / predicted.sum())


def cosine_rows(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    normalized = x / np.maximum(norm, 1e-12)
    return normalized @ normalized.T


def expert_pair_features(ids, probs, i, j):
    k = ids.shape[-1]
    shared = (ids[i, :, None] == ids[j, None, :]).sum(axis=(1, 2))
    overlap = shared / k
    jaccard = shared / (2 * k - shared)
    membership = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(membership, ids, True, axis=1)
    common = membership[i] & membership[j]
    shared_mass = ((probs[i] * common).sum(-1) + (probs[j] * common).sum(-1)) / 2
    # Relative mass within each token's ORIGINAL top-k, not full E probabilities.
    private_i = (probs[i] * (membership[i] & ~membership[j])).sum(-1)
    private_j = (probs[j] * (membership[j] & ~membership[i])).sum(-1)
    topmass_i = (probs[i] * membership[i]).sum(-1)
    topmass_j = (probs[j] * membership[j]).sum(-1)
    private_fraction = (private_i / np.maximum(topmass_i, 1e-12)
                        + private_j / np.maximum(topmass_j, 1e-12)) / 2
    cosine = cosine_rows(probs)[i, j]
    # Frequency in current masked rows only; precomputed by caller slicing probs/ids.
    frequency = membership.mean(axis=0)
    selected_frequency = (frequency[ids[i]].mean(-1) + frequency[ids[j]].mean(-1)) / 2
    return dict(overlap_at_k=overlap, jaccard=jaccard,
                same_top1=(ids[i, 0] == ids[j, 0]).astype(float),
                router_cosine=cosine, shared_raw_mass=shared_mass,
                private_topk_mass_fraction=private_fraction,
                union_size=2 * k - shared,
                selected_expert_mask_frequency=selected_frequency)


def stage(ratio):
    return ('0-.25', '.25-.5', '.5-.75', '.75-1')[min(3, int(ratio * 4))]


def distance_bin(distance):
    return '1' if distance == 1 else '2-3' if distance <= 3 else '4-7' if distance <= 7 else '8-15' if distance <= 15 else '16+'


def confidence_bin(confidence):
    return '<.5' if confidence < .5 else '.5-.9' if confidence < .9 else '>=.9'


def load_trace(path):
    with np.load(path, allow_pickle=False) as file:
        data = {k: file[k] for k in file.files}
    metadata = json.loads(str(data['metadata'].item()))
    if metadata.get('schema') != 'sdar_dependency_v1' or metadata.get('method') != 'vanilla':
        raise ValueError(f'Unsupported schema/method: {path}')
    f, t = data['mask'].shape
    l = len(data['layer_ids'])
    for key in ('attention', 'hidden_cosine'):
        if data[key].shape != (f, l, t, t) or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key}: {path}')
    if data['router_probs'].shape[:3] != (f, l, t):
        raise ValueError(f'Invalid router alignment: {path}')
    for key in ('positions', 'confidence', 'accepted', 'predictions', 'token_ids'):
        if data[key].shape != (f, t):
            raise ValueError(f'Invalid {key} alignment: {path}')
    if np.any(data['accepted'] & ~data['mask']):
        raise ValueError(f'Accepted nonmasked positions: {path}')
    return metadata, data


def analyze_trace(path, edge_fraction=0.2):
    metadata, d = load_trace(path)
    sample = metadata['prompt_sha256']
    history_rows, relation_rows = [], []
    for f in range(len(d['mask'])):
        mask = d['mask'][f]
        indices = np.flatnonzero(mask)
        if len(indices) < 2:
            continue
        local_i, local_j = np.triu_indices(len(indices), 1)
        i, j = indices[local_i], indices[local_j]
        distance = np.abs(d['positions'][f, i] - d['positions'][f, j])
        confidence = (d['confidence'][f, i] + d['confidence'][f, j]) / 2
        # Retrospective role, NOT the original TEAM Hot/Cold definition.
        role = d['accepted'][f, i].astype(int) + d['accepted'][f, j].astype(int)
        strata = [(distance_bin(int(a)), confidence_bin(float(b)),
                   ('neither_accepted', 'one_accepted', 'both_accepted')[c])
                  for a, b, c in zip(distance, confidence, role)]
        adjacent = (f > 0 and d['block_id'][f] == d['block_id'][f - 1]
                    and d['step'][f] == d['step'][f - 1] + 1
                    and np.array_equal(d['positions'][f], d['positions'][f - 1]))
        for layer_idx, layer_id in enumerate(d['layer_ids']):
            attention = d['attention'][f, layer_idx]
            dependency = (attention[i, j] + attention[j, i]) / 2
            features = expert_pair_features(d['expert_ids'][f, layer_idx, indices],
                                           d['router_probs'][f, layer_idx, indices],
                                           local_i, local_j)
            q20, q80 = np.quantile(dependency, [.2, .8])
            groups = np.full(len(i), 'middle', dtype=object)
            if q20 == q80:
                groups[:] = 'tied'
            else:
                groups[dependency <= q20] = 'weak'
                groups[dependency >= q80] = 'strong'
            base = dict(sample=sample, trace=path.name, block=int(d['block_id'][f]),
                        step=int(d['step'][f]), layer=int(layer_id),
                        mask_stage=stage(float(mask.mean())))
            for stratum in sorted(set(strata)):
                selected = np.asarray([s == stratum for s in strata])
                for group in ('weak', 'middle', 'strong', 'tied'):
                    rows = selected & (groups == group)
                    if not rows.any():
                        continue
                    relation_rows.append(dict(base, distance=stratum[0],
                        confidence_bin=stratum[1], role=stratum[2], dependency_group=group,
                        pairs=int(rows.sum()), dependency_mean=float(dependency[rows].mean()),
                        **{key: float(value[rows].mean()) for key, value in features.items()}))
            if not adjacent:
                continue
            eligible = d['mask'][f - 1, i] & d['mask'][f - 1, j]
            old_a = d['attention'][f - 1, layer_idx]
            old_p = d['router_probs'][f - 1, layer_idx]
            old_ids = d['expert_ids'][f - 1, layer_idx]
            old_overlap = (old_ids[i, :, None] == old_ids[j, None, :]).sum((1, 2)) / old_ids.shape[-1]
            proxies = {
                'previous_attention': (old_a[i, j] + old_a[j, i]) / 2,
                'previous_router_cosine': cosine_rows(old_p)[i, j],
                'previous_topk_overlap': old_overlap,
                'previous_hidden_cosine': d['hidden_cosine'][f - 1, layer_idx, i, j],
                'previous_confidence_similarity': -np.abs(d['confidence'][f - 1, i] - d['confidence'][f - 1, j]),
                'position_proximity': -distance.astype(float),
            }
            two_steps = (f >= 2 and d['block_id'][f - 2] == d['block_id'][f]
                         and d['step'][f - 2] == d['step'][f] - 2
                         and np.array_equal(d['positions'][f - 2], d['positions'][f]))
            if two_steps:
                delta = d['confidence'][f - 1] - d['confidence'][f - 2]
                proxy = -np.abs(delta[i] - delta[j])
                proxy[~(d['mask'][f - 2, i] & d['mask'][f - 2, j])] = np.nan
                proxies['previous_confidence_change_similarity'] = proxy
            # Overall plus distance/confidence strata (minimum six eligible pairs).
            selections = [('all', 'all', eligible)]
            for db, cb, _ in sorted(set(strata)):
                selections.append((db, cb, eligible & np.asarray([
                    s[0] == db and s[1] == cb for s in strata])))
            seen = set()
            for db, cb, selected in selections:
                if (db, cb) in seen:
                    continue
                seen.add((db, cb))
                for name, proxy in proxies.items():
                    valid = selected & np.isfinite(proxy)
                    if valid.sum() < 6:
                        continue
                    history_rows.append(dict(base, distance=db, confidence_bin=cb,
                        proxy=name, pairs=int(valid.sum()),
                        newly_accepted_previous=int(d['accepted'][f - 1].sum()),
                        spearman=spearman(dependency[valid], proxy[valid]),
                        strong_edge_recall=strong_edge_recall(
                            dependency[valid], proxy[valid], edge_fraction),
                        strong_edge_precision=strong_edge_precision(
                            dependency[valid], proxy[valid], edge_fraction)))
    return metadata, history_rows, relation_rows


def summarize(rows, keys, metrics):
    # Macro average forwards within each prompt/stratum, then prompts equally.
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        group = tuple(row[k] for k in keys)
        for metric in metrics:
            if row[metric] is not None:
                grouped[group][row['sample']][metric].append(row[metric])
    result = []
    rng = np.random.default_rng(20261012)
    for group, samples in sorted(grouped.items()):
        row = dict(zip(keys, group))
        row['prompts'] = len(samples)
        for metric in metrics:
            values = np.asarray([np.mean(s[metric]) for s in samples.values() if s[metric]])
            row[metric] = float(values.mean()) if len(values) else None
            if len(values) > 1:
                draws = rng.choice(values, (1000, len(values)), replace=True).mean(axis=1)
                lo, hi = np.quantile(draws, [.025, .975])
                row[metric + '_ci_low'], row[metric + '_ci_high'] = float(lo), float(hi)
            else:
                row[metric + '_ci_low'] = row[metric + '_ci_high'] = None
        result.append(row)
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matched_contrasts(rows, metrics):
    """Strong-minus-weak within the SAME forward/layer/distance/confidence/role."""
    keys = ['sample', 'trace', 'block', 'step', 'layer', 'mask_stage',
            'distance', 'confidence_bin', 'role']
    grouped = defaultdict(dict)
    for row in rows:
        grouped[tuple(row[k] for k in keys)][row['dependency_group']] = row
    result = []
    for group, buckets in grouped.items():
        if 'strong' not in buckets or 'weak' not in buckets:
            continue
        strong, weak = buckets['strong'], buckets['weak']
        result.append(dict(zip(keys, group),
                           strong_pairs=strong['pairs'], weak_pairs=weak['pairs'],
                           **{m: strong[m] - weak[m] for m in metrics}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--edge-fraction', type=float, default=.2)
    args = parser.parse_args()
    if not 0 < args.edge_fraction < .5:
        parser.error('edge-fraction must be between 0 and .5')
    files = sorted(args.trace_dir.glob('dependency_*.npz'))
    if not files:
        parser.error('No dependency NPZ traces found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, relation, metadata = [], [], []
    seen = set()
    for path in files:
        info, h, r = analyze_trace(path, args.edge_fraction)
        if info['prompt_sha256'] in seen:
            raise ValueError('Duplicate prompt traces: use a fresh directory per run/dataset')
        seen.add(info['prompt_sha256'])
        metadata.append(info)
        history.extend(h)
        relation.extend(r)
    signatures = {(m['model_source_sha256'], m.get('model_path'), m['threshold'],
                   tuple(m['layer_ids']), m.get('seed'),
                   json.dumps(m.get('generation_settings', {}), sort_keys=True)) for m in metadata}
    if len(signatures) != 1:
        raise ValueError('Mixed model source/settings in one analysis directory')
    h_summary = summarize(history, ['layer', 'mask_stage', 'distance', 'confidence_bin', 'proxy'],
                          ['spearman', 'strong_edge_recall', 'strong_edge_precision'])
    r_metrics = ['overlap_at_k', 'jaccard', 'same_top1', 'router_cosine',
                 'shared_raw_mass', 'private_topk_mass_fraction', 'union_size',
                 'selected_expert_mask_frequency']
    r_summary = summarize(relation, ['layer', 'mask_stage', 'distance', 'confidence_bin', 'role', 'dependency_group'], r_metrics)
    contrasts = matched_contrasts(relation, r_metrics)
    c_summary = summarize(contrasts, ['layer', 'mask_stage', 'distance', 'confidence_bin', 'role'], r_metrics)
    for name, rows in [('history_forward.csv', history), ('expert_relation_forward.csv', relation),
                       ('history_summary.csv', h_summary), ('expert_relation_summary.csv', r_summary),
                       ('matched_contrast_forward.csv', contrasts), ('matched_contrast_summary.csv', c_summary)]:
        write_csv(args.output_dir / name, rows)
    report = dict(prompts=len(seen), traces=len(files), history_rows=len(history),
                  relation_rows=len(relation), edge_fraction=args.edge_fraction,
                  notes=['attention is an FP32 proxy, not ground-truth dependence',
                         'accepted-role groups are retrospective, not TEAM Hot/Cold',
                         'CI is prompt-cluster bootstrap; small pilots are exploratory',
                         'correlation/stratification does not establish causal sharing safety',
                         'matched contrasts require both strong and weak groups in the SAME stratum; no extrapolation',
                         'ties can retain more than edge_fraction edges; constant proxies are undefined',
                         'no confidence change feature is computed across block/step gaps'],
                  metadata=metadata)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Vanilla dependency motivation pilot', '',
             f'Prompts: {len(seen)}; traces: {len(files)}. Attention is a reconstructed proxy.', '',
             '## Q1: overall history proxies (stratified details are in history_summary.csv)', '',
             '| Layer | Mask fraction | Proxy | Prompts | Spearman | Strong-edge recall | Precision |',
             '|---|---|---|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for row in h_summary:
        if row['distance'] == 'all' and row['confidence_bin'] == 'all':
            lines.append(f"| {row['layer']} | {row['mask_stage']} | {row['proxy']} | {row['prompts']} | "
                         f"{fmt(row['spearman'])} | {fmt(row['strong_edge_recall'])} | {fmt(row['strong_edge_precision'])} |")
    lines += ['', '## Q2: expert relationships', '',
              'expert_relation_summary.csv: strong/weak/tied group features.',
              'matched_contrast_summary.csv: strong-minus-weak differences matched within forward, layer, distance, confidence bin and accepted-role.', '',
              'Positive overlap contrast suggests more shared identities; positive private-mass contrast suggests more distinct demands. Neither establishes causal complementarity or safe pruning.', '',
              'Prompt-cluster bootstrap intervals are in CSVs. Sparse strata and small pilots are exploratory. '
              'Undefined correlations are NA, not zero. Do not use observation runtime to claim speedup.']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('prompts', 'traces', 'history_rows', 'relation_rows')}))


if __name__ == '__main__':
    main()
