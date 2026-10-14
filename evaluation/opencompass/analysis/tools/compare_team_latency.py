#!/usr/bin/env python3
"""Paired generation latency; reject mismatched prompts or instrumentation."""
import argparse
import json
import math
from pathlib import Path


def load_records(path):
    records = {}
    for file in sorted(path.rglob('*_metrics_*.jsonl')):
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get('prompt_sha256')
            elapsed = row.get('generation_latency_s')
            if not prompt or elapsed is None or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Only new per-prompt generation timing records are supported')
            if prompt in records:
                raise ValueError('Duplicate prompts; compare a fresh, single run')
            records[prompt] = row
    if not records:
        raise ValueError(f'No timing records under {path}')
    return records


def compare(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError('Prompt sets differ; compare identical sample ranges')
    fields = ('model_source_sha256', 'model_path', 'sample_seed', 'seed_base', 'generation_settings',
              'cuda_visible_devices', 'torch_version', 'metrics_backend', 'metrics_collected',
              'latency_scope', 'latency_includes_metric_collection', 'accepted_token_mode')
    for key in baseline:
        for field in fields:
            if baseline[key].get(field) != candidate[key].get(field):
                raise ValueError(f'Mismatched {field} for prompt {key}')
    base_time = sum(r['generation_latency_s'] for r in baseline.values())
    candidate_time = sum(r['generation_latency_s'] for r in candidate.values())
    return dict(prompts=len(baseline), baseline_generation_latency_s=base_time,
                candidate_generation_latency_s=candidate_time, speedup=base_time / candidate_time,
                scope='paired generation wall-clock only; use the same idle GPUs and environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(load_records(args.baseline_dir), load_records(args.candidate_dir))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
