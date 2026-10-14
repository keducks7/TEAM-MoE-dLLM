#!/usr/bin/env python3
"""Seeded TEAM ACC/metrics run. No observer/probe/metric forward hooks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from mmengine.config import Config
    from opencompass.cli import main as cli
    seed = int(os.environ.get('SEED', '42'))
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    cfg = Config.fromfile(args.config, format_python_code=False)
    model = cfg.models[0]
    requested = model['run_cfg']['num_gpus']
    if requested > torch.cuda.device_count():
        parser.error(f'Requested {requested} GPUs, but CUDA sees {torch.cuda.device_count()}')
    source = Path(model['path']) / 'modeling_sdar_moe.py'
    if not source.is_file():
        parser.error(f'Missing checkpoint model source: {source}')
    text = source.read_text()
    if 'team_collect_expert_stats' not in text or '_select_team_expert_pool' not in text:
        parser.error('Copy updated repository modeling_sdar_moe.py into the checkpoint directory first')
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(seed=seed, sample_seed='sha256(seed,prompt_token_ids)',
        config=args.config, strategy=model['generation_kwargs']['team_pool_strategy'],
        budget=model['generation_kwargs']['team_pool_budget'],
        coverage_target=model['generation_kwargs']['team_pool_coverage_target'],
        collect_metrics=model['collect_metrics'], collect_latency=model['collect_latency'],
        metrics_backend=model['metrics_backend'], accepted_token_mode=model['accepted_token_mode'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpus=requested,
        torch_version=torch.__version__, model_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_start=os.environ.get('SAMPLE_START', '32'), samples=os.environ.get('SAMPLES', 'all'),
        records_dir=model['metrics_output_dir'], work_dir=cfg.work_dir,
        timing_scope='generation only: prefill + denoising + cache commit; model loading/evaluation excluded')
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cfg.dump(str(args.manifest.parent / 'resolved_config.py'))
    cli.get_config_from_arg = lambda _args: cfg
    sys.argv = ['run.py', args.config, '--debug']
    cli.main()


if __name__ == '__main__':
    main()
