#!/usr/bin/env python3
"""Configure a Vanilla pilot at runtime; leave existing configs untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def validate_gpu_count(requested, available, visible):
    """Reject unavailable logical CUDA IDs before allocating model weights."""
    if requested > available:
        raise ValueError(
            f'Requested {requested} GPUs, but this process sees {available}; '
            f'CUDA_VISIBLE_DEVICES={visible!r}. For two physical GPUs 0 and 1, '
            'set CUDA_VISIBLE_DEVICES=0,1 and MODEL_GPUS=2 together. '
            'max_memory uses logical IDs 0..N-1, not physical GPU IDs.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['humaneval', 'mbpp', 'gsm8k'], default='humaneval')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--sample-start', type=int, default=0)
    parser.add_argument('--layers', default='auto', help='auto = first/middle/last; all or comma-separated physical layer IDs')
    parser.add_argument('--max-blocks', type=int, default=8)
    parser.add_argument('--gen-length', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--memory', default='74GiB')
    parser.add_argument('--hook-off', action='store_true', help='Same settings without observation, for prediction parity checks')
    parser.add_argument('--dry-run', action='store_true', help='Write manifest and resolved config without loading weights')
    parser.add_argument('--probe', action='store_true', help='Extra paired diagnostic forwards; original decoding remains baseline')
    parser.add_argument('--probe-layer', type=int, default=24)
    parser.add_argument('--probe-pool-size', type=int, default=12)
    parser.add_argument('--probe-stride', type=int, default=2)
    parser.add_argument('--probe-max-steps', type=int, default=48)
    parser.add_argument('--probe-mass-tolerance', type=float, default=.03)
    parser.add_argument('--probe-confidence-tolerance', type=float, default=.05)
    args = parser.parse_args()
    if min(args.samples, args.max_blocks, args.gen_length, args.gpus) < 1 or args.sample_start < 0:
        parser.error('Counts must be positive; sample-start must be nonnegative')
    if args.probe and args.hook_off:
        parser.error('--probe and --hook-off cannot be combined')
    if min(args.probe_stride, args.probe_max_steps) < 1 or args.probe_layer < 0 or args.probe_pool_size < 8:
        parser.error('Invalid probe layer/pool/stride/limit')
    if min(args.probe_mass_tolerance, args.probe_confidence_tolerance) < 0:
        parser.error('Matching tolerances must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    args.output_dir = args.output_dir.resolve()
    import numpy as np
    import torch
    if not args.dry_run:
        try:
            validate_gpu_count(args.gpus, torch.cuda.device_count(),
                               os.environ.get('CUDA_VISIBLE_DEVICES'))
        except ValueError as error:
            parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from mmengine.config import Config
    from opencompass.cli import main as cli
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config_path = root / f'configs/sdar/vanilla/accuracy/eval_sdar_hf_{args.dataset}_vanilla.py'
    cfg = Config.fromfile(str(config_path), format_python_code=False)
    for dataset in cfg.datasets:
        dataset['reader_cfg']['test_range'] = f'[{args.sample_start}:{args.sample_start + args.samples}]'
        dataset['infer_cfg']['inferencer']['batch_size'] = 1
    model = cfg.models[0]
    model['path'] = args.model_path
    model['abbr'] = f'SDAR-vanilla-dependency-{args.dataset}-seed{args.seed}'
    model['run_cfg']['num_gpus'] = args.gpus
    model['model_kwargs'].update(device_map='auto', max_memory={i: args.memory for i in range(args.gpus)})
    model.update(collect_metrics=False, collect_dependency_trace=not args.hook_off,
                 dependency_trace_output_dir=str(args.output_dir / 'traces'),
                 dependency_layers=args.layers, dependency_max_blocks=args.max_blocks)
    if args.probe:
        model.update(dependency_probe_output_dir=str(args.output_dir / 'probes'),
                     dependency_probe_options=dict(layer=args.probe_layer,
                         pool_size=args.probe_pool_size, stride=args.probe_stride,
                         max_steps=args.probe_max_steps,
                         mass_tolerance=args.probe_mass_tolerance,
                         confidence_tolerance=args.probe_confidence_tolerance))
    model['generation_kwargs']['gen_length'] = args.gen_length
    cfg.work_dir = str(args.output_dir / 'opencompass')
    cfg.infer['runner']['retry'] = 0
    cfg.eval['partitioner']['n'] = 1
    manifest = vars(args).copy()
    manifest['output_dir'] = str(args.output_dir)
    manifest.update(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    baseline_config=str(config_path),
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    limitations=['observational only; not a speed benchmark',
                                 'first max-blocks blocks observed; generation itself is NOT truncated',
                                 'process seed + --debug; check On/Off predictions before interpreting differences'])
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cfg.dump(str(args.output_dir / 'resolved_config.py'))
    if args.dry_run:
        print(args.output_dir)
        return
    # --debug executes inference in this seeded process, not a fresh worker.
    cli.get_config_from_arg = lambda _args: cfg
    sys.argv = ['run.py', str(config_path), '--debug', '--mode', 'infer']
    cli.main()


if __name__ == '__main__':
    main()
