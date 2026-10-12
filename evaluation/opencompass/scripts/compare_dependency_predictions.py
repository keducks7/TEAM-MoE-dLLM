"""Compare generated strings only, not timing/metadata or evaluator scores."""
import argparse
import json
from pathlib import Path


def prediction_strings(value, path=()):
    result = {}
    if isinstance(value, dict):
        if 'prediction' in value:
            result[path] = value['prediction']
        else:
            for key, item in value.items():
                result.update(prediction_strings(item, path + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(prediction_strings(item, path + (str(index),)))
    return result


def load_predictions(root):
    result = {}
    for file in sorted(root.rglob('*.json')):
        if 'predictions' not in file.parts:
            continue
        index = file.parts.index('predictions')
        relative = '/'.join(file.parts[index + 1:])
        predictions = prediction_strings(json.loads(file.read_text(encoding='utf-8')))
        for sample, value in predictions.items():
            key = (relative, sample)
            if key in result:
                raise ValueError(f'Multiple experiments in directory: {root}')
            result[key] = value
    if not result:
        raise ValueError(f'No generated prediction strings found under {root}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    args = parser.parse_args()
    on, off = load_predictions(args.on), load_predictions(args.off)
    missing = set(on) ^ set(off)
    changed = [key for key in set(on) & set(off) if on[key] != off[key]]
    print(f'On: {len(on)}, Off: {len(off)}, missing: {len(missing)}, changed: {len(changed)}')
    if missing or changed:
        raise SystemExit('Prediction parity FAILED: investigate before interpreting pilot observations')
    print('Prediction parity PASSED')


if __name__ == '__main__':
    main()
