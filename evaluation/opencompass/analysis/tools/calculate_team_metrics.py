#!/usr/bin/env python3
"""Calculate APF, TPF, and APT from legacy TEAM Mark or unified JSONL.

The mark evaluator writes one expert-distribution JSON file and one
decoded-order JSON file per generated sequence.  APF is the mean number of
distinct, actually activated experts (expert id != -1) per MoE layer and
forward pass.  TPF is the mean number of accepted tokens per forward pass.
APT is APF / TPF, matching the definition used in the TEAM paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class MetricTotals:
    activated_experts: int = 0
    layer_forwards: int = 0
    expert_forwards: int = 0
    accepted_tokens: int = 0
    decoded_forwards: int = 0

    @property
    def apf(self) -> float:
        if self.layer_forwards == 0:
            raise ValueError("No MoE layer/forward records were found")
        return self.activated_experts / self.layer_forwards

    @property
    def tpf(self) -> float:
        if self.decoded_forwards == 0:
            raise ValueError("No decoded forward records were found")
        return self.accepted_tokens / self.decoded_forwards

    @property
    def apt(self) -> float:
        if self.tpf == 0:
            raise ValueError("APT is undefined because TPF is zero")
        return self.apf / self.tpf


def numeric_key(value: str) -> tuple[int, int | str]:
    """Sort JSON keys numerically when possible."""
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def leaf_values(value: Any) -> Iterator[Any]:
    """Yield scalar values from arbitrarily nested JSON lists."""
    if isinstance(value, list):
        for item in value:
            yield from leaf_values(item)
    else:
        yield value


def forward_keys(payload: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for block, steps in payload.items():
        if not isinstance(steps, dict):
            raise ValueError(f"Block {block!r} must contain a JSON object")
        keys.update((str(block), str(step)) for step in steps)
    return keys


def count_experts(payload: dict[str, Any], totals: MetricTotals) -> None:
    for block in sorted(payload, key=numeric_key):
        steps = payload[block]
        if not isinstance(steps, dict):
            raise ValueError(f"Block {block!r} must contain a JSON object")
        for step in sorted(steps, key=numeric_key):
            layers = steps[step]
            if not isinstance(layers, dict) or not layers:
                raise ValueError(
                    f"Expert record {block}/{step} has no layer data")
            totals.expert_forwards += 1
            for layer, selected_experts in layers.items():
                expert_ids = set()
                for expert_id in leaf_values(selected_experts):
                    if isinstance(expert_id, bool) or not isinstance(expert_id, int):
                        raise ValueError(
                            f"Expert id at {block}/{step}/{layer} is not an integer")
                    if expert_id >= 0:
                        expert_ids.add(expert_id)
                totals.activated_experts += len(expert_ids)
                totals.layer_forwards += 1


def count_accepted_tokens(payload: dict[str, Any], totals: MetricTotals) -> None:
    for block in sorted(payload, key=numeric_key):
        steps = payload[block]
        if not isinstance(steps, dict):
            raise ValueError(f"Block {block!r} must contain a JSON object")
        for step in sorted(steps, key=numeric_key):
            record = steps[step]
            if not isinstance(record, dict) or "transfer_index" not in record:
                raise ValueError(
                    f"Decoded record {block}/{step} has no transfer_index")
            transfer_index = list(leaf_values(record["transfer_index"]))
            if any(not isinstance(item, bool) for item in transfer_index):
                raise ValueError(
                    f"transfer_index at {block}/{step} contains a non-boolean value")
            totals.accepted_tokens += sum(transfer_index)
            totals.decoded_forwards += 1


def find_json_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
    files = sorted(path for path in directory.rglob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No files matching {pattern!r} under {directory}")
    return files


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")
    return payload


def relative_file_map(directory: Path, files: Iterable[Path]) -> dict[Path, Path]:
    return {path.relative_to(directory): path for path in files}


def calculate(
    expert_dir: Path,
    decoded_dir: Path,
    pattern: str,
) -> tuple[MetricTotals, list[str], int, int]:
    expert_files = find_json_files(expert_dir, pattern)
    decoded_files = find_json_files(decoded_dir, pattern)
    expert_map = relative_file_map(expert_dir, expert_files)
    decoded_map = relative_file_map(decoded_dir, decoded_files)

    warnings: list[str] = []
    only_expert = sorted(set(expert_map) - set(decoded_map))
    only_decoded = sorted(set(decoded_map) - set(expert_map))
    if only_expert:
        warnings.append(
            f"{len(only_expert)} sequence file(s) exist only in the expert directory")
    if only_decoded:
        warnings.append(
            f"{len(only_decoded)} sequence file(s) exist only in the decoded directory")

    totals = MetricTotals()
    expert_payloads: dict[Path, dict[str, Any]] = {}
    decoded_payloads: dict[Path, dict[str, Any]] = {}

    for relative_path, path in expert_map.items():
        payload = load_json(path)
        expert_payloads[relative_path] = payload
        count_experts(payload, totals)
    for relative_path, path in decoded_map.items():
        payload = load_json(path)
        decoded_payloads[relative_path] = payload
        count_accepted_tokens(payload, totals)

    for relative_path in sorted(set(expert_map) & set(decoded_map)):
        expert_keys = forward_keys(expert_payloads[relative_path])
        decoded_keys = forward_keys(decoded_payloads[relative_path])
        if expert_keys != decoded_keys:
            warnings.append(
                f"Forward records differ for {relative_path}: "
                f"expert={len(expert_keys)}, decoded={len(decoded_keys)}")

    if totals.expert_forwards != totals.decoded_forwards:
        warnings.append(
            "The total expert and decoded forward counts differ: "
            f"{totals.expert_forwards} vs {totals.decoded_forwards}")

    return totals, warnings, len(expert_files), len(decoded_files)


def calculate_compact_metrics(
    metrics_dir: Path,
    pattern: str,
) -> tuple[MetricTotals, list[str], int, int]:
    files = find_json_files(metrics_dir, pattern)
    totals = MetricTotals()
    records = 0
    warnings: list[str] = []
    for path in files:
        try:
            with path.open('r', encoding='utf-8') as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get('schema') not in {
                            'sdar_vanilla_metrics_v1', 'sdar_metrics_v1'}:
                        raise ValueError(
                            f'Unknown metric schema in {path}:{line_number}')
                    totals.activated_experts += int(record['activated_experts'])
                    totals.layer_forwards += int(record['layer_forwards'])
                    totals.accepted_tokens += int(record['accepted_tokens'])
                    totals.decoded_forwards += int(record['denoising_forwards'])
                    totals.expert_forwards += int(record['denoising_forwards'])
                    records += 1
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f'Cannot read {path}: {error}') from error
    if records == 0:
        raise ValueError(f'No metric records were found under {metrics_dir}')
    if totals.decoded_forwards == 0:
        raise ValueError('Metric records contain no denoising forwards')
    if totals.layer_forwards % totals.decoded_forwards != 0:
        warnings.append(
            'Layer-forward count is not divisible by denoising-forward count; '
            'check whether all samples used the same MoE layer count.')
    return totals, warnings, len(files), records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate APF/TPF/APT from legacy TEAM *_mark JSON files or "
            "compact Vanilla/TEAM metric JSONL files."))
    parser.add_argument(
        "--expert-dir",
        type=Path,
        help="Directory containing expert_distribution sequence JSON files.")
    parser.add_argument(
        "--decoded-dir",
        type=Path,
        help="Directory containing decoded_order sequence JSON files.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        help="Directory containing compact Vanilla or TEAM metric JSONL files.")
    parser.add_argument(
        "--pattern",
        default="sequence_*.json",
        help="Recursive input filename glob (default: sequence_*.json).")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.metrics_dir is not None:
            if args.expert_dir is not None or args.decoded_dir is not None:
                raise ValueError(
                    '--metrics-dir cannot be combined with --expert-dir or '
                    '--decoded-dir')
            compact_pattern = (args.pattern if args.pattern != 'sequence_*.json'
                               else '*_metrics_*.jsonl')
            totals, warnings, metric_files, metric_records = \
                calculate_compact_metrics(
                    args.metrics_dir.resolve(), compact_pattern)
            expert_files = metric_files
            decoded_files = metric_files
        else:
            if args.expert_dir is None or args.decoded_dir is None:
                raise ValueError(
                    'Either --metrics-dir or both --expert-dir and '
                    '--decoded-dir are required')
            totals, warnings, expert_files, decoded_files = calculate(
                args.expert_dir.resolve(), args.decoded_dir.resolve(),
                args.pattern)
            metric_files = None
            metric_records = None
        result = {
            "APF": totals.apf,
            "TPF": totals.tpf,
            "APT": totals.apt,
            "expert_files": expert_files,
            "decoded_files": decoded_files,
            "expert_forwards": totals.expert_forwards,
            "decoded_forwards": totals.decoded_forwards,
            "layer_forwards": totals.layer_forwards,
            "accepted_tokens": totals.accepted_tokens,
        }
        if metric_files is not None:
            result['metric_files'] = metric_files
            result['metric_records'] = metric_records
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"APF: {result['APF']:.4f}")
        print(f"TPF: {result['TPF']:.4f}")
        print(f"APT: {result['APT']:.4f}")
        print()
        if metric_files is None:
            print(f"Expert JSON files : {expert_files}")
            print(f"Decoded JSON files: {decoded_files}")
        else:
            print(f"Metric JSONL files: {metric_files}")
            print(f"Metric records    : {metric_records}")
        print(f"Expert forwards   : {totals.expert_forwards}")
        print(f"Decoded forwards  : {totals.decoded_forwards}")
        print(f"Accepted tokens   : {totals.accepted_tokens}")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
