#!/usr/bin/env python3
"""Calculate APF, TPF, and APT from legacy TEAM Mark or unified JSONL.

The mark evaluator writes one expert-distribution JSON file and one
decoded-order JSON file per generated sequence.  APF is the mean number of
distinct, actually activated experts (expert id != -1) per MoE layer and
forward pass.  TPF is the mean number of accepted tokens per forward pass.
APT is APF / TPF, matching the definition used in the TEAM paper.

For original TEAM Mark records, transfer_index omits verified speculative
prefills. By default, reconstruct mask-to-token transitions from the next
forward's delayed-cache (-1) markers. This requires the original four-branch
Mark layout and completed blocks. Use --accepted-token-mode transfer for the
old counting rule or other record formats. Compact JSONL totals are unchanged;
missing speculative counts cannot be recovered from those totals alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class MetricTotals:
    activated_experts: int = 0
    layer_forwards: int = 0
    expert_forwards: int = 0
    accepted_tokens: int = 0
    decoded_forwards: int = 0
    transfer_accepted_tokens: int = 0
    generation_latencies_s: list[float] = field(default_factory=list)
    timed_accepted_tokens: int = 0
    pool_layer_forwards: int = 0
    pool_remainder_layer_forwards: int = 0
    pool_remainder_slots_sum: int = 0
    pool_remainder_records: int = 0
    stop_reason_counts: dict[str, int] = field(default_factory=dict)

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
            totals.transfer_accepted_tokens += sum(transfer_index)
            totals.decoded_forwards += 1


def recover_team_accepted_tokens(
    expert_payload: dict[str, Any],
    decoded_payload: dict[str, Any],
    block_length: int,
    denoising_steps: int,
) -> int:
    """Recover accepted tokens for the original BS=1 four-branch Mark loop.

    At forward j+1 the cached positions are precisely the non-mask positions
    after forward j, excluding j's transfer_index. Thus their union restores
    the selected branch's state, including its verified speculative prefills.
    The last state is a complete block: in the original loop, fewer than
    denoising_steps+1 recorded steps implies exit through the no-mask check.
    This is NOT suitable for full_skip, cache refresh, or partial run logs.
    """
    if block_length <= 0 or not 0 < denoising_steps <= block_length:
        raise ValueError('Require 0 < denoising_steps <= block_length')
    if block_length % denoising_steps:
        raise ValueError('block_length must be divisible by denoising_steps')
    if not decoded_payload or forward_keys(expert_payload) != forward_keys(decoded_payload):
        raise ValueError('TEAM recovery requires matching, nonempty forward records')
    block_keys = sorted(decoded_payload, key=numeric_key)
    if block_keys != [str(i) for i in range(len(block_keys))]:
        raise ValueError('TEAM recovery requires contiguous blocks starting at 0')
    all_positions = set(range(block_length))
    accepted = 0
    expected_layers: set[str] | None = None
    for block in block_keys:
        steps = decoded_payload[block]
        keys = sorted(steps, key=numeric_key)
        if not keys or keys != [str(i) for i in range(len(keys))]:
            raise ValueError(f'Non-contiguous TEAM steps in block {block}')
        if len(keys) >= denoising_steps + 1:
            raise ValueError(
                f'Block {block} may be incomplete (step limit reached); '
                'cannot infer its final state. Check --denoising-steps or '
                'use --accepted-token-mode transfer for the old count.')
        transfers: list[set[int]] = []
        caches: list[set[int]] = []
        for index, step in enumerate(keys):
            record = steps[step]
            transfer = record.get('transfer_index') if isinstance(record, dict) else None
            if (not isinstance(transfer, list) or len(transfer) != 1
                    or not isinstance(transfer[0], list)
                    or len(transfer[0]) != block_length
                    or any(not isinstance(x, bool) for x in transfer[0])):
                raise ValueError(f'Expected BS=1 transfer_index at {block}/{step}')
            transfers.append({i for i, flag in enumerate(transfer[0]) if flag})
            layers = expert_payload[block][step]
            if not isinstance(layers, dict) or not layers:
                raise ValueError(f'Missing expert layers at {block}/{step}')
            if expected_layers is None:
                expected_layers = set(layers)
            if set(layers) != expected_layers:
                raise ValueError(f'MoE layer keys differ at {block}/{step}')
            if index == 0:
                caches.append(set())
                continue
            layer_cache = None
            for layer, rows in layers.items():
                # Accept both [4*L, K] and [1, 4*L, K] serialization.
                if (isinstance(rows, list) and len(rows) == 1
                        and isinstance(rows[0], list) and rows[0]
                        and isinstance(rows[0][0], list)):
                    rows = rows[0]
                if not isinstance(rows, list) or len(rows) != 4 * block_length:
                    raise ValueError(f'Expected four-branch experts at {block}/{step}/{layer}')
                flags = []
                width = None
                for row in rows:
                    if (not isinstance(row, list) or not row
                            or any(isinstance(x, bool) or not isinstance(x, int)
                                   or x < -1 for x in row)):
                        raise ValueError(f'Invalid expert row at {block}/{step}/{layer}')
                    if width is None:
                        width = len(row)
                    if len(row) != width or (any(x == -1 for x in row)
                                              and not all(x == -1 for x in row)):
                        raise ValueError(f'Inconsistent expert row at {block}/{step}/{layer}')
                    flags.append(all(x == -1 for x in row))
                if any(flags[b * block_length:(b + 1) * block_length]
                       != flags[:block_length] for b in range(1, 4)):
                    raise ValueError(f'Branch cache masks differ at {block}/{step}/{layer}')
                current = {i for i, flag in enumerate(flags[:block_length]) if flag}
                if layer_cache is not None and current != layer_cache:
                    raise ValueError(f'Layer cache masks differ at {block}/{step}')
                layer_cache = current
            caches.append(layer_cache)
            if caches[-1] & (transfers[index] | transfers[index - 1]):
                raise ValueError(f'Cached and fresh positions overlap at {block}/{step}')
        # At step 1, only the prompt tail predates step 0's transfers.
        before = caches[1] if len(keys) > 1 else all_positions - transfers[0]
        if before != set(range(len(before))) or (block != '0' and before):
            raise ValueError(f'Invalid initial prompt-tail cache in block {block}')
        for index, step in enumerate(keys):
            after = (caches[index + 1] | transfers[index]
                     if index + 1 < len(keys) else all_positions)
            if not before <= after or transfers[index] & before:
                raise ValueError(f'Non-monotonic decode state at {block}/{step}')
            extra = len(after - before) - len(transfers[index])
            if extra not in (0, 1, 2) or (index == 0 and extra != 0):
                raise ValueError(f'Invalid speculative acceptance count at {block}/{step}: {extra}')
            accepted += len(after - before)
            before = after
    return accepted


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
    accepted_token_mode: str = 'team-mark',
    block_length: int = 32,
    denoising_steps: int = 32,
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

    if accepted_token_mode not in {'team-mark', 'transfer'}:
        raise ValueError(f'Unknown accepted token mode: {accepted_token_mode}')
    if accepted_token_mode == 'team-mark' and (only_expert or only_decoded):
        raise ValueError('TEAM recovery requires matching expert/decoded files')
    totals = MetricTotals()
    # Stream one sequence pair at a time; expert dumps can occupy many GB.
    for relative_path in sorted(set(expert_map) | set(decoded_map)):
        expert = load_json(expert_map[relative_path]) if relative_path in expert_map else None
        decoded = load_json(decoded_map[relative_path]) if relative_path in decoded_map else None
        if expert is not None:
            count_experts(expert, totals)
        if decoded is not None:
            count_accepted_tokens(decoded, totals)
        if expert is None or decoded is None:
            continue
        expert_keys = forward_keys(expert)
        decoded_keys = forward_keys(decoded)
        if expert_keys != decoded_keys:
            warnings.append(
                f"Forward records differ for {relative_path}: "
                f"expert={len(expert_keys)}, decoded={len(decoded_keys)}")
        if accepted_token_mode == 'team-mark':
            try:
                recovered = recover_team_accepted_tokens(
                    expert, decoded, block_length, denoising_steps)
            except ValueError as error:
                raise ValueError(f'Cannot recover {relative_path}: {error}') from error
            recorded = sum(sum(leaf_values(record['transfer_index']))
                           for steps in decoded.values() for record in steps.values())
            totals.accepted_tokens += recovered - recorded

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
    signatures = set()
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
                    signatures.add((record.get('metrics_collected', True),
                        record.get('accepted_token_mode', 'legacy_unspecified'),
                        record.get('team_pool_strategy', 'legacy_unspecified'),
                        record.get('team_pool_budget'), record.get('team_pool_coverage_target'),
                        record.get('metrics_backend', 'legacy_unspecified'),
                        record.get('latency_scope', 'legacy_unspecified'),
                        record.get('model_source_sha256'), record.get('seed_base'),
                        json.dumps(record.get('generation_settings', {}), sort_keys=True)))
                    if len(signatures) > 1:
                        raise ValueError('Mixed strategy/counting/backend/timing settings; aggregate each run separately')
                    totals.activated_experts += int(record['activated_experts'])
                    totals.layer_forwards += int(record['layer_forwards'])
                    totals.accepted_tokens += int(record['accepted_tokens'])
                    totals.decoded_forwards += int(record['denoising_forwards'])
                    totals.expert_forwards += int(record['denoising_forwards'])
                    for name in ('pool_layer_forwards', 'pool_remainder_layer_forwards',
                                 'pool_remainder_slots_sum'):
                        value = record.get(name, 0)
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise ValueError(f'Invalid {name} at {path}:{line_number}')
                        setattr(totals, name, getattr(totals, name) + value)
                    totals.pool_remainder_records += int('pool_remainder_layer_forwards' in record)
                    if record.get('pool_remainder_layer_forwards', 0) > record.get('pool_layer_forwards', 0):
                        raise ValueError(f'Remainder layer count exceeds pool layers at {path}:{line_number}')
                    reason = record.get('generation_stop_reason')
                    if reason is not None:
                        if reason not in {'stop_token', 'length_limit', 'incomplete'}:
                            raise ValueError(f'Invalid generation_stop_reason at {path}:{line_number}')
                        totals.stop_reason_counts[reason] = totals.stop_reason_counts.get(reason, 0) + 1
                    latency = record.get('generation_latency_s')
                    if latency is not None:
                        latency = float(latency)
                        if not math.isfinite(latency) or latency <= 0:
                            raise ValueError(f'Invalid generation latency at {path}:{line_number}')
                        totals.generation_latencies_s.append(latency)
                        totals.timed_accepted_tokens += int(record['accepted_tokens'])
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
        '--accepted-token-mode', choices=('team-mark', 'transfer'), default='team-mark',
        help=('Mark JSON only: recover verified speculative tokens from original '
              'four-branch delayed-cache records (default: team-mark), or count '
              'transfer_index only (transfer). Not applicable to compact JSONL.'))
    parser.add_argument('--block-length', type=int, default=32,
                        help='Original TEAM Mark block length (default: 32).')
    parser.add_argument('--denoising-steps', type=int, default=32,
                        help='Original TEAM Mark denoising_steps (default: 32).')
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
                args.pattern, args.accepted_token_mode,
                args.block_length, args.denoising_steps)
            metric_files = None
            metric_records = None
        result = {
            "APF": totals.apf if totals.layer_forwards else None,
            "TPF": totals.tpf,
            "APT": totals.apt if totals.layer_forwards else None,
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
            if totals.stop_reason_counts:
                result['generation_stop_reason_counts'] = totals.stop_reason_counts
            if totals.pool_remainder_records and totals.pool_layer_forwards:
                result.update(pool_remainder_layer_forwards=totals.pool_remainder_layer_forwards,
                    pool_remainder_layer_fraction=totals.pool_remainder_layer_forwards / totals.pool_layer_forwards,
                    pool_mean_remainder_slots=totals.pool_remainder_slots_sum / totals.pool_layer_forwards)
            times = totals.generation_latencies_s
            if times:
                result.update(generation_latency_s=sum(times),
                    mean_generation_latency_s=statistics.mean(times),
                    median_generation_latency_s=statistics.median(times),
                    p95_generation_latency_s=statistics.quantiles(times, n=100, method='inclusive')[94]
                        if len(times) > 1 else times[0],
                    latency_records=len(times),
                    accepted_tokens_per_second=totals.timed_accepted_tokens / sum(times))
                if len(times) != metric_records:
                    warnings.append('Some records have no latency; timing aggregates use only timed records')
            else:
                result['latency_records'] = 0
        else:
            result['accepted_token_mode'] = args.accepted_token_mode
            result['transfer_accepted_tokens'] = totals.transfer_accepted_tokens
            result['recovered_speculative_tokens'] = (
                totals.accepted_tokens - totals.transfer_accepted_tokens)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"APF: {result['APF']:.4f}" if result['APF'] is not None else 'APF: unavailable (timing-only run)')
        print(f"TPF: {result['TPF']:.4f}")
        print(f"APT: {result['APT']:.4f}" if result['APT'] is not None else 'APT: unavailable (timing-only run)')
        if result.get('latency_records'):
            print(f"Generation latency total: {result['generation_latency_s']:.4f} s")
            print(f"Generation latency mean : {result['mean_generation_latency_s']:.4f} s/sample")
            print(f"Generation latency p95  : {result['p95_generation_latency_s']:.4f} s/sample")
            print(f"Accepted tokens/s       : {result['accepted_tokens_per_second']:.4f}")
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
        if result.get('generation_stop_reason_counts'):
            print('Generation stops  : ' + json.dumps(result['generation_stop_reason_counts'], sort_keys=True))
        if 'pool_remainder_layer_forwards' in result:
            print(f"Pool remainder layers: {result['pool_remainder_layer_forwards']} "
                  f"({result['pool_remainder_layer_fraction']:.2%})")
            print(f"Mean remainder slots : {result['pool_mean_remainder_slots']:.4f}")
        if metric_files is None:
            print(f"Accepted mode     : {args.accepted_token_mode}")
            print(f"Transfer tokens   : {totals.transfer_accepted_tokens}")
            print(f"Recovered spec.   : {result['recovered_speculative_tokens']}")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
