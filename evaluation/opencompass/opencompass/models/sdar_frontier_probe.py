"""Single-source expert replacement and frozen-frontier diagnostic replays.

Current acceptance/attention are offline selection variables, not an online policy.
No replay is committed to generation. Reuses the existing idle MoE hooks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .sdar_dependency_probe import DependencyProbe, cache_signature, greedy_decisions


def source_candidates(frame, layer_index, previous=None, shuffle_seed=0):
    frontier = np.flatnonzero(frame['accepted'] & frame['mask'])
    sources = np.flatnonzero(frame['mask'] & ~frame['accepted'])
    if not len(frontier) or len(sources) < 2:
        return [], 'no_frontier' if not len(frontier) else 'too_few_sources'
    attention = frame['attention'][layer_index]
    scores = attention[np.ix_(frontier, sources)].mean(0)
    low, high = np.quantile(scores, [.2, .8])
    if low == high:
        return [], 'tied_dependency'
    shuffled = np.random.default_rng(shuffle_seed).permutation(scores)
    historical = (previous is not None
                  and previous['block_id'] == frame['block_id']
                  and previous['step'] + 1 == frame['step']
                  and np.array_equal(previous['positions'], frame['positions']))
    result = []
    for n, token in enumerate(sources):
        prob = frame['router_probs'][layer_index, token]
        original = frame['expert_ids'][layer_index, token].astype(int)
        k = len(original)
        if len(prob) <= k:
            raise ValueError('Need an expert outside original Top-k')
        outside = np.ones(len(prob), bool)
        outside[original] = False
        extra = int(np.argmax(np.where(outside, prob, -np.inf)))
        # Recorded IDs use actual torch.topk ordering; avoid inventing a tie order.
        removed = int(original[-1])
        mass = float(prob[original].sum())
        if mass <= 0 or not np.isfinite(prob).all():
            raise ValueError('Invalid router probabilities')
        distance = np.abs(frame['positions'][frontier] - frame['positions'][token])
        old_score = None
        if historical and previous['mask'][token] and previous['mask'][frontier].all():
            old_score = float(previous['attention'][layer_index][frontier, token].mean())
        # Truly historical frontier: high-confidence remaining positions chosen
        # using only previous confidence (top quarter, excluding source itself).
        old_frontier_score = None
        if historical and previous['mask'][token]:
            eligible = np.flatnonzero(previous['mask'] & ~previous['accepted'])
            eligible = eligible[eligible != token]
            if len(eligible):
                count = max(1, int(np.ceil(len(eligible) / 4)))
                order = np.argsort(-previous['confidence'][eligible], kind='stable')
                predicted = eligible[order[:count]]
                old_frontier_score = float(previous['attention'][layer_index][predicted, token].mean())
        result.append(dict(token=int(token), global_position=int(frame['positions'][token]),
            group='strong' if scores[n] >= high else 'weak' if scores[n] <= low else 'middle',
            dependency=float(scores[n]), shuffled_dependency=float(shuffled[n]),
            historical_dependency_to_current_frontier=old_score,
            historical_dependency_to_predicted_frontier=old_frontier_score,
            source_confidence=float(frame['confidence'][token]),
            nearest_frontier_distance=int(distance.min()), mean_frontier_distance=float(distance.mean()),
            original_pool=original.tolist(), replacement_pool=[*original[:-1].tolist(), extra],
            removed_expert=removed, replacement_expert=extra,
            removed_mass=float(prob[removed] / mass), replacement_mass=float(prob[extra] / mass),
            boundary_tie=bool(prob[removed] == prob[extra]),
            frontier_count=int(len(frontier)), mask_count=int(frame['mask'].sum())))
    return result, None


def match_sources(candidates, mass_tolerance=.03, confidence_tolerance=.05):
    best, best_cost = None, float('inf')
    for a in candidates:
        if a['group'] != 'strong':
            continue
        for b in candidates:
            if b['group'] != 'weak' or a['token'] == b['token']:
                continue
            if a['nearest_frontier_distance'] != b['nearest_frontier_distance']:
                continue
            ca, cb = a['source_confidence'], b['source_confidence']
            if (ca < .5, ca < .9) != (cb < .5, cb < .9):
                continue
            gaps = [abs(a[k] - b[k]) for k in ('removed_mass', 'replacement_mass')]
            if max(gaps) > mass_tolerance or abs(ca - cb) > confidence_tolerance:
                continue
            cost = sum(gaps) + abs(ca - cb)
            if cost < best_cost:
                best, best_cost = (a, b), cost
    return best


def select_sources(candidates, mass_tolerance=.03, confidence_tolerance=.05, seed=0):
    match = match_sources(candidates, mass_tolerance, confidence_tolerance)
    if match is not None:
        return match, 'strict_matched'
    strong = [c for c in candidates if c['group'] == 'strong']
    weak = [c for c in candidates if c['group'] == 'weak']
    if not strong or not weak:
        return None, 'unavailable'
    rng = np.random.default_rng(seed)
    # This is NOT a relaxed match. These observations are analyzed separately.
    return (strong[int(rng.integers(len(strong)))], weak[int(rng.integers(len(weak)))]), 'stratified_unmatched'


def receiver_snapshot(logits, frame, minimum, threshold):
    import torch
    pred, confidence, accepted = greedy_decisions(logits, frame['mask'], threshold, minimum)
    values = logits[0].detach().float()
    original = torch.as_tensor(frame['predictions'], device=values.device, dtype=torch.long)
    selected = values.gather(1, original[:, None]).squeeze(1)
    top_values, top_ids = values.topk(2, dim=-1)
    competitor = torch.where(top_ids[:, 0] == original, top_values[:, 1], top_values[:, 0])
    return dict(predictions=pred, confidence=confidence, accepted=accepted,
                original_logprob=(selected - values.logsumexp(-1)).cpu().numpy(),
                original_margin=(selected - competitor).cpu().numpy())


def measured_effect(baseline, changed, receivers, source):
    selected = np.asarray(receivers, int)
    logprob = changed['original_logprob'][selected] - baseline['original_logprob'][selected]
    margin = changed['original_margin'][selected] - baseline['original_margin'][selected]
    flips = changed['predictions'][selected] != baseline['predictions'][selected]
    lost = baseline['accepted'][selected] & ~changed['accepted'][selected]
    return dict(frontier_prediction_flip_rate=float(flips.mean()),
                frontier_acceptance_lost_rate=float(lost.mean()),
                frontier_logprob_delta=float(logprob.mean()),
                frontier_logprob_abs_delta=float(np.abs(logprob).mean()),
                frontier_margin_delta=float(margin.mean()),
                frontier_margin_abs_delta=float(np.abs(margin).mean()),
                source_prediction_flip=bool(changed['predictions'][source] != baseline['predictions'][source]),
                source_confidence_delta=float(changed['confidence'][source] - baseline['confidence'][source]),
                receiver_prediction_flips=flips.tolist(), receiver_acceptance_lost=lost.tolist(),
                receiver_logprob_delta=logprob.tolist(), receiver_margin_delta=margin.tolist())


class FrontierProbe(DependencyProbe):
    def __init__(self, model, output_dir, layer=24, stride=2, max_steps=48,
                 mass_tolerance=.03, confidence_tolerance=.05):
        super().__init__(model, output_dir, layer=layer, stride=stride, max_steps=max_steps,
                         mass_tolerance=mass_tolerance, confidence_tolerance=confidence_tolerance)
        if self.target.top_k != 8:
            raise ValueError('This protocol requires original Top-8 experts')
        self.previous = None
        self.protocol_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.replay_hash = hashlib.sha256(Path(__file__).with_name('sdar_dependency_probe.py').read_bytes()).hexdigest()

    def start_sequence(self, metadata):
        self.metadata, self.attempts, self.previous = dict(metadata), 0, None
        if (self.output_dir / f'frontier_sequence{metadata["sequence"]:06d}.jsonl').exists():
            raise FileExistsError('Refusing duplicate frontier probe sequence')

    def _write(self, record):
        path = self.output_dir / f'frontier_sequence{self.metadata["sequence"]:06d}.jsonl'
        with path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, allow_nan=False) + '\n')

    def run(self, observer, forward_kwargs, baseline_logits, extract_logits, threshold, minimum):
        import torch
        from torch.nn import functional as F
        frame = observer.last_record
        previous, self.previous = self.previous, frame
        if frame is None or frame['step'] % self.stride or self.attempts >= self.max_steps:
            return
        self.attempts += 1
        layer_index = observer.layer_ids.index(self.layer)
        key = f'{self.metadata["prompt_sha256"]}:{frame["block_id"]}:{frame["step"]}'
        shuffle_seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        candidates, reason = source_candidates(frame, layer_index, previous, shuffle_seed)
        match, selection = select_sources(candidates, self.mass_tolerance,
                                          self.confidence_tolerance, shuffle_seed + 1)
        frontier = np.flatnonzero(frame['mask'] & frame['accepted']).tolist()
        record = dict(schema='sdar_frontier_probe_v1', sample=self.metadata['prompt_sha256'],
            sequence=int(self.metadata['sequence']), block=int(frame['block_id']), step=int(frame['step']),
            layer=self.layer, source_sha256=self.metadata['model_source_sha256'],
            seed=self.metadata.get('seed'), threshold=float(threshold), minimum=int(minimum),
            protocol_sha256=self.protocol_hash, replay_sha256=self.replay_hash,
            generation_settings=self.metadata.get('generation_settings', {}),
            mass_tolerance=self.mass_tolerance, confidence_tolerance=self.confidence_tolerance,
            frontier=frontier, candidates=candidates, selection=selection, status='no_match',
            no_match_reason=reason or ('matching_constraints' if match is None else None))
        if match is None:
            self._write(record)
            return
        if forward_kwargs.get('store_kv') is not False:
            raise ValueError('Frontier replays must not store KV')
        baseline = receiver_snapshot(baseline_logits, frame, minimum, threshold)
        if not np.array_equal(baseline['accepted'], frame['accepted']):
            raise RuntimeError('Baseline acceptance does not match observer')
        if not np.array_equal(baseline['predictions'][frame['mask']], frame['predictions'][frame['mask']]):
            raise RuntimeError('Baseline prediction does not match observer')
        record['baseline_frontier_margin'] = baseline['original_margin'][frontier].tolist()
        cache = forward_kwargs['past_key_values']
        signature = cache_signature(cache)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        def replay(trial):
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.trial, self.gate_data = trial, None
            logits = extract_logits(self.model(**forward_kwargs), self.model)
            if cache_signature(cache) != signature:
                raise RuntimeError('KV mutated during frontier replay')
            if trial is not None:
                if self.gate_data is None:
                    raise RuntimeError('Target gate hook was not invoked')
                probs = F.softmax(self.gate_data[1], dim=-1, dtype=torch.float32).cpu().numpy()
                if not np.allclose(probs, frame['router_probs'][layer_index], atol=1e-6, rtol=1e-5):
                    raise RuntimeError('Gate changed before frontier intervention')
            return logits

        try:
            sham = replay(None)
            record['sham_logit_max_abs'] = float((sham - baseline_logits).abs().max().item())
            sham_stats = receiver_snapshot(sham, frame, minimum, threshold)
            if not np.array_equal(sham_stats['accepted'], baseline['accepted']) or not np.array_equal(
                    sham_stats['predictions'][frame['mask']], baseline['predictions'][frame['mask']]):
                raise RuntimeError('Sham replay changes decisions')
            del sham
            record['status'] = 'matched' if selection == 'strict_matched' else 'observed_unmatched'
            for candidate in match:
                token = candidate['token']
                control_logits = replay(dict(tokens=[token], pool=candidate['original_pool']))
                control = receiver_snapshot(control_logits, frame, minimum, threshold)
                control_error = float((control_logits - baseline_logits).abs().max().item())
                del control_logits
                changed_logits = replay(dict(tokens=[token], pool=candidate['replacement_pool']))
                changed = receiver_snapshot(changed_logits, frame, minimum, threshold)
                del changed_logits
                control_ok = (np.array_equal(control['accepted'], baseline['accepted']) and
                    np.array_equal(control['predictions'][frame['mask']], baseline['predictions'][frame['mask']]))
                record[candidate['group']] = dict(candidate,
                    control_decisions_equal=bool(control_ok), control_logit_max_abs=control_error,
                    control_effects=measured_effect(baseline, control, frontier, token),
                    effects=measured_effect(baseline, changed, frontier, token),
                    relative_to_control=measured_effect(control, changed, frontier, token))
                if not control_ok:
                    record['status'] = 'control_failed'
        finally:
            self.trial, self.gate_data = None, None
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self._write(record)
