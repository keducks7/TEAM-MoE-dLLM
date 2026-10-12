"""Offline-style paired route interventions embedded in a Vanilla run.

Extra forwards are diagnostics, never a speed method. Original decoding uses
only the original forward. Current attention is allowed for offline selection.
"""
import json
from pathlib import Path

import numpy as np


def confidence_bin(value):
    return 0 if value < .5 else 1 if value < .9 else 2


def candidate_pairs(frame, layer_index, pool_size=12):
    """DES-style pool inside two tokens' union; protect both original Top-1s."""
    mask = frame['mask']
    positions = np.flatnonzero(mask)
    if len(positions) < 4:
        return []
    probs = frame['router_probs'][layer_index]
    ids = frame['expert_ids'][layer_index]
    k = ids.shape[-1]
    if pool_size < k:
        raise ValueError('Pool must contain at least the original per-token Top-k')
    i, j = np.triu_indices(len(positions), 1)
    i, j = positions[i], positions[j]
    a = frame['attention'][layer_index]
    scores = (a[i, j] + a[j, i]) / 2
    lo, hi = np.quantile(scores, [.2, .8])
    if lo == hi:
        return []
    result = []
    for left, right, score in zip(i, j, scores):
        group = 'strong' if score >= hi else 'weak' if score <= lo else None
        if group is None:
            continue
        union = np.union1d(ids[left], ids[right])
        if len(union) <= pool_size:
            continue  # No actual restriction; do not count as intervention.
        protected = np.unique([ids[left, 0], ids[right, 0]])
        free = union[~np.isin(union, protected)]
        vote = probs[left, free] + probs[right, free]
        ranked = free[np.argsort(-vote, kind='stable')]
        pool = np.sort(np.r_[protected, ranked[:pool_size - len(protected)]]).astype(int)
        losses = []
        for token in (left, right):
            original = ids[token]
            mass = probs[token, original]
            losses.append(float(mass[~np.isin(original, pool)].sum() / mass.sum()))
        confidences = sorted([float(frame['confidence'][left]), float(frame['confidence'][right])])
        controls = sorted([
            dict(confidence=float(frame['confidence'][token]), removed_mass=loss,
                 accepted=bool(frame['accepted'][token]))
            for token, loss in zip((left, right), losses)],
            key=lambda c: (c['accepted'], c['confidence'], c['removed_mass']))
        overlap = int(np.intersect1d(ids[left], ids[right]).size)
        accepted_role = int(frame['accepted'][left]) + int(frame['accepted'][right])
        result.append(dict(group=group, tokens=[int(left), int(right)],
            global_positions=frame['positions'][[left, right]].astype(int).tolist(),
            dependency=float(score), pool=pool.tolist(), pool_size=int(pool_size),
            original_overlap=overlap, removed_mass=losses,
            removed_mass_mean=float(np.mean(losses)), confidence=confidences, token_controls=controls,
            stratum=(int(abs(frame['positions'][left] - frame['positions'][right])),
                     *[confidence_bin(c) for c in confidences], overlap, accepted_role)))
    return result


def select_matched_pairs(frame, layer_index, pool_size=12, mass_tolerance=.03,
                        confidence_tolerance=.05):
    pairs = candidate_pairs(frame, layer_index, pool_size)
    strong = [p for p in pairs if p['group'] == 'strong']
    weak = [p for p in pairs if p['group'] == 'weak']
    best, best_cost = None, float('inf')
    for a in strong:
        for b in weak:
            if a['stratum'] != b['stratum'] or set(a['tokens']) & set(b['tokens']):
                continue
            # Align per-token confidence, loss and acceptance together, rather
            # than matching independently sorted marginal distributions.
            controls = list(zip(a['token_controls'], b['token_controls']))
            if any(x['accepted'] != y['accepted'] for x, y in controls):
                continue
            confidence_gap = max(abs(x['confidence'] - y['confidence']) for x, y in controls)
            mass_gap = max(abs(x['removed_mass'] - y['removed_mass']) for x, y in controls)
            if mass_gap > mass_tolerance or confidence_gap > confidence_tolerance:
                continue
            cost = mass_gap + confidence_gap
            if cost < best_cost:
                best, best_cost = (a, b), cost
    return best


def restricted_output(module, output, hidden, logits, tokens, pool):
    """Recompute routed experts only for two rows; preserve all other rows."""
    import torch
    from torch.nn import functional as F
    if getattr(module, 'shared_expert', None) is not None:
        raise RuntimeError('Shared experts are unsupported by this SDAR diagnostic')
    tensor = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(tensor) or hidden.ndim != 2:
        raise RuntimeError('Unsupported Vanilla MoE output/input')
    rows = torch.tensor(tokens, device=hidden.device, dtype=torch.long)
    allowed = torch.tensor(pool, device=logits.device, dtype=torch.long)
    probs = F.softmax(logits.detach(), dim=-1, dtype=torch.float32).index_select(0, rows)
    weights, slots = probs.index_select(-1, allowed).topk(int(module.top_k), dim=-1)
    selected = allowed[slots]
    if getattr(module, 'norm_topk_prob', False):
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights.to(hidden.dtype)
    current = hidden.index_select(0, rows)
    replacement = torch.zeros_like(current)
    for expert_id in torch.unique(selected).tolist():
        token_indices, expert_slots = torch.where(selected == expert_id)
        value = module.experts[expert_id](current.index_select(0, token_indices))
        value = value.to(device=hidden.device) * weights[token_indices, expert_slots, None]
        replacement.index_add_(0, token_indices, value.to(replacement.dtype))
    changed = tensor.clone().reshape(-1, tensor.shape[-1])
    changed.index_copy_(0, rows, replacement)
    changed = changed.reshape_as(tensor)
    return (changed, *output[1:]) if isinstance(output, tuple) else changed


def greedy_decisions(logits, mask, threshold, minimum):
    import torch
    from torch.nn import functional as F
    # Same dtype and strict > threshold as original sample_with_greedy.
    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)
    predictions, confidence = predictions[0], confidence[0]
    masked = torch.as_tensor(mask, device=confidence.device, dtype=torch.bool)
    eligible = torch.where(masked, confidence, -torch.inf)
    accepted = (eligible > threshold) if threshold < 1 else torch.zeros_like(masked)
    if threshold >= 1 or int(accepted.sum()) < minimum:
        accepted = torch.zeros_like(masked)
        accepted[eligible.topk(minimum).indices] = True
    return (predictions.detach().cpu().numpy(), confidence.detach().float().cpu().numpy(),
            accepted.detach().cpu().numpy())


def cache_signature(cache):
    result = []
    for layer in range(len(cache)):
        for tensor in cache[layer][:2]:
            try:
                version = tensor._version
            except RuntimeError:
                version = None
            result.append((tuple(tensor.shape), tensor.data_ptr(), version))
    return result


def decision_effect(frame, predictions, confidence, accepted, tokens):
    own = np.zeros_like(frame['mask'], dtype=bool)
    own[tokens] = True
    other = frame['mask'] & ~own
    flips = predictions != frame['predictions']
    lost = frame['accepted'] & ~accepted
    gained = accepted & ~frame['accepted']
    return dict(own_prediction_flip_rate=float(flips[own].mean()),
                other_prediction_flip_rate=float(flips[other].mean()) if other.any() else None,
                own_confidence_delta=float((confidence - frame['confidence'])[own].mean()),
                other_confidence_abs_delta=float(np.abs(confidence - frame['confidence'])[other].mean()) if other.any() else None,
                own_accepted_lost=int((lost & own).sum()), other_accepted_lost=int((lost & other).sum()),
                own_accepted_gained=int((gained & own).sum()), other_accepted_gained=int((gained & other).sum()),
                baseline_accepted_prediction_flips=int((flips & frame['accepted']).sum()),
                accepted_set_changes=int((lost | gained).sum()))


class DependencyProbe:
    def __init__(self, model, output_dir, layer=24, pool_size=12, stride=2,
                 max_steps=48, mass_tolerance=.03, confidence_tolerance=.05):
        self.model, self.layer = model, int(layer)
        self.pool_size, self.stride, self.max_steps = int(pool_size), int(stride), int(max_steps)
        if min(self.stride, self.max_steps) < 1 or not 0 <= self.layer < len(model.model.layers):
            raise ValueError('Invalid probe layer/stride/max_steps')
        if min(mass_tolerance, confidence_tolerance) < 0:
            raise ValueError('Matching tolerances must be nonnegative')
        self.mass_tolerance, self.confidence_tolerance = mass_tolerance, confidence_tolerance
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target = model.model.layers[self.layer].mlp
        if not hasattr(self.target, 'experts') or self.pool_size < self.target.top_k:
            raise ValueError('Unsupported target MoE/pool size')
        self.trial = None
        self.gate_data = None

        def gate(_module, inputs, output):
            if self.trial is not None:
                self.gate_data = (inputs[0].detach(), output.detach())

        def replace(module, _args, output):
            if self.trial is None:
                return None
            if self.gate_data is None:
                raise RuntimeError('No gate data in intervention replay')
            hidden, logits = self.gate_data
            return restricted_output(module, output, hidden, logits,
                                     self.trial['tokens'], self.trial['pool'])

        self.handles = [self.target.gate.register_forward_hook(gate),
                        self.target.register_forward_hook(replace)]

    def start_sequence(self, metadata):
        self.metadata = dict(metadata)
        self.attempts = 0
        self.matched = 0
        path = self.output_dir / f'probe_sequence{self.metadata["sequence"]:06d}.jsonl'
        if path.exists():
            raise FileExistsError('Refusing to append duplicate probe sequences')

    def run(self, observer, forward_kwargs, baseline_logits, extract_logits,
            threshold, minimum):
        import torch
        frame = observer.last_record
        if frame is None or frame['step'] % self.stride or self.attempts >= self.max_steps:
            return
        self.attempts += 1
        if self.layer not in observer.layer_ids:
            raise ValueError('Target probe layer must be included in observation layers')
        layer_index = observer.layer_ids.index(self.layer)
        match = select_matched_pairs(frame, layer_index, self.pool_size,
                                     self.mass_tolerance, self.confidence_tolerance)
        record = dict(schema='sdar_dependency_probe_v1',
            sample=self.metadata['prompt_sha256'], sequence=self.metadata['sequence'],
            block=int(frame['block_id']), step=int(frame['step']), layer=self.layer,
            source_sha256=self.metadata['model_source_sha256'],
            threshold=float(threshold), seed=self.metadata.get('seed'),
            pool_size=self.pool_size, mask_count=int(frame['mask'].sum()),
            mass_tolerance=self.mass_tolerance, confidence_tolerance=self.confidence_tolerance)
        if match is None:
            record['status'] = 'no_match'
            self._write(record)
            return
        self.matched += 1
        if forward_kwargs.get('store_kv') is not False:
            raise ValueError('Probe replays must not store KV')
        cache = forward_kwargs['past_key_values']
        before_cache = cache_signature(cache)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        baseline_pred, baseline_conf, baseline_accept = greedy_decisions(
            baseline_logits, frame['mask'], threshold, minimum)
        if (not np.array_equal(baseline_pred[frame['mask']], frame['predictions'][frame['mask']])
                or not np.array_equal(baseline_accept, frame['accepted'])):
            raise RuntimeError('Probe decision implementation differs from original decoding')
        try:
            # Sham replay exposes numerical nondeterminism before attributing
            # any differences to route restriction.
            self.trial = None
            replay = extract_logits(self.model(**forward_kwargs), self.model)
            pred, conf, accepted = greedy_decisions(replay, frame['mask'], threshold, minimum)
            sham_error = float((replay - baseline_logits).abs().max().item())
            if (not np.array_equal(pred[frame['mask']], baseline_pred[frame['mask']])
                    or not np.array_equal(accepted, baseline_accept)):
                raise RuntimeError('Sham replay changes decode decisions; stop before interpreting probes')
            if cache_signature(cache) != before_cache:
                raise RuntimeError('KV changed during sham replay')
            record.update(status='matched', sham_logit_max_abs=sham_error,
                          baseline_confidence_error=float(np.abs(baseline_conf - frame['confidence']).max()))
            for trial in match:
                torch.random.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                self.trial, self.gate_data = trial, None
                logits = extract_logits(self.model(**forward_kwargs), self.model)
                # Earlier layers must be unchanged: route intervention begins
                # ONLY after the target gate. Check the replayed gate scores.
                if 'router_probs' in frame:
                    from torch.nn import functional as F
                    replay_probs = F.softmax(self.gate_data[1], dim=-1, dtype=torch.float32).cpu().numpy()
                    expected_probs = frame['router_probs'][layer_index]
                    if not np.allclose(replay_probs, expected_probs, atol=1e-6, rtol=1e-5):
                        raise RuntimeError('Target gate inputs changed before route intervention')
                pred, conf, accepted = greedy_decisions(logits, frame['mask'], threshold, minimum)
                record[trial['group']] = dict(trial, effects=decision_effect(
                    frame, pred, conf, accepted, trial['tokens']))
                if cache_signature(cache) != before_cache:
                    raise RuntimeError('KV changed during probe replay')
        finally:
            self.trial, self.gate_data = None, None
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self._write(record)

    def _write(self, record):
        path = self.output_dir / f'probe_sequence{self.metadata["sequence"]:06d}.jsonl'
        with path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')
