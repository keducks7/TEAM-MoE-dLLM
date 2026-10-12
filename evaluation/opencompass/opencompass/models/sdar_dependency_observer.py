"""Read-only Vanilla SDAR observations; never replace model outputs or SDPA.

Attention probabilities are reconstructed from the executed normalized Q/K,
RoPE and prefix KV. They are an FP32 diagnostic proxy, not fused-kernel output.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def _attention_implementation(module):
    """Find class-defined attention, independent of Accelerate's forward wrapper.

    device_map dispatch can replace the instance forward with functools.partial.
    The class implementation still provides SDAR's source and RoPE globals.
    Never replace the executed forward or remove its device-management hooks.
    """
    implementation = inspect.unwrap(type(module).forward)
    if not inspect.isfunction(implementation):
        raise RuntimeError('Unsupported attention: cannot inspect class forward')
    return implementation


def reconstruct_attention(module, query, key, kwargs):
    """Head-mean current-block attention with the FULL prefix denominator."""
    if module.training or getattr(module, 'attention_dropout', 0) != 0:
        raise ValueError('Dependency observations require eval and zero attention dropout')
    rotary = _attention_implementation(module).__globals__.get('apply_rotary_pos_emb')
    if rotary is None:
        raise RuntimeError('Unsupported attention: cannot locate apply_rotary_pos_emb')
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    query, key = rotary(query, key, *kwargs['position_embeddings'])
    tokens = query.shape[-2]
    cache = kwargs.get('past_key_value')
    if cache is not None and len(cache) > module.layer_idx:
        key = torch.cat([cache[module.layer_idx][0], key], dim=-2)
    if query.shape[0] != 1 or key.shape[-2] < tokens:
        raise ValueError('Only batch_size=1, unpacked Vanilla attention is supported')
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    mask = kwargs.get('attention_mask')
    result = torch.zeros((tokens, tokens), device=query.device, dtype=torch.float32)
    # Limit peak memory: do not materialize [all heads, Q, full prefix].
    for head in range(query.shape[1]):
        logits = query[:, head:head + 1].float() @ key[:, head:head + 1].float().transpose(-1, -2)
        logits *= module.scaling
        if mask is not None:
            allowed = mask
            if allowed.ndim == 3:
                allowed = allowed.unsqueeze(1)
            if allowed.shape[1] != 1:
                allowed = allowed[:, head:head + 1]
            if allowed.dtype == torch.bool:
                logits = logits.masked_fill(~allowed, -torch.inf)
            else:
                logits = logits + allowed
        probs = logits.softmax(dim=-1)[0, 0, :, -tokens:]
        result += probs / query.shape[1]
    if not torch.isfinite(result).all():
        raise RuntimeError('Non-finite reconstructed attention; check mask compatibility')
    return result


class DependencyObserver:
    def __init__(self, model, output_dir, layers='auto', max_blocks=8):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_blocks = int(max_blocks)
        if self.max_blocks < 1:
            raise ValueError('dependency_max_blocks must be positive')
        decoder_layers = model.model.layers
        n = len(decoder_layers)
        self.layer_ids = sorted(set([0, n // 2, n - 1] if layers == 'auto'
                                    else range(n) if layers == 'all'
                                    else map(int, layers.split(','))))
        if not self.layer_ids or min(self.layer_ids) < 0 or max(self.layer_ids) >= n:
            raise ValueError(f'Invalid dependency layers for {n} layers')
        self.active = False
        self.sequence = 0
        self.handles = []
        self.capture = {}
        for layer_id in self.layer_ids:
            layer = decoder_layers[layer_id]
            attn, moe = layer.self_attn, layer.mlp
            # Fail explicitly rather than silently observe incompatible remote code.
            source = inspect.getsource(_attention_implementation(attn))
            if 'scaled_dot_product_attention' not in source:
                raise RuntimeError('This observer supports SDAR SDPA attention only')
            if 'attention_mask.bool()' not in source:
                raise RuntimeError('Unsupported SDAR attention-mask conversion')
            if not hasattr(moe, 'experts') or not hasattr(attn, 'q_norm'):
                raise RuntimeError('Unsupported SDAR module layout')

            def before(_module, args, kwargs, lid=layer_id):
                if self.active:
                    if args:
                        raise RuntimeError('Expected SDAR attention keyword inputs')
                    self.capture[lid] = {'kwargs': dict(kwargs)}

            def norm_hook(_module, _args, output, lid=layer_id, name='query'):
                if self.active:
                    self.capture[lid][name] = output.detach()

            def after(module, _args, _kwargs, _output, lid=layer_id):
                if self.active:
                    item = self.capture[lid]
                    kwargs = item['kwargs']
                    # The actual SDAR implementation casts its mask to bool.
                    kwargs['attention_mask'] = kwargs['attention_mask'].bool()
                    item['attention'] = reconstruct_attention(
                        module, item.pop('query'), item.pop('key'), kwargs).cpu().numpy()
                    item.pop('kwargs')

            def gate_hook(_module, args, logits, lid=layer_id, topk=int(moe.top_k)):
                if not self.active:
                    return
                hidden = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                if logits.ndim != 2 or logits.shape[0] != self.pending['mask'].size:
                    raise RuntimeError('Gate rows do not match Vanilla current-block positions')
                probs = F.softmax(logits.detach(), dim=-1, dtype=torch.float32)
                weights, ids = probs.topk(topk, dim=-1)
                normalized = F.normalize(hidden, dim=-1)
                item = self.capture[lid]
                item.update(router_probs=probs.cpu().numpy(),
                            expert_ids=ids.cpu().numpy().astype(np.int16),
                            topk_probs=weights.cpu().numpy(),
                            hidden_cosine=(normalized @ normalized.T).cpu().numpy())

            self.handles.append(attn.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(attn.q_norm.register_forward_hook(norm_hook))
            def key_hook(module, args, output, lid=layer_id):
                return norm_hook(module, args, output, lid=lid, name='key')
            self.handles.append(attn.k_norm.register_forward_hook(key_hook))
            self.handles.append(attn.register_forward_hook(after, with_kwargs=True))
            self.handles.append(moe.gate.register_forward_hook(gate_hook))

    def start_sequence(self, input_ids, model_path, threshold, generation_settings=None):
        self.records = []
        self.blocks = []
        self.metadata = {
            'schema': 'sdar_dependency_v1', 'method': 'vanilla',
            'sequence': self.sequence, 'pid': os.getpid(), 'model_path': model_path,
            'threshold': threshold, 'layer_ids': self.layer_ids,
            'max_blocks': self.max_blocks, 'attention': 'fp32_qk_reconstruction',
            'attention_normalization': 'full_prefix_and_current_block',
            'topk_probs': 'raw_softmax_not_topk_renormalized',
            'prompt_sha256': hashlib.sha256(
                input_ids.detach().cpu().numpy().tobytes()).hexdigest(),
            'prompt_length': int(input_ids.shape[1]),
            'num_decoder_layers': len(self.model.model.layers),
            'seed': int(torch.initial_seed()),
            'generation_settings': generation_settings or {},
        }
        source = Path(inspect.getfile(type(self.model)))
        self.metadata['model_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()

    def begin_step(self, block_id, step, positions, token_ids, mask):
        self.last_record = None
        if token_ids.shape[0] != 1:
            raise ValueError('Dependency collection requires batch_size=1')
        if block_id not in self.blocks:
            self.blocks.append(block_id)
        self.active = self.blocks.index(block_id) < self.max_blocks
        if not self.active:
            return
        self.capture = {}
        self.pending = dict(block_id=block_id, step=step,
                            positions=positions[0].detach().cpu().numpy(),
                            token_ids=token_ids[0].detach().cpu().numpy().copy(),
                            mask=mask[0].detach().cpu().numpy().copy())

    def end_forward(self):
        self.active = False

    def finish_step(self, predictions, confidence, accepted):
        if not self.capture:
            return
        expected = {'attention', 'router_probs', 'expert_ids', 'topk_probs', 'hidden_cosine'}
        record = self.pending
        for name in expected:
            if any(name not in self.capture.get(lid, {}) for lid in self.layer_ids):
                raise RuntimeError(f'Missing dependency hook data: {name}')
            record[name] = np.stack([self.capture[lid][name] for lid in self.layer_ids])
        record.update(predictions=predictions[0].detach().cpu().numpy(),
                      confidence=confidence[0].detach().float().cpu().numpy(),
                      accepted=accepted[0].detach().cpu().numpy())
        self.records.append(record)
        self.last_record = record
        self.capture = {}

    def finish_sequence(self):
        if not self.records:
            raise RuntimeError('Dependency observation produced no records')
        target = self.output_dir / f'dependency_pid{os.getpid()}_sequence{self.sequence:06d}.npz'
        if target.exists():
            raise FileExistsError(target)
        arrays = {name: np.stack([r[name] for r in self.records]) for name in self.records[0]}
        np.savez_compressed(target, metadata=np.asarray(json.dumps(self.metadata)),
                            layer_ids=np.asarray(self.layer_ids), **arrays)
        self.sequence += 1
        self.records = []
