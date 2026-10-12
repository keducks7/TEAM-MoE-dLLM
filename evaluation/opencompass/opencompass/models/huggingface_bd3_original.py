# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional, Union

import json
import os

from einops import rearrange
import torch
import numpy as np
from torch.nn import functional as F
from mmengine.device import is_npu_available

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


## bd3 generation
import torch
from torch import block_diag
from torch.nn import functional as F
import numpy as np
from transformers.cache_utils import (
    Cache,
    DynamicCache,
)


def _install_vanilla_metric_hooks(model, collect_expert_outputs=False):
    """Observe MoE routing/outputs without changing model return values.

    Expert-output hooks are optional because they add norm/cosine reductions to
    each executed expert MLP.  The reductions stay on the layer device until a
    whole denoising forward has completed; full hidden states are never saved.
    """
    if hasattr(model, '_sdar_metric_hook_handles'):
        if (collect_expert_outputs
                and not hasattr(model, '_sdar_expert_output_hook_handles')):
            raise RuntimeError(
                'Vanilla metric hooks were installed without expert-output '
                'hooks. Reload the model with collect_expert_output_stats=True.')
        return

    handles = []
    expert_output_handles = []
    for module in model.modules():
        if module.__class__.__name__ != 'SDARMoeSparseMoeBlock':
            continue

        layer_index = len(handles)
        top_k_experts = int(module.top_k)
        norm_topk_prob = bool(getattr(module, 'norm_topk_prob', False))

        def capture_gate_output(_gate, _inputs, router_logits,
                                top_k=top_k_experts,
                                normalize=norm_topk_prob,
                                layer_id=layer_index):
            collector = getattr(model, '_sdar_metrics_collector', None)
            if collector is None:
                return
            k = min(top_k, router_logits.shape[-1])
            routing_weights = F.softmax(router_logits, dim=-1,
                                        dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, k=k, dim=-1)
            if normalize:
                routing_weights /= routing_weights.sum(
                    dim=-1, keepdim=True)
            active_experts = torch.zeros(
                router_logits.shape[-1],
                dtype=torch.bool,
                device=router_logits.device)
            active_experts.scatter_(0, selected_experts.reshape(-1), True)
            collector['_active_expert_counts'].append(active_experts.sum())
            if collector.get('_trace_enabled', False):
                collector['_layer_routes'].append((
                    layer_id,
                    selected_experts.detach(),
                    routing_weights.detach(),
                ))

            if (collector.get('_expert_output_enabled', False)
                    or collector.get('_expert_token_enabled', False)):
                mask_flat = collector.get('_current_mask_flat')
                if mask_flat is None:
                    raise RuntimeError(
                        'Expert-output collection requires the current token mask.')
                if mask_flat.numel() != selected_experts.shape[0]:
                    raise RuntimeError(
                        'Expert-output mask length does not match MoE gate tokens.')
                state_is_masked = mask_flat.reshape(-1).to(
                    device=selected_experts.device, dtype=torch.bool)
                collector.setdefault('_expert_route_context', {})[layer_id] = (
                    selected_experts.detach(),
                    routing_weights.detach(),
                    state_is_masked,
                )
                if collector.get('_expert_output_enabled', False):
                    shape = (2, int(router_logits.shape[-1]))
                    collector.setdefault('_expert_layer_output_stats', {})[
                        layer_id] = {
                            'count': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.int32),
                            'routing_weight_sum': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.float32),
                            'input_norm_sum': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.float32),
                            'output_norm_sum': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.float32),
                            'weighted_output_norm_sum': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.float32),
                            'output_input_cosine_sum': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.float32),
                            'output_input_norm_ratio_sum': torch.zeros(
                                shape, device=router_logits.device,
                                dtype=torch.float32),
                        }
                if collector.get('_expert_token_enabled', False):
                    token_shape = tuple(selected_experts.shape)
                    collector.setdefault('_expert_layer_token_stats', {})[
                        layer_id] = {
                            name: torch.full(
                                token_shape, torch.nan,
                                device=router_logits.device,
                                dtype=torch.float32)
                            for name in (
                                'input_norm', 'output_norm',
                                'weighted_output_norm',
                                'output_input_cosine',
                                'output_input_norm_ratio')
                        }

        handles.append(module.gate.register_forward_hook(capture_gate_output))

        if collect_expert_outputs:
            for expert_index, expert in enumerate(module.experts):
                def capture_expert_output(
                        _expert, expert_inputs, expert_output,
                        layer_id=layer_index, expert_id=expert_index):
                    collector = getattr(model, '_sdar_metrics_collector', None)
                    if (collector is None
                            or not (
                                collector.get('_expert_output_enabled', False)
                                or collector.get(
                                    '_expert_token_enabled', False))):
                        return
                    context = collector.get('_expert_route_context', {}).get(
                        layer_id)
                    layer_stats = collector.get(
                        '_expert_layer_output_stats', {}).get(layer_id)
                    layer_token_stats = collector.get(
                        '_expert_layer_token_stats', {}).get(layer_id)
                    if context is None or (layer_stats is None
                                           and layer_token_stats is None):
                        raise RuntimeError(
                            f'Missing route context for layer {layer_id}.')
                    if not torch.is_tensor(expert_output):
                        raise RuntimeError(
                            'Expert-output collection expects each expert MLP '
                            'to return one tensor.')
                    selected_experts, routing_weights, state_is_masked = context
                    token_indices, topk_slots = torch.where(
                        selected_experts == expert_id)
                    if expert_output.shape[0] != token_indices.numel():
                        raise RuntimeError(
                            f'Layer {layer_id} expert {expert_id} output rows '
                            'do not match its routed token count.')
                    if token_indices.numel() == 0:
                        return
                    expert_input = expert_inputs[0]
                    if expert_input.shape != expert_output.shape:
                        raise RuntimeError(
                            'Expert input/output shapes differ; norm comparison '
                            'would not be valid.')
                    weights = routing_weights[token_indices, topk_slots].float()
                    masked = state_is_masked[token_indices]
                    input_float = expert_input.float()
                    output_float = expert_output.float()
                    input_norm = torch.linalg.vector_norm(input_float, dim=-1)
                    output_norm = torch.linalg.vector_norm(output_float, dim=-1)
                    cosine = F.cosine_similarity(
                        input_float, output_float, dim=-1, eps=1e-8)
                    norm_ratio = output_norm / input_norm.clamp_min(1e-8)
                    weighted_output_norm = output_norm * weights
                    if layer_token_stats is not None:
                        layer_token_stats['input_norm'][
                            token_indices, topk_slots] = input_norm
                        layer_token_stats['output_norm'][
                            token_indices, topk_slots] = output_norm
                        layer_token_stats['weighted_output_norm'][
                            token_indices, topk_slots] = weighted_output_norm
                        layer_token_stats['output_input_cosine'][
                            token_indices, topk_slots] = cosine
                        layer_token_stats['output_input_norm_ratio'][
                            token_indices, topk_slots] = norm_ratio
                    if layer_stats is not None:
                        for state_id, selection in enumerate((masked, ~masked)):
                            count = selection.sum()
                            layer_stats['count'][state_id, expert_id] = count
                            layer_stats['routing_weight_sum'][
                                state_id, expert_id] = weights[selection].sum()
                            layer_stats['input_norm_sum'][
                                state_id, expert_id] = input_norm[selection].sum()
                            layer_stats['output_norm_sum'][
                                state_id, expert_id] = output_norm[selection].sum()
                            layer_stats['weighted_output_norm_sum'][
                                state_id, expert_id] = weighted_output_norm[
                                    selection].sum()
                            layer_stats['output_input_cosine_sum'][
                                state_id, expert_id] = cosine[selection].sum()
                            layer_stats['output_input_norm_ratio_sum'][
                                state_id, expert_id] = norm_ratio[selection].sum()

                expert_output_handles.append(
                    expert.register_forward_hook(capture_expert_output))

    if not handles:
        raise RuntimeError(
            'No SDARMoeSparseMoeBlock modules were found for Vanilla metric '
            'collection.')
    model._sdar_metric_hook_handles = handles
    if collect_expert_outputs:
        model._sdar_expert_output_hook_handles = expert_output_handles
    model._sdar_metrics_collector = None


def _layer_statistics_to_numpy(layer_stats, statistic_names,
                               output_dtype=None):
    """Stack layer-local tensors without moving sharded layers to one GPU."""
    ordered_layers = sorted(layer_stats)
    results = {}
    for statistic in statistic_names:
        device_groups = {}
        for position, layer_id in enumerate(ordered_layers):
            tensor = layer_stats[layer_id][statistic]
            device_groups.setdefault(tensor.device, []).append(
                (position, tensor))
        combined = None
        for tensors_on_device in device_groups.values():
            positions = [item[0] for item in tensors_on_device]
            stacked = torch.stack([
                item[1] for item in tensors_on_device
            ]).detach().to(device='cpu').numpy()
            if combined is None:
                combined = np.empty(
                    (len(ordered_layers),) + stacked.shape[1:],
                    dtype=stacked.dtype)
            combined[positions] = stacked
        if output_dtype is not None:
            combined = combined.astype(output_dtype, copy=False)
        results[statistic] = combined
    return ordered_layers, results


def _extract_causal_lm_logits(model_output, model):
    """Handle both ModelOutput and tuple returns from SDAR remote code."""
    if hasattr(model_output, 'logits'):
        return model_output.logits

    if isinstance(model_output, (tuple, list)):
        # TEAM-style remote code may wrap the CausalLM ModelOutput in a tuple.
        for item in model_output:
            if hasattr(item, 'logits'):
                return item.logits

        # Standard Hugging Face return_dict=False puts the logits tensor in
        # the output tuple.  Validate its vocabulary dimension so a base
        # model's hidden state cannot be mistaken for logits.
        vocab_size = int(model.config.vocab_size)
        for item in model_output:
            if (torch.is_tensor(item) and item.ndim >= 3
                    and item.shape[-1] == vocab_size):
                return item

    raise RuntimeError(
        'The loaded model did not return causal-LM logits.  Make sure '
        'AutoModelForCausalLM loaded SDARMoeForCausalLM rather than the base '
        f'model. Got output type: {type(model_output).__name__}.')

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def top_k_logits(logits, k):
    """ 保留概率最大的k个值，其他的设为-inf以屏蔽 """
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def top_p_logits(logits, p):
    """ nucleus sampling: 保留累积概率大于p的前几个token，其他设为-inf """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 获得大于p的第一个index并截断
    sorted_mask = cumulative_probs > p
    # 保证至少保留一个token
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits

def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    # 1️⃣ 原始概率：先应用温度，然后 softmax
    assert temperature > 0, "Temperature must be positive"
    logits = logits / temperature if temperature != 1.0 else logits
    ori_probs = F.softmax(logits, dim=-1)  # 用于置信度排序

    # 2️⃣ 再做 top-k / top-p 筛选用于采样
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    assert probs.dim() == 2

    # 3️⃣ multinomial 采样
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]

    # 4️⃣ 获取 token 在原始概率分布里的概率
    token_prob = torch.gather(ori_probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)

def sample_with_greedy(logits):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    probs = F.softmax(logits, dim=-1)  # 用于置信度排序
    assert probs.dim() == 2

    # greedy 采样
    token = torch.argmax(probs, dim=-1, keepdim=True)

    # 获取 token 在原始概率分布里的概率
    token_prob = torch.gather(probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)

def get_num_transfer_tokens(block_length, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    base = block_length // steps
    remainder = block_length % steps

    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1

    return num_transfer_tokens

@torch.no_grad()
def block_diffusion_generate(
    model,
    tokenizer,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8, 
    temperature=0.,
    top_k=0,
    top_p=1.0,
    remasking='low_confidence',
    threshold=1.0,
    stopping_criteria_idx=None,
    metrics_collector=None,
    dependency_observer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        denoising_steps: Sampling steps, less than or equal to block_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK]
    '''

    # 仅仅支持单个输入，不支持 batch inference
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    attention_mask = prompt['attention_mask']
    tokenizer = tokenizer
    past_key_values = DynamicCache()

    # prepare block_diag and position_ids
    num_blocks = (input_ids.shape[1] + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    # 生成块级下三角掩码（num_blocks × num_blocks）
    block_mask = torch.tril(
        torch.ones(num_blocks, num_blocks, device=model.device),
        diagonal=0
    )
    # 扩展到每个块内部（变成 total_length × total_length）
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0) \
                                .repeat_interleave(block_length, dim=1) \
                                .unsqueeze(0)  # 添加 batch 维度

    position_ids = torch.arange(
        0, total_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    # prepare input_ids
    x = torch.full((input_ids.shape[0], total_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()

    # calculate prefill_length
    prefill_blocks = input_ids.shape[1] // block_length
    prefill_length = prefill_blocks * block_length

    # prefill stage
    if prefill_length > 0:
        # 预填充阶段，计算得到 kv cache
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        _ = model(
            input_ids=cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values, # 这是一个全局变量
            use_cache=True,
            store_kv=True)

    # generate output recursively
    assert block_length % denoising_steps == 0 and block_length >= denoising_steps
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)
    end_generate = False
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block * block_length: (num_block + 1) * block_length]
        cur_attn_mask = block_diffusion_attention_mask[:, num_block *
            block_length: (num_block + 1) * block_length, :(num_block + 1) * block_length]
        cur_position_ids = position_ids[:, num_block * block_length: (num_block + 1) * block_length]
        for i in range(denoising_steps + 1): # 有一步需要专门用于缓存 kv cache，后续可以优化掉
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                _ = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True)
                break
            else:
                if dependency_observer is not None:
                    dependency_observer.begin_step(
                        num_block, i, cur_position_ids, cur_x, mask_index)
                if metrics_collector is not None:
                    metrics_collector['_active_expert_counts'] = []
                    if metrics_collector.get('_trace_enabled', False):
                        metrics_collector['_layer_routes'] = []
                        trace_token_ids = cur_x.detach().clone()
                        trace_mask_before = mask_index.detach().clone()
                    if (metrics_collector.get('_expert_output_enabled', False)
                            or metrics_collector.get(
                                '_expert_token_enabled', False)):
                        metrics_collector['_expert_route_context'] = {}
                        if metrics_collector.get(
                                '_expert_output_enabled', False):
                            metrics_collector[
                                '_expert_layer_output_stats'] = {}
                        if metrics_collector.get(
                                '_expert_token_enabled', False):
                            metrics_collector[
                                '_expert_layer_token_stats'] = {}
                        metrics_collector['_current_mask_flat'] = (
                            mask_index.detach().reshape(-1))
                    model._sdar_metrics_collector = metrics_collector
                try:
                    model_output = model(
                        input_ids=cur_x,
                        attention_mask=cur_attn_mask,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=False)
                finally:
                    if dependency_observer is not None:
                        dependency_observer.end_forward()
                    if metrics_collector is not None:
                        model._sdar_metrics_collector = None
                logits = _extract_causal_lm_logits(model_output, model)

                if metrics_collector is not None:
                    active_counts = metrics_collector.pop(
                        '_active_expert_counts')
                    if not active_counts:
                        raise RuntimeError(
                            'No MoE gate outputs were captured during a '
                            'denoising forward.')
                    # Layers can reside on different GPUs under device_map.
                    metrics_collector['activated_experts'] += sum(
                        int(count.item()) for count in active_counts)
                    metrics_collector['layer_forwards'] += len(active_counts)
                    metrics_collector['denoising_forwards'] += 1
                    token_output_stats = None
                    if metrics_collector.get('_expert_output_enabled', False):
                        layer_stats = metrics_collector.pop(
                            '_expert_layer_output_stats')
                        if len(layer_stats) != len(active_counts):
                            raise RuntimeError(
                                'Expert-output layer count does not match the '
                                'APF layer count.')
                        _, step_stats = _layer_statistics_to_numpy(
                            layer_stats, (
                                'count', 'routing_weight_sum',
                                'input_norm_sum', 'output_norm_sum',
                                'weighted_output_norm_sum',
                                'output_input_cosine_sum',
                                'output_input_norm_ratio_sum'))
                        by_step = metrics_collector[
                            '_expert_output_by_step']
                        if i not in by_step:
                            by_step[i] = step_stats
                        else:
                            for statistic, values in step_stats.items():
                                by_step[i][statistic] += values
                        metrics_collector['_expert_output_forward_counts'][
                            i] = metrics_collector[
                                '_expert_output_forward_counts'].get(i, 0) + 1
                        state_counts = np.asarray([
                            int(mask_index.sum().item()),
                            int((~mask_index).sum().item()),
                        ], dtype=np.int64)
                        if i not in metrics_collector[
                                '_expert_output_state_token_counts']:
                            metrics_collector[
                                '_expert_output_state_token_counts'][i] = (
                                    state_counts)
                        else:
                            metrics_collector[
                                '_expert_output_state_token_counts'][i] += (
                                    state_counts)
                        metrics_collector[
                            '_expert_output_pair_opportunities'][i] = (
                                metrics_collector[
                                    '_expert_output_pair_opportunities'].get(
                                        i, 0)
                                + int(state_counts[0] * state_counts[1]))
                    if metrics_collector.get('_expert_token_enabled', False):
                        layer_token_stats = metrics_collector.pop(
                            '_expert_layer_token_stats')
                        if len(layer_token_stats) != len(active_counts):
                            raise RuntimeError(
                                'Expert-token layer count does not match the '
                                'APF layer count.')
                        _, token_output_stats = _layer_statistics_to_numpy(
                            layer_token_stats, (
                                'input_norm', 'output_norm',
                                'weighted_output_norm',
                                'output_input_cosine',
                                'output_input_norm_ratio'),
                            output_dtype=np.float16)
                        for statistic, values in token_output_stats.items():
                            if not np.isfinite(values).all():
                                raise RuntimeError(
                                    f'Expert-token statistic {statistic} was '
                                    'not filled for every routed assignment.')
                    if (metrics_collector.get('_expert_output_enabled', False)
                            or metrics_collector.get(
                                '_expert_token_enabled', False)):
                        metrics_collector.pop('_expert_route_context', None)
                        metrics_collector.pop('_current_mask_flat', None)
                    if metrics_collector.get('_trace_enabled', False):
                        layer_routes = sorted(
                            metrics_collector.pop('_layer_routes'),
                            key=lambda item: item[0])
                        if len(layer_routes) != len(active_counts):
                            raise RuntimeError(
                                'Route trace layer count does not match the '
                                'APF layer count.')
                        if cur_x.shape[0] != 1:
                            raise RuntimeError(
                                'Vanilla route tracing currently requires '
                                'batch_size=1.')

                        expert_ids = []
                        routing_weights = []
                        layer_ids = []
                        for layer_id, selected, weights in layer_routes:
                            expected_tokens = cur_x.numel()
                            if selected.shape[0] != expected_tokens:
                                raise RuntimeError(
                                    'MoE gate token count does not match the '
                                    'current decoding block.')
                            expert_ids.append(
                                selected.reshape(
                                    cur_x.shape[0], cur_x.shape[1], -1)[0]
                                .to(device='cpu', dtype=torch.int16).numpy())
                            routing_weights.append(
                                weights.reshape(
                                    cur_x.shape[0], cur_x.shape[1], -1)[0]
                                .to(device='cpu', dtype=torch.float16).numpy())
                            layer_ids.append(layer_id)

                        block_start = num_block * block_length
                        global_positions = np.arange(
                            block_start, block_start + cur_x.shape[1],
                            dtype=np.int32)
                        prompt_before = global_positions < prompt_length
                        state_before = np.where(
                            trace_mask_before[0].to(device='cpu').numpy(),
                            2,
                            np.where(prompt_before, 0, 1)).astype(np.int8)
                        pending_trace = {
                            'block_id': num_block,
                            'block_start': block_start,
                            'denoising_step': i,
                            'layer_ids': np.asarray(layer_ids, dtype=np.int16),
                            'global_positions': global_positions,
                            'token_ids_before': trace_token_ids[0].to(
                                device='cpu', dtype=torch.int32).numpy(),
                            'mask_before': trace_mask_before[0].to(
                                device='cpu').numpy(),
                            'state_before': state_before,
                            'expert_ids': np.stack(expert_ids, axis=0),
                            'routing_weights': np.stack(
                                routing_weights, axis=0),
                        }
                        if token_output_stats is not None:
                            pending_trace.update({
                                f'expert_{name}': values
                                for name, values in token_output_stats.items()
                            })
                        metrics_collector['_pending_trace'] = pending_trace

                logits_with_noise = add_gumbel_noise(
                    logits, temperature=temperature)
                # x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    # p = F.softmax(logits, dim=-1)
                    # x0_p = torch.squeeze(
                    #     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                    # x0, x0_p = sample_with_temperature_topk_topp(
                    #     logits,
                    #     temperature=temperature,
                    #     top_k=top_k,
                    #     top_p=top_p
                    # )
                    x0, x0_p = sample_with_greedy(logits)
                elif remasking == 'random':
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0 = torch.where(mask_index, x0, cur_x)
                if threshold < 1.0:
                    # 动态解码过程
                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

                    for j in range(confidence.shape[0]):
                        # 找到满足置信度阈值的token
                        high_conf_mask = confidence[j] > threshold
                        num_high_confidence = high_conf_mask.sum()

                        if num_high_confidence >= num_transfer_tokens[i]:
                            # 如果高置信度数量已经超过或等于num_transfer_tokens，直接使用这些token
                            transfer_index[j] = high_conf_mask
                        else:
                            # 当高置信度tokens数量不够时，直接简单地从整体confidence中选取topk即可
                            _, idx = torch.topk(confidence[j], num_transfer_tokens[i])
                            transfer_index[j, idx] = True
                else:
                    # 静态解码过程
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    transfer_index = torch.zeros_like(
                        x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(
                            confidence[j], k=num_transfer_tokens[i])
                        transfer_index[j, select_index] = True

                if dependency_observer is not None:
                    dependency_observer.finish_step(x0, x0_p, transfer_index)
                cur_x[transfer_index] = x0[transfer_index]
                if metrics_collector is not None:
                    metrics_collector['accepted_tokens'] += int(
                        transfer_index.sum().item())
                    if metrics_collector.get('_trace_enabled', False):
                        pending_trace = metrics_collector.pop('_pending_trace')
                        pending_trace['accepted_after'] = transfer_index[0].to(
                            device='cpu').numpy()
                        metrics_collector['trace_forwards'].append(
                            pending_trace)

        x[:, num_block * block_length: (num_block + 1) * block_length] = cur_x
        for stop_idx in stopping_criteria_idx:
            if stop_idx in x[:, prompt_length:]:
                end_generate = True
                break
        if end_generate:
            break

    return x


def forward_add_noise(
    input_ids: torch.Tensor, #(b, l)
    eps: float,
    mask_token: int,
    ignore_index: int = -100
):
    bsz, length = input_ids.shape
    t = torch.rand(bsz, device=input_ids.device)  # U[0,1]
    p_mask = (1 - eps) * t + eps  # avoid devision by zero
    p_mask = p_mask[:, None].repeat(1, length)
    # True is masked
    masked_indices = torch.rand(
        (bsz, length), device=input_ids.device) < p_mask
    # `noisy_inputs.shape` is same as `input_ids`, some tokens are replaced by `mask_token`
    noisy_inputs = torch.where(masked_indices, mask_token, input_ids)
    # 需要被预测的 labels, 对应的地方变为 -100
    masked_labels = torch.where(masked_indices, input_ids, ignore_index)
    return noisy_inputs, masked_labels, p_mask  # minibsz, length

def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(self.max_sequence_id_len, len(sequence_ids))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(s in lookback_tokens_batch[i] for s in self.stop_words)
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])

def _get_possible_max_seq_len(max_seq_len, path):
    if max_seq_len is not None:
        return max_seq_len

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    possible_keys = [
        'max_position_embeddings',
        'seq_length',
        'model_max_length',
        'max_sequence_length'
    ]
    for k in possible_keys:
        if hasattr(config, k):
            return getattr(config, k)
    raise ValueError('max_seq_len is not provided and cannot be inferred from the model config.')


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def _format_with_fast_chat_template(inputs: List[str], name: str='vicuna'):
    try:
        from fastchat.model import get_conversation_template
    except ImportError:
        raise ModuleNotFoundError('fastchat not found. Please install with\npip install "fschat[model_worker,webui]"')

    outputs = []
    for _input in inputs:
        template = get_conversation_template(name)
        for item in _input:
            if item['role'] == 'user':
                template.append_message(template.roles[0], item['content'])
            elif item['role'] == 'assistant':
                template.append_message(template.roles[1], item['content'])
            elif item['role'] == 'system':
                continue
            else:
                raise ValueError(f"Unknown role {item['role']}")
        template.append_message(template.roles[1], None)
        outputs.append(template.get_prompt())
    return outputs


def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]
    )
    return APITemplateParser(meta_template or default_meta_template)


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs


@MODELS.register_module()
class BD3withChatTemplate_original(BaseModel):
    """Model wrapper for bd3 models designed for chat.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 pad_token_id: Optional[int] = None,
                 fastchat_template: Optional[str] = None,
                 stop_words: Optional[str] = [],
                 mode: str = 'none',
                 collect_metrics: bool = False,
                 metrics_output_dir: Optional[str] = None,
                 collect_routing_trace: bool = False,
                 routing_trace_output_dir: Optional[str] = None,
                 collect_expert_output_stats: bool = False,
                 expert_output_stats_dir: Optional[str] = None,
                 collect_expert_token_stats: bool = False,
                 collect_dependency_trace: bool = False,
                 dependency_trace_output_dir: Optional[str] = None,
                 dependency_layers: str = 'auto',
                 dependency_max_blocks: int = 8,
                 **other_kwargs):

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        self.collect_routing_trace = collect_routing_trace
        self.routing_trace_output_dir = routing_trace_output_dir
        self.collect_expert_output_stats = collect_expert_output_stats
        self.expert_output_stats_dir = expert_output_stats_dir
        self.collect_expert_token_stats = collect_expert_token_stats
        self.dependency_observer = None
        if collect_dependency_trace:
            if tokenizer_only or not dependency_trace_output_dir:
                raise ValueError('Dependency tracing requires a model and output directory')
            from .sdar_dependency_observer import DependencyObserver
            self.dependency_observer = DependencyObserver(
                self.model, dependency_trace_output_dir,
                dependency_layers, dependency_max_blocks)
        self.collect_metrics = (
            collect_metrics or collect_routing_trace
            or collect_expert_output_stats or collect_expert_token_stats)
        self.metrics_output_dir = metrics_output_dir
        self._metrics_sequence = 0
        self._metrics_totals = {
            'sequences': 0,
            'activated_experts': 0,
            'layer_forwards': 0,
            'accepted_tokens': 0,
            'denoising_forwards': 0,
        }
        if self.collect_metrics:
            if not self.metrics_output_dir:
                raise ValueError(
                    'metrics_output_dir is required when collect_metrics=True')
            os.makedirs(self.metrics_output_dir, exist_ok=True)
            _install_vanilla_metric_hooks(
                self.model,
                collect_expert_outputs=(
                    self.collect_expert_output_stats
                    or self.collect_expert_token_stats))
        if self.collect_expert_token_stats and not self.collect_routing_trace:
            raise ValueError(
                'collect_expert_token_stats=True requires '
                'collect_routing_trace=True so token responses can be aligned '
                'with token IDs and expert IDs.')
        if self.collect_routing_trace:
            if not self.routing_trace_output_dir:
                raise ValueError(
                    'routing_trace_output_dir is required when '
                    'collect_routing_trace=True')
            os.makedirs(self.routing_trace_output_dir, exist_ok=True)
        if self.collect_expert_output_stats:
            if not self.expert_output_stats_dir:
                raise ValueError(
                    'expert_output_stats_dir is required when '
                    'collect_expert_output_stats=True')
            os.makedirs(self.expert_output_stats_dir, exist_ok=True)
        assert mode in ['none', 'mid']
        self.mode = mode
        self.logger.info(f'using stop words: {self.stop_words}')

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

    def get_metrics_summary(self):
        """Return micro-averaged metrics accumulated by this process."""
        if not self.collect_metrics:
            return None
        totals = self._metrics_totals.copy()
        layer_forwards = totals['layer_forwards']
        denoising_forwards = totals['denoising_forwards']
        apf = (totals['activated_experts'] / layer_forwards
               if layer_forwards else None)
        tpf = (totals['accepted_tokens'] / denoising_forwards
               if denoising_forwards else None)
        totals.update({
            'method': 'vanilla',
            'APF': apf,
            'TPF': tpf,
            'APT': apf / tpf if apf is not None and tpf else None,
        })
        return totals

    def _load_tokenizer(self, path: Optional[str], kwargs: dict, pad_token_id: Optional[int] = None):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if pad_token_id is not None:
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != pad_token_id:
                self.logger.warning(f'pad_token_id is not consistent. Using {pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = pad_token_id
            return
        if self.tokenizer.pad_token_id is not None:
            return
        self.logger.warning('pad_token_id is not set for the tokenizer.')
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='cuda', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        # This wrapper requires vocabulary logits.  Falling back silently to
        # AutoModel would return hidden states and produce invalid decoding.
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        print(f"model configs:\n{self.model.config}")
        print(f"model 参数类型: {next(self.model.parameters()).dtype}")
        self.model.eval()
        self.model.generation_config.do_sample = False

    def get_ppl_tokenwise(self, inputs: List[str], label: List[List[int]], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get inference-ppl per token given a list of inputs and label.

        Args:
            inputs (List[str]): A list of strings.
            label (List[List[int]]): A list of list of label, each label is a tuple of (start, end, 1)
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'

        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            import numpy as np
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        loss = loss.cpu().numpy()

        decode_messages = [[self.tokenizer.decode([input_id]) for input_id in token] for token in tokens['input_ids']]
        char_messages = [[ch for ch in message] for message in messages]

        # shifted to align label and loss
        for i in range(len(decode_messages)):
            decode_messages[i] = decode_messages[i][1:]

        aggregated_label_list = [[] for _ in range(len(decode_messages))]

        tag_list = [[] for _ in range(len(decode_messages))]

        for tmp_index, label_list in enumerate(label):
            for single_label in label_list:
                left = single_label[0]
                right = single_label[1]
                for i in range(left, right):
                    aggregated_label_list[tmp_index].append(i)


        def align_sequences(seq1, seq2, sep_len):
            """
            seq1: decoded sequence from token, one token may contain multiple characters
            seq2: original separate character sequence
            """
            i, j = 0, 0
            matched_pairs = []
            while i < len(seq1) and j < len(seq2):
                word = seq1[i]
                if len(word) == 0:
                    matched_pairs.append((word, []))
                    i += 1
                    continue

                if '\ufffd' in word:
                    for _ in range(sep_len):
                        matched_pairs.append((word, [j]))
                        i += 1
                    j += 1
                    continue

                char_sequence = ''
                while j < len(seq2) and (char_sequence != word):
                    char_sequence += seq2[j]
                    if char_sequence == word:
                        matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                        j += 1
                        break
                    elif len(char_sequence) > len(word):
                        if word == char_sequence[-len(word):]:
                            matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                            j += 1
                            break
                        else:
                            j += 1
                    else:
                        j += 1
                i += 1

            return matched_pairs


        if 'qwen' in self.path or 'Qwen' in self.path:
            sep_len = 2
        elif 'Llama-3' in self.path:
            sep_len = 2
        elif 'Yi' in self.path:
            sep_len = 3
        elif 'Llama-2' in self.path:
            sep_len = 3
        elif 'deepseek' in self.path:
            sep_len = 2
        else:
            sep_len = 3


        matched_pairs_list = [align_sequences(decode_messages[i], char_messages[i], sep_len) for i in range(len(decode_messages))]
        for match_index, matched_pairs in enumerate(matched_pairs_list):
            for i, (word, indices) in enumerate(matched_pairs):
                for j in indices:
                    if j in aggregated_label_list[match_index]:
                        tag_list[match_index].append(i)
                        break

        inference_loss_list = []
        token_len_list = []
        for i in range(len(loss)):
            inference_loss = 0
            token_len = 0
            for j in range(len(loss[i])):
                if j in tag_list[i]:

                    inference_loss += loss[i][j]
                    print(loss[i][j])
                    token_len += 1
            inference_loss_list.append(inference_loss)
            token_len_list.append(token_len)

        return inference_loss_list, token_len_list

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                #  diffusion_kwargs: dict = dict(),
                 **kwargs) -> List[str]:
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )

        if self.fastchat_template:
            print("go into fastchat_template")
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            tokenize_kwargs['add_special_tokens'] = False
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        formatted_prompt_text = messages[0] if batch_size == 1 else None

        # list of tensors
        # inputs_ids = [torch.tensor(t, device=self.model.device) for t in tokens['input_ids']]

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        if self.mode == 'mid':
            # Reserve space for the tokens to be generated in the future.
            max_prompt_len = self.max_seq_len - max_out_len

            # Retain the first 0.5 * max_prompt_len tokens and the last 0.5 * max_prompt_len tokens, discarding the middle ones,
            # because the prompts' questions are usually at the beginning or the end.
            # To avoid the warning:
            # This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length.
            # Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
            half_max_prompt_len = max_prompt_len // 2
            if half_max_prompt_len > 0 and tokens['input_ids'].shape[1] > max_prompt_len:
                for key in tokens.keys():
                    if tokens[key].shape[1] > max_prompt_len:
                        field_values = tokens[key]
                        tokens[key] = torch.cat(
                            (field_values[:, :half_max_prompt_len], field_values[:, -half_max_prompt_len:]), dim=1
                        )

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        # generation_kwargs.update(diffusion_kwargs)
        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(stopping_criteria, self.tokenizer, batch_size)
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        stopping_criteria_idx = [self.tokenizer.encode(stop, add_special_tokens=False)[0] for stop in stopping_criteria]
        metrics_collector = None
        if self.collect_metrics:
            metrics_collector = {
                'activated_experts': 0,
                'layer_forwards': 0,
                'accepted_tokens': 0,
                'denoising_forwards': 0,
            }
            if self.collect_routing_trace:
                metrics_collector['_trace_enabled'] = True
                metrics_collector['trace_forwards'] = []
            if self.collect_expert_output_stats:
                metrics_collector['_expert_output_enabled'] = True
                metrics_collector['_expert_output_by_step'] = {}
                metrics_collector['_expert_output_forward_counts'] = {}
                metrics_collector['_expert_output_state_token_counts'] = {}
                metrics_collector['_expert_output_pair_opportunities'] = {}
            if self.collect_expert_token_stats:
                metrics_collector['_expert_token_enabled'] = True

        if self.dependency_observer is not None:
            if generation_kwargs['remasking'] != 'low_confidence':
                raise ValueError('Dependency pilot requires low_confidence remasking')
            self.dependency_observer.start_sequence(
                tokens['input_ids'], self.path, generation_kwargs['threshold'],
                {key: generation_kwargs[key] for key in (
                    'gen_length', 'block_length', 'denoising_steps',
                    'temperature', 'top_k', 'top_p', 'remasking', 'mask_id')})
        full_outputs = block_diffusion_generate(
            self.model,
            self.tokenizer,
            tokens,
            mask_id=generation_kwargs['mask_id'],
            denoising_steps=generation_kwargs['denoising_steps'],
            gen_length=generation_kwargs['gen_length'],
            block_length=generation_kwargs['block_length'],
            temperature=generation_kwargs['temperature'],
            top_k=generation_kwargs['top_k'],
            top_p=generation_kwargs['top_p'],
            # cfg_scale=generation_kwargs['cfg_scale'],
            remasking=generation_kwargs['remasking'],
            threshold=generation_kwargs['threshold'],
            stopping_criteria_idx=stopping_criteria_idx,
            metrics_collector=metrics_collector,
            dependency_observer=self.dependency_observer,
            )
        if self.dependency_observer is not None:
            self.dependency_observer.finish_sequence()
        prompt_length = tokens["input_ids"].shape[1]
        outputs = full_outputs[:, prompt_length:]

        if self.collect_metrics:
            trace_forwards = metrics_collector.pop('trace_forwards', None)
            metrics_collector.pop('_trace_enabled', None)
            expert_output_by_step = metrics_collector.pop(
                '_expert_output_by_step', None)
            expert_output_forward_counts = metrics_collector.pop(
                '_expert_output_forward_counts', None)
            expert_output_state_token_counts = metrics_collector.pop(
                '_expert_output_state_token_counts', None)
            expert_output_pair_opportunities = metrics_collector.pop(
                '_expert_output_pair_opportunities', None)
            metrics_collector.pop('_expert_output_enabled', None)
            metrics_collector.pop('_expert_token_enabled', None)
            layer_forwards = metrics_collector['layer_forwards']
            denoising_forwards = metrics_collector['denoising_forwards']
            accepted_tokens = metrics_collector['accepted_tokens']
            apf = (metrics_collector['activated_experts'] / layer_forwards
                   if layer_forwards else None)
            tpf = (accepted_tokens / denoising_forwards
                   if denoising_forwards else None)
            record = {
                'schema': 'sdar_metrics_v1',
                'method': 'vanilla',
                'sequence': self._metrics_sequence,
                **metrics_collector,
                'APF': apf,
                'TPF': tpf,
                'APT': apf / tpf if apf is not None and tpf else None,
            }
            self._metrics_totals['sequences'] += 1
            for name in (
                    'activated_experts', 'layer_forwards',
                    'accepted_tokens', 'denoising_forwards'):
                self._metrics_totals[name] += int(metrics_collector[name])
            metrics_file = os.path.join(
                self.metrics_output_dir,
                f'vanilla_metrics_pid{os.getpid()}.jsonl')
            with open(metrics_file, 'a', encoding='utf-8') as file:
                file.write(json.dumps(record) + '\n')

            if self.collect_routing_trace:
                if not trace_forwards:
                    raise RuntimeError(
                        'Routing trace collection produced no forward records.')
                trace_file = os.path.join(
                    self.routing_trace_output_dir,
                    f'routing_trace_pid{os.getpid()}_sequence'
                    f'{self._metrics_sequence:06d}.npz')
                trace_payload = dict(
                    schema=np.asarray('sdar_vanilla_routing_trace_v2'),
                    sequence=np.asarray(self._metrics_sequence, dtype=np.int32),
                    prompt_length=np.asarray(prompt_length, dtype=np.int32),
                    block_length=np.asarray(
                        generation_kwargs['block_length'], dtype=np.int16),
                    state_names=np.asarray([
                        'prompt', 'generated_decoded', 'masked']),
                    formatted_prompt_text=np.asarray(
                        formatted_prompt_text or ''),
                    final_token_ids=full_outputs[0].detach().to(
                        device='cpu', dtype=torch.int32).numpy(),
                    final_generated_text=np.asarray(self.tokenizer.decode(
                        full_outputs[0, prompt_length:].detach().to(
                            device='cpu').tolist(),
                        skip_special_tokens=False)),
                    block_ids=np.asarray(
                        [item['block_id'] for item in trace_forwards],
                        dtype=np.int32),
                    block_starts=np.asarray(
                        [item['block_start'] for item in trace_forwards],
                        dtype=np.int32),
                    denoising_steps=np.asarray(
                        [item['denoising_step'] for item in trace_forwards],
                        dtype=np.int16),
                    layer_ids=trace_forwards[0]['layer_ids'],
                    global_positions=np.stack([
                        item['global_positions'] for item in trace_forwards]),
                    token_ids_before=np.stack(
                        [item['token_ids_before'] for item in trace_forwards]),
                    mask_before=np.stack(
                        [item['mask_before'] for item in trace_forwards]),
                    state_before=np.stack(
                        [item['state_before'] for item in trace_forwards]),
                    accepted_after=np.stack(
                        [item['accepted_after'] for item in trace_forwards]),
                    expert_ids=np.stack(
                        [item['expert_ids'] for item in trace_forwards]),
                    routing_weights=np.stack(
                        [item['routing_weights'] for item in trace_forwards]),
                )
                if self.collect_expert_token_stats:
                    for statistic in (
                            'input_norm', 'output_norm',
                            'weighted_output_norm', 'output_input_cosine',
                            'output_input_norm_ratio'):
                        trace_payload[f'expert_{statistic}'] = np.stack([
                            item[f'expert_{statistic}']
                            for item in trace_forwards
                        ])
                np.savez_compressed(
                    trace_file,
                    **trace_payload,
                )
            if self.collect_expert_output_stats:
                if not expert_output_by_step:
                    raise RuntimeError(
                        'Expert-output collection produced no forward records.')
                steps = sorted(expert_output_by_step)
                expert_output_file = os.path.join(
                    self.expert_output_stats_dir,
                    f'expert_output_stats_pid{os.getpid()}_sequence'
                    f'{self._metrics_sequence:06d}.npz')
                statistic_names = (
                    'count', 'routing_weight_sum', 'input_norm_sum',
                    'output_norm_sum', 'weighted_output_norm_sum',
                    'output_input_cosine_sum',
                    'output_input_norm_ratio_sum')
                arrays = {
                    statistic: np.stack([
                        expert_output_by_step[step][statistic]
                        for step in steps
                    ])
                    for statistic in statistic_names
                }
                np.savez_compressed(
                    expert_output_file,
                    schema=np.asarray('sdar_vanilla_expert_output_stats_v1'),
                    sequence=np.asarray(
                        self._metrics_sequence, dtype=np.int32),
                    denoising_steps=np.asarray(steps, dtype=np.int16),
                    layer_ids=np.arange(
                        arrays['count'].shape[1], dtype=np.int16),
                    state_names=np.asarray(['masked', 'decoded']),
                    forward_counts=np.asarray([
                        expert_output_forward_counts[step]
                        for step in steps
                    ], dtype=np.int32),
                    state_token_counts=np.stack([
                        expert_output_state_token_counts[step]
                        for step in steps
                    ]),
                    masked_decoded_pair_opportunities=np.asarray([
                        expert_output_pair_opportunities[step]
                        for step in steps
                    ], dtype=np.int64),
                    **arrays,
                )
            self._metrics_sequence += 1

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        decodeds = [t.replace('<|MASK|>', '') for t in decodeds]
        for stop in stopping_criteria:
            decodeds = [t.split(stop)[0] for t in decodeds]
        return decodeds

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])

    def generate_from_template(self, templates: List[PromptType],
                               max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates, mode='gen')
        if hasattr(self, 'sync_rank') and self.sync_rank:
            inputs = self.sync_inputs(inputs)
        return self.generate(inputs, max_out_len=max_out_len, **kwargs)

def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs
