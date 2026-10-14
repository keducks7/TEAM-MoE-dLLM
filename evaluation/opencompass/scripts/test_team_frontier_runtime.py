"""CPU checks of real source: four-branch DCD, native counters, timer and TPF."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT.parents[1] / 'modeling_sdar_moe.py'
WRAPPER = ROOT / 'opencompass/models/huggingface_bd3_decoded_jump_expert_limit_speculative.py'


class ToyExpert(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.proj(x)


scope = dict(torch=torch, nn=nn, F=F, np=np, time=time, DynamicCache=list, SDARMoeMLP=ToyExpert)
tree = ast.parse(MODEL_SOURCE.read_text())
nodes = [n for n in tree.body if getattr(n, 'name', '') in
         {'_select_team_expert_pool', 'SDARMoeSparseMoeBlock'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL_SOURCE), 'exec'), scope)
names = {'_install_team_metric_hooks', '_install_team_native_metrics', '_team_cuda_devices', '_synchronize_model_cuda',
         '_run_team_generation', '_record_team_acceptance', '_finalize_team_collector', '_team_denoising_forward',
         'add_gumbel_noise', 'top_k_logits', 'top_p_logits', 'sample_with_temperature_topk_topp',
         'get_num_transfer_tokens', '_pack_team_positions', 'block_diffusion_generate'}
nodes = [n for n in ast.parse(WRAPPER.read_text()).body if getattr(n, 'name', '') in names]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WRAPPER), 'exec'), scope)


def new_moe(experts=16, topk=8):
    return scope['SDARMoeSparseMoeBlock'](SimpleNamespace(num_experts=experts,
        num_experts_per_tok=topk, norm_topk_prob=True, hidden_size=4, moe_intermediate_size=8))


def collector(backend='native', experts=True):
    return dict(activated_experts=0, layer_forwards=0, accepted_tokens=0, denoising_forwards=0,
        transfer_accepted_tokens=0, mask_transition_tokens=0, accepted_token_mode='mask_transitions',
        collect_experts=experts, metrics_backend=backend, pool_layer_forwards=0,
        pool_actual_budget_sum=0, pool_budget_relaxations=0, pool_pending_expert_sum=0)


class ToyModel(nn.Module):
    """Execute real MoE/cache layout; controlled logits exercise SEH verification."""
    def __init__(self):
        super().__init__()
        self.mlp = new_moe()
        self.spec_calls = 0

    @property
    def device(self):
        return self.mlp.gate.weight.device

    def forward(self, input_ids, stored_hidden_states=None, choice_id=0, **kwargs):
        n = input_ids.shape[1]
        pos = torch.arange(n, device=self.device).remainder(32).float()
        x = torch.stack([input_ids[0].remainder(3).float(), pos / 32, torch.ones(n), -pos / 32], -1)[None]
        past = None
        if stored_hidden_states is not None:
            past = stored_hidden_states[0][:, choice_id * 32:(choice_id + 1) * 32].repeat(1, 4, 1)
            self.spec_calls += 1
        output, _ = self.mlp(x, past,
            decoded_index=kwargs.get('decoded_index'), expert_limit_index=kwargs.get('expert_limit_index'),
            hot_mask=kwargs.get('hot_mask'), team_mask_index=kwargs.get('team_mask_index'),
            team_pool_strategy=kwargs.get('team_pool_strategy', 'none'),
            team_pool_budget=kwargs.get('team_pool_budget', 36),
            team_pool_coverage_target=kwargs.get('team_pool_coverage_target', .9),
            team_collect_expert_stats=kwargs.get('team_collect_expert_stats', False))
        confidence = .7 + .2 * pos / 31
        logits = torch.zeros(1, n, 3)
        logits[0, :, 1] = torch.log(2 * confidence / (1 - confidence))
        return SimpleNamespace(logits=logits), [output]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_disabled_and_native_stats_preserve_output_and_rng(self):
        moe = new_moe()
        x, past = torch.randn(1, 8, 4), torch.randn(1, 8, 4)
        decoded = torch.tensor([[True, True, False, False, False, False, False, False]])
        cold = torch.tensor([[False, False, False, False, True, True, True, True]])
        a = moe(x, past, decoded, cold)[0]
        rng = torch.random.get_rng_state().clone()
        b = moe(x, past, decoded, cold, team_collect_expert_stats=True)[0]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(b[decoded], past[decoded]))

    def test_pending_dcd_experts_force_explicit_equal_capacity(self):
        ids = torch.tensor([[0, 1], [2, 3], [4, 0]])
        mass = torch.tensor([[.8, .2, 0, 0, 0], [0, 0, .6, .4, 0], [.4, 0, 0, 0, .6]])
        hot, pending = torch.tensor([False, False, True]), torch.tensor([True, True, False])
        capacities = []
        for strategy in ['reference', 'protected']:
            pool, info = scope['_select_team_expert_pool'](ids, mass, hot, pending, 2, strategy, .9)
            self.assertTrue(pool[:4].all())
            self.assertEqual(info['budget_relaxed'], 1)
            capacities.append(info['actual_budget'])
        self.assertEqual(capacities, [5, 5])

    def test_reference_at_large_capacity_reproduces_original_lac(self):
        moe = new_moe()
        x, past = torch.randn(1, 128, 4), torch.randn(1, 128, 4)
        decoded = torch.zeros(1, 128, dtype=torch.bool); decoded[:, ::4] = True
        masked = ~decoded
        hot = masked.clone(); hot[:, 2::4] = False
        cold = masked & ~hot
        baseline = moe(x, past, decoded, cold)[0]
        reference = moe(x, past, decoded, cold, hot_mask=hot,
            team_mask_index=masked, team_pool_strategy='reference', team_pool_budget=128)[0]
        self.assertTrue(torch.equal(baseline, reference))
        self.assertEqual(moe._team_last_pool_info['actual_budget'], moe._team_last_pool_info['team_reference_union'])

    def test_protected_hot_top1_and_cached_decoded_positions(self):
        moe = new_moe()
        x, past = torch.randn(1, 128, 4), torch.randn(1, 128, 4)
        decoded = torch.zeros(1, 128, dtype=torch.bool); decoded[:, ::4] = True
        masked = ~decoded.clone(); masked[:, 1::8] = False
        hot = masked.clone(); hot[:, 2::4] = False
        cold = masked & ~hot
        a = moe(x, past, decoded, cold)[0]
        b = moe(x, past, decoded, cold, hot_mask=hot, team_mask_index=masked,
            team_pool_strategy='protected', team_pool_budget=8, team_collect_expert_stats=True)[0]
        self.assertTrue(torch.equal(b[decoded], past[decoded]))
        self.assertTrue(torch.isfinite(b).all())
        self.assertGreaterEqual(moe._team_last_pool_info['actual_budget'], 8)
        self.assertLessEqual(moe._team_last_active_experts, moe._team_last_pool_info['actual_budget'])
        self.assertEqual(a.shape, b.shape)

    def test_real_generation_seh_native_and_hook_counts_match(self):
        model = ToyModel().eval()
        scope['_install_team_native_metrics'](model)
        scope['_install_team_metric_hooks'](model)
        prompt = dict(input_ids=torch.full((1, 32), 2), attention_mask=torch.ones(1, 32))
        args = dict(tokenizer=None, prompt=prompt, mask_id=151669, gen_length=32,
            block_length=32, denoising_steps=32, temperature=1., top_k=1, top_p=1.,
            threshold=.95, stopping_criteria_idx=[], team_pool_strategy='protected', team_pool_budget=8)
        native = collector()
        torch.manual_seed(12)
        a = scope['block_diffusion_generate'](model=model, metrics_collector=native, **args)
        hook = collector('hook')
        torch.manual_seed(12)
        b = scope['block_diffusion_generate'](model=model, metrics_collector=hook, **args)
        self.assertTrue(torch.equal(a, b))
        self.assertGreater(model.spec_calls, 0)
        self.assertEqual(native['accepted_tokens'], 32)
        self.assertGreater(native['accepted_tokens'], native['transfer_accepted_tokens'])
        for key in ['activated_experts', 'layer_forwards', 'denoising_forwards', 'accepted_tokens']:
            self.assertEqual(native[key], hook[key])
        timer_only = collector(experts=False)
        torch.manual_seed(12)
        c = scope['block_diffusion_generate'](model=model, metrics_collector=timer_only, **args)
        scope['_finalize_team_collector'](timer_only, c[:, 32:], 151669)
        self.assertTrue(torch.equal(a, c))
        self.assertEqual(timer_only['accepted_tokens'], 32)
        self.assertEqual(timer_only['layer_forwards'], 0)

    def test_seh_transition_count_includes_verified_prefill(self):
        c = collector()
        scope['_record_team_acceptance'](c, torch.tensor([[True, True, False]]),
            torch.tensor([[7, 8, 9]]), torch.tensor([[False, True, False]]), 151669)
        self.assertEqual(c['accepted_tokens'], 2)
        self.assertEqual(c['transfer_accepted_tokens'], 1)

    def test_timer_seed_restoration_and_sample_order_independence(self):
        saved = scope['block_diffusion_generate']
        scope['block_diffusion_generate'] = lambda **kwargs: torch.rand(3)
        try:
            model = ToyModel()
            state = torch.random.get_rng_state().clone()
            first, elapsed = scope['_run_team_generation'](model, {}, True, 456)
            self.assertGreater(elapsed, 0)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            torch.rand(100)
            second, _ = scope['_run_team_generation'](model, {}, False, 456)
            self.assertTrue(torch.equal(first, second))
        finally:
            scope['block_diffusion_generate'] = saved

    def test_timer_synchronizes_all_cuda_shards_from_device_map(self):
        model = ToyModel()
        model.hf_device_map = {'layer0': 0, 'layer1': 'cuda:1', 'offloaded': 'cpu'}
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'synchronize') as sync:
            scope['_synchronize_model_cuda'](model)
        self.assertEqual([call.args[0] for call in sync.call_args_list], [0, 1])

    def test_four_configs_disable_old_interventions_and_hook(self):
        for task in ['humaneval', 'mbpp', 'gsm8k', 'math']:
            path = ROOT / f'configs/sdar/team/frontier_pool/eval_sdar_hf_{task}_frontier_pool.py'
            nodes = [n for n in ast.parse(path.read_text()).body
                     if not isinstance(n, (ast.Import, ast.ImportFrom, ast.With))]
            model = dict(abbr='toy', path='toy', run_cfg={}, model_kwargs={}, generation_kwargs={})
            cfg = dict(models=[model], datasets=[dict(reader_cfg={}, infer_cfg=dict(inferencer={}))],
                infer=dict(runner={}), eval=dict(partitioner={}))
            with patch.dict(os.environ, {'MODE': 'metrics', 'TEAM_POOL_STRATEGY': 'protected'}):
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), cfg)
            self.assertTrue(model['collect_metrics'])
            self.assertEqual(model['metrics_backend'], 'native')
            self.assertEqual(model['accepted_token_mode'], 'mask_transitions')
            self.assertEqual(model['generation_kwargs']['cold_expert_budget'], 0)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AggregationTests(unittest.TestCase):
    def test_timing_only_and_native_metrics_aggregation(self):
        tool = load_file('team_metric_tool', ROOT / 'analysis/tools/calculate_team_metrics.py')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'team_metrics_pid1.jsonl'
            base = dict(schema='sdar_metrics_v1', activated_experts=20, layer_forwards=2,
                accepted_tokens=6, denoising_forwards=2, accepted_token_mode='mask_transitions')
            path.write_text(json.dumps(dict(base, generation_latency_s=2)) + '\n'
                            + json.dumps(dict(base, generation_latency_s=4)) + '\n')
            totals, _, _, count = tool.calculate_compact_metrics(Path(tmp), '*_metrics_*.jsonl')
            self.assertEqual(totals.apf, 10)
            self.assertEqual(totals.tpf, 3)
            self.assertEqual(sum(totals.generation_latencies_s), 6)
            self.assertEqual(totals.timed_accepted_tokens, 12)
            path.write_text(json.dumps(dict(base, activated_experts=0, layer_forwards=0,
                metrics_collected=False, generation_latency_s=2)) + '\n')
            totals, _, _, _ = tool.calculate_compact_metrics(Path(tmp), '*_metrics_*.jsonl')
            self.assertEqual(totals.layer_forwards, 0)
            self.assertEqual(totals.tpf, 3)

    def test_latency_comparison_rejects_different_instrumentation(self):
        tool = load_file('team_time_compare', ROOT / 'analysis/tools/compare_team_latency.py')
        a = {'prompt': dict(generation_latency_s=4, metrics_collected=False)}
        b = {'prompt': dict(generation_latency_s=2, metrics_collected=False)}
        self.assertEqual(tool.compare(a, b)['speedup'], 2)
        b['prompt']['metrics_collected'] = True
        with self.assertRaisesRegex(ValueError, 'metrics_collected'):
            tool.compare(a, b)


if __name__ == '__main__':
    unittest.main()
