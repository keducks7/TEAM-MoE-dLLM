"""CPU smoke tests of the actual MoE class, without Transformers/model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F


class ToyExpert(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.proj(x)


source = Path(__file__).resolve().parents[3] / 'modeling_sdar_moe.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
            and n.name == 'SDARMoeSparseMoeBlock')
namespace = dict(torch=torch, nn=nn, F=F, SDARMoeMLP=ToyExpert)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)


class InterventionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.moe = namespace['SDARMoeSparseMoeBlock'](SimpleNamespace(
            num_experts=12, num_experts_per_tok=8, norm_topk_prob=True,
            hidden_size=4, moe_intermediate_size=8))
        self.x = torch.randn(1, 8, 4)
        self.past = torch.randn_like(self.x)
        self.decoded = torch.tensor([[True, False, False, False, False, False, False, False]])
        self.cold = torch.tensor([[False, False, False, False, True, True, True, True]])

    def run_moe(self, **kwargs):
        return self.moe(self.x, self.past, self.decoded, self.cold, **kwargs)[0]

    def test_off_is_identical(self):
        a = self.run_moe()
        b = self.run_moe(cold_reuse_mask=None, hot_mask=~self.cold & ~self.decoded,
                         hot_prune_count=0)
        self.assertTrue(torch.equal(a, b))

    def test_cold_reuse_and_hot_unchanged(self):
        a = self.run_moe()
        b = self.run_moe(cold_reuse_mask=self.cold)
        self.assertTrue(torch.equal(b[self.cold], self.past[self.cold]))
        torch.testing.assert_close(a[~self.cold], b[~self.cold])

    def test_hot_pruning_preserves_cold_and_decoded(self):
        a = self.run_moe()
        b = self.run_moe(hot_mask=~self.cold & ~self.decoded,
                         hot_prune_count=2, hot_protect_top1=False)
        torch.testing.assert_close(a[self.cold], b[self.cold])
        self.assertTrue(torch.equal(b[self.decoded], self.past[self.decoded]))
        self.assertTrue(torch.isfinite(b).all())

    def test_cold_zero_skips_router_rows(self):
        sizes = []
        handle = self.moe.gate.register_forward_pre_hook(
            lambda module, args: sizes.append(args[0].shape[0]))
        try:
            b = self.run_moe(cold_zero_mask=self.cold)
        finally:
            handle.remove()
        self.assertEqual(sizes, [3])
        self.assertTrue(torch.equal(b[self.cold], torch.zeros_like(b[self.cold])))
        self.assertTrue(torch.equal(b[self.decoded], self.past[self.decoded]))


class PackingTest(unittest.TestCase):
    def test_decoder_cache_uses_selected_parent_and_restores_coordinates(self):
        layer = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == 'SDARMoeDecoderLayer')
        forward = next(n for n in layer.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'forward')
        forward.returns = None
        for arg in forward.args.args + forward.args.kwonlyargs:
            arg.annotation = None
        forward.args.kwarg.annotation = None
        scope = dict(torch=torch)
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), scope)
        moe = namespace['SDARMoeSparseMoeBlock'](SimpleNamespace(
            num_experts=12, num_experts_per_tok=8, norm_topk_prob=True,
            hidden_size=4, moe_intermediate_size=8))
        fake = SimpleNamespace(input_layernorm=nn.Identity(),
                               post_attention_layernorm=nn.Identity(), mlp=moe,
                               self_attn=lambda **kw: (torch.zeros_like(kw['hidden_states']), None))
        active = torch.tensor([0, 3, 32, 35, 64, 67, 96, 99])
        past = torch.arange(128*4, dtype=torch.float32).view(1, 128, 4)
        x = torch.zeros(1, active.numel(), 4)
        output, cache = scope['forward'](
            fake, x, past_hidden_states=past, choice_id=2,
            decoded_index=torch.ones(1, active.numel(), dtype=torch.bool),
            expert_limit_index=torch.zeros(1, active.numel(), dtype=torch.bool),
            team_active_indices=active)
        expected = past[:, 64:96].repeat(1, 4, 1).index_select(1, active)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(cache.shape, past.shape)
        self.assertTrue(torch.equal(cache.index_select(1, active), expected))
        missing = torch.ones(128, dtype=torch.bool)
        missing[active] = False
        self.assertEqual(int(torch.count_nonzero(cache[:, missing])), 0)

    def test_branch_mask_and_original_positions(self):
        wrapper = source.parent / 'evaluation/opencompass/opencompass/models/huggingface_bd3_decoded_jump_expert_limit_speculative.py'
        nodes = ast.parse(wrapper.read_text()).body
        helper = next(n for n in nodes if isinstance(n, ast.FunctionDef)
                      and n.name == '_pack_team_positions')
        scope = dict(torch=torch)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(wrapper), 'exec'), scope)
        # Four branches of length 2, plus a 3-token cached prefix.
        attention = torch.zeros(1, 8, 11, dtype=torch.bool)
        attention[:, :, :3] = True
        for b in range(4):
            attention[:, b*2:b*2+2, 3+b*2:3+b*2+2] = True
        skip = torch.tensor([[False, True, True, False, False, True, True, False]])
        position = torch.tensor([[3, 4, 3, 4, 3, 4, 3, 4]])
        cache = [torch.randn(1, 8, 4)]
        kwargs = dict(input_ids=torch.arange(8).view(1, 8),
                      position_ids=position, cache_position=torch.arange(3, 11),
                      attention_mask=attention, stored_hidden_states=cache,
                      hot_mask=~skip, cold_reuse_mask=None)
        packed, active = scope['_pack_team_positions'](kwargs, skip)
        self.assertEqual(active.tolist(), [0, 3, 4, 7])
        self.assertEqual(packed['position_ids'].tolist(), [[3, 4, 3, 4]])
        self.assertEqual(packed['cache_position'].tolist(), [3, 6, 7, 10])
        self.assertTrue(packed['attention_mask'][:, :, :3].all())
        self.assertTrue(torch.equal(packed['attention_mask'][0, :, 3:], torch.eye(4, dtype=torch.bool)))
        self.assertIs(packed['stored_hidden_states'], cache)


if __name__ == '__main__':
    unittest.main()
