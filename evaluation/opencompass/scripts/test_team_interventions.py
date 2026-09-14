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


if __name__ == '__main__':
    unittest.main()
