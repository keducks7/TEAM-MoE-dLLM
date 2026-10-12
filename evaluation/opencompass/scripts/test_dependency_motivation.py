"""NumPy tests plus optional CPU PyTorch observation/parity tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # inspect.getfile needs the defining module registered for toy models.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_module('dependency_analysis', ROOT / 'analysis/tools/analyze_dependency_motivation.py')


def fixture(path, gap=False):
    rng = np.random.default_rng(1)
    f, l, t, e, k = 3, 2, 8, 12, 8
    probs = rng.random((f, l, t, e)).astype(np.float32)
    probs /= probs.sum(-1, keepdims=True)
    attention = rng.random((f, l, t, t)).astype(np.float32)
    attention /= attention.sum(-1, keepdims=True)
    attention[1:] = attention[0]
    mask = np.ones((f, t), dtype=bool)
    mask[1:, 0] = False
    accepted = np.zeros_like(mask)
    accepted[0, 0] = True
    ids = np.argsort(-probs, axis=-1)[..., :k]
    meta = dict(schema='sdar_dependency_v1', method='vanilla', prompt_sha256='toy',
                model_source_sha256='toy_source', threshold=.95, layer_ids=[0, 5])
    np.savez(path, metadata=np.asarray(json.dumps(meta)),
             layer_ids=np.array([0, 5]), attention=attention,
             router_probs=probs, expert_ids=ids,
             topk_probs=np.take_along_axis(probs, ids, -1),
             hidden_cosine=np.tile(np.eye(t), (f, l, 1, 1)),
             mask=mask, accepted=accepted, positions=np.tile(np.arange(t), (f, 1)),
             confidence=rng.random((f, t)), predictions=np.zeros((f, t)),
             token_ids=np.zeros((f, t)), block_id=np.array([1, 1, 2]),
             step=np.array([0, 2 if gap else 1, 0]))


class AnalysisTests(unittest.TestCase):
    def test_rank_ties_and_constants(self):
        np.testing.assert_equal(analysis.ranks(np.array([2, 1, 1])), [2, .5, .5])
        self.assertIsNone(analysis.spearman(np.ones(8), np.arange(8)))
        self.assertAlmostEqual(analysis.spearman(np.arange(8), np.arange(8)), 1)

    def test_topk_private_mass(self):
        ids = np.array([[0, 1], [0, 2]])
        probs = np.array([[.6, .3, .1], [.6, .1, .3]])
        result = analysis.expert_pair_features(ids, probs, np.array([0]), np.array([1]))
        self.assertAlmostEqual(result['overlap_at_k'][0], .5)
        self.assertAlmostEqual(result['private_topk_mass_fraction'][0], 1 / 3)
        self.assertEqual(result['union_size'][0], 3)

    def test_alignment_and_no_block_crossing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dependency_test.npz'
            fixture(path)
            _, history, relation = analysis.analyze_trace(path)
            self.assertTrue(history)
            self.assertTrue(relation)
            self.assertTrue(all(r['block'] == 1 and r['step'] == 1 for r in history))
            # Seven persistently masked positions => 21 unordered pairs.
            row = next(r for r in history if r['proxy'] == 'previous_attention' and r['distance'] == 'all')
            self.assertEqual(row['pairs'], 21)
            self.assertAlmostEqual(row['spearman'], 1)
            self.assertFalse(any(r['proxy'] == 'previous_confidence_change_similarity' for r in history))
            fixture(path, gap=True)
            self.assertFalse(analysis.analyze_trace(path)[1])

    def test_prompt_cluster_summary(self):
        rows = [dict(sample='a', layer=0, value=1.), dict(sample='a', layer=0, value=1.),
                dict(sample='b', layer=0, value=0.)]
        result = analysis.summarize(rows, ['layer'], ['value'])[0]
        self.assertEqual(result['prompts'], 2)
        self.assertEqual(result['value'], .5)

    def test_matched_contrasts_do_not_mix_states(self):
        base = dict(sample='a', trace='x', block=0, step=0, layer=0,
                    mask_stage='.75-1', distance='1', confidence_bin='>=.9', role='neither_accepted')
        rows = [dict(base, dependency_group='strong', pairs=2, overlap_at_k=.8),
                dict(base, dependency_group='weak', pairs=3, overlap_at_k=.2),
                dict(base, step=1, dependency_group='strong', pairs=4, overlap_at_k=.9)]
        result = analysis.matched_contrasts(rows, ['overlap_at_k'])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['overlap_at_k'], .6)

    def test_cli_report(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / 'dependency_fixture.npz')
            subprocess.run([sys.executable, str(ROOT / 'analysis/tools/analyze_dependency_motivation.py'),
                            '--trace-dir', str(root), '--output-dir', str(root / 'report')],
                           check=True, capture_output=True)
            report = root / 'report'
            self.assertTrue((report / 'report.md').exists())
            self.assertTrue((report / 'matched_contrast_summary.csv').exists())
            self.assertEqual(json.loads((report / 'summary.json').read_text())['prompts'], 1)

    def test_prediction_comparison_ignores_evaluation_metadata(self):
        compare = load_module('prediction_compare', ROOT / 'scripts/compare_dependency_predictions.py')
        self.assertEqual(compare.prediction_strings({'0': {'prediction': 'abc', 'latency': 12}}),
                         compare.prediction_strings({'0': {'prediction': 'abc', 'latency': 13}}))


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None


if torch is not None:
    observer = load_module('dependency_observer', ROOT / 'opencompass/models/sdar_dependency_observer.py')

    def apply_rotary_pos_emb(q, k, cos, sin):
        return q, k

    class ToyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_norm, self.k_norm = nn.Identity(), nn.Identity()
            self.layer_idx = 0
            self.num_key_value_groups = 2
            self.scaling = .5
            self.attention_dropout = 0

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, **kwargs):
            q = self.q_norm(hidden_states.view(1, -1, 2, 4)).transpose(1, 2)
            k = self.k_norm(hidden_states[..., :4].view(1, -1, 1, 4)).transpose(1, 2)
            if past_key_value:
                k = torch.cat([past_key_value[0][0], k], dim=-2)
            attention_mask = attention_mask.bool()
            out = F.scaled_dot_product_attention(q, k, k, attn_mask=attention_mask,
                                                enable_gqa=True, scale=self.scaling)
            return out.transpose(1, 2).reshape_as(hidden_states), None

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            layer = nn.Module()
            layer.self_attn = ToyAttention()
            layer.mlp = nn.Module()
            layer.mlp.top_k = 8
            layer.mlp.experts = nn.ModuleList([nn.Identity() for _ in range(12)])
            layer.mlp.gate = nn.Linear(8, 12)
            self.model.layers = nn.ModuleList([layer])

        def forward(self, x, kwargs):
            y = self.model.layers[0].self_attn(hidden_states=x, **kwargs)[0]
            logits = self.model.layers[0].mlp.gate(y.reshape(-1, 8))
            return y, logits


@unittest.skipIf(torch is None, 'PyTorch not installed; run this suite on server for hook parity')
class HookTests(unittest.TestCase):
    def test_outputs_rng_and_cache_unchanged_and_prefix_denominator(self):
        torch.manual_seed(4)
        model = ToyModel().eval()
        x = torch.randn(1, 8, 8)
        prefix = torch.randn(1, 1, 3, 4)
        cache = [(prefix, prefix.clone())]
        mask = torch.ones(1, 8, 11, dtype=torch.bool)
        mask[:, :, 0] = False
        kwargs = dict(position_embeddings=(torch.zeros(1), torch.zeros(1)),
                      attention_mask=mask, past_key_value=cache, store_kv=False)
        baseline = model(x, kwargs)
        state = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            hooks = observer.DependencyObserver(model, tmp)
            hooks.start_sequence(torch.arange(8)[None], 'toy', .95)
            hooks.begin_step(0, 0, torch.arange(8)[None], torch.zeros(1, 8, dtype=torch.long),
                             torch.ones(1, 8, dtype=torch.bool))
            traced = model(x, kwargs)
            hooks.end_forward()
            for a, b in zip(baseline, traced):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(torch.equal(prefix, cache[0][0]))
            recorded = hooks.capture[0]['attention']
            q = x.view(1, 8, 2, 4).transpose(1, 2)
            k = torch.cat([prefix, x[..., :4].view(1, 8, 1, 4).transpose(1, 2)], -2).repeat_interleave(2, 1)
            weights = (q @ k.transpose(-1, -2) * .5).masked_fill(~mask.unsqueeze(1), -torch.inf).softmax(-1)
            np.testing.assert_allclose(recorded, weights.mean(1)[0, :, -8:].numpy(), rtol=1e-5, atol=1e-6)
            self.assertTrue(np.all(recorded.sum(-1) < 1))
            hooks.finish_step(torch.zeros(1, 8), torch.ones(1, 8), torch.zeros(1, 8, dtype=torch.bool))
            hooks.finish_sequence()
            self.assertEqual(len(list(Path(tmp).glob('*.npz'))), 1)


if __name__ == '__main__':
    unittest.main()
