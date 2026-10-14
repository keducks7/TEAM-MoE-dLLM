"""Scientific checks for temporal leakage, censoring, and equal pool budgets."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis/tools'))
import analyze_decoding_expert_structure as analysis

spec = importlib.util.spec_from_file_location('trace_fixture', ROOT / 'scripts/test_dependency_motivation.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


class StructureTests(unittest.TestCase):
    def test_shared_mass_uses_original_topk_normalization(self):
        ids = np.array([[0, 1], [0, 2]])
        probs = np.array([[.6, .3, .1], [.6, .1, .3]])
        mass = analysis.routing_mass(ids, probs)
        result = analysis.pair_features(ids, mass, np.array([0]), np.array([1]))
        self.assertAlmostEqual(result['overlap'][0], .5)
        self.assertAlmostEqual(result['shared_topk_mass'][0], 2 / 3)
        self.assertAlmostEqual(result['private_topk_mass'][0], 1 / 3)

    def test_directed_dependency_excludes_self(self):
        attn = np.array([[.9, .1], [.3, .7]])
        np.testing.assert_allclose(analysis.dependency_scores(attn, [0, 1], [0, 1]), [.3, .1])
        np.testing.assert_allclose(analysis.dependency_scores(attn, [0], [0, 1]), [0, .1])

    def test_next_labels_are_censored_at_gaps_and_block_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dependency_fixture.npz'
            fixture_module.fixture(path)
            _, data = analysis.load_trace(path)
            self.assertIsNotNone(analysis.next_accepted(data, 0))
            self.assertIsNone(analysis.next_accepted(data, 1))
            self.assertIsNone(analysis.next_accepted(data, 2))
            fixture_module.fixture(path, gap=True)
            _, data = analysis.load_trace(path)
            self.assertIsNone(analysis.next_accepted(data, 0))
            self.assertIsNone(analysis.historical_features(data, 1, 0))

    def test_historical_policy_ignores_current_and_future_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dependency_fixture.npz'
            fixture_module.fixture(path)
            _, data = analysis.load_trace(path)
            original = analysis.historical_features(data, 1, 0)
            data['confidence'][1:] = 0
            data['attention'][1:] = 0
            data['accepted'][1:] = True
            changed = analysis.historical_features(data, 1, 0)
            for key in original:
                np.testing.assert_array_equal(original[key], changed[key])

    def test_equal_budget_and_normalized_mass_coverage(self):
        ids = np.array([[0, 1], [0, 2]])
        probs = np.array([[.6, .3, .1], [.6, .1, .3]])
        mass = analysis.routing_mass(ids, probs)
        groups = dict(all=np.ones(2, dtype=bool), accept_now=np.array([True, False]),
            accept_next=np.zeros(2, dtype=bool), remaining=np.array([False, True]),
            dependency_sources=np.zeros(2, dtype=bool))
        pool = analysis.choose_pool(mass.sum(0), np.ones(3, dtype=bool), 2, np.array([2, 1, 1]))
        np.testing.assert_array_equal(pool, [0, 1])
        result = analysis.pool_metrics(ids, probs, mass, pool, groups)
        self.assertAlmostEqual(result['mass_all'], 5 / 6)
        self.assertEqual(result['mass_accept_now'], 1.)
        self.assertIsNone(result['mass_accept_next'])
        self.assertEqual(result['active_experts_after'], 2)
        self.assertEqual(result['active_expert_reduction'], 1)

    def test_trace_analysis_keeps_policies_on_identical_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dependency_fixture.npz'
            fixture_module.fixture(path)
            _, _, _, _, _, pools = analysis.analyze_trace(path, budgets=(8, 16))
            self.assertTrue(pools)
            self.assertEqual({r['step'] for r in pools}, {1})
            self.assertEqual({r['block'] for r in pools}, {1})
            for requested in (8, 16):
                rows = [r for r in pools if r['layer'] == 0 and r['budget_requested'] == requested]
                self.assertEqual(len(rows), 9)
                self.assertEqual(len({r['budget_actual'] for r in rows}), 1)
                self.assertTrue(all(r['mass_accept_next'] is None for r in rows))
            uncompressed = [r for r in pools if not r['compressed']]
            self.assertTrue(all(abs(r['mass_all'] - 1) < 1e-12 for r in uncompressed))

    def test_prompt_summary_does_not_overweight_long_trajectories(self):
        base = dict(layer=0, trace='x', block=0, step=0)
        rows = [dict(base, sample='a', value=1),
                dict(base, sample='a', step=1, value=1),
                dict(base, sample='b', value=0)]
        summary = analysis.macro_summary(rows, ['layer'], ['value'])[0]
        self.assertEqual(summary['value'], .5)
        self.assertEqual(summary['value_prompts'], 2)


if __name__ == '__main__':
    unittest.main()
