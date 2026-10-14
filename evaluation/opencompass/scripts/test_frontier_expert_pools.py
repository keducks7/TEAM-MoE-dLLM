"""Checks for minority protection, strict budgets, and dependency ablations."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis/tools'))
import analyze_frontier_expert_pools as pools

spec = importlib.util.spec_from_file_location('pool_fixture', ROOT / 'scripts/test_dependency_motivation.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class PoolTests(unittest.TestCase):
    def test_top1_protection_and_worst_hot_coverage(self):
        ids = np.array([[0, 1], [2, 0], [0, 3]])
        mass = np.array([[.8, .2, 0, 0], [.4, 0, .6, 0], [.5, 0, 0, .5]])
        hot = np.array([True, True, False])
        selected, info = pools.protected_prefix(ids, mass, hot, 3, .9)
        np.testing.assert_array_equal(np.flatnonzero(selected), [0, 1, 2])
        self.assertEqual(info['protected_top1_count'], 2)
        self.assertEqual(info['hot_fill_count'], 1)
        self.assertEqual(info['hot_target_reached_before_fill'], 1)

    def test_budget_exhaustion_does_not_silently_add_experts(self):
        ids = np.array([[0, 1], [2, 0]])
        mass = np.array([[.8, .2, 0], [.4, 0, .6]])
        selected, info = pools.protected_prefix(ids, mass, np.ones(2, dtype=bool), 2, .9)
        self.assertEqual(selected.sum(), 2)
        self.assertEqual(info['hot_target_reached_before_fill'], 0)

    def test_infeasible_budget_is_explicit(self):
        self.assertEqual(pools.aligned_budget(8, 20, 10, 8), (10, 1))
        self.assertEqual(pools.aligned_budget(16, 20, 10, 8), (16, 0))
        self.assertEqual(pools.aligned_budget(32, 20, 10, 8), (20, 0))
        with self.assertRaises(ValueError):
            pools.protected_prefix(np.array([[0], [1]]), np.eye(2), np.ones(2, dtype=bool), 1)

    def test_dependency_can_change_remaining_pool_without_changing_protection(self):
        ids = np.array([[0, 1], [2, 0], [2, 0], [3, 0]])
        mass = np.array([[.95, .05, 0, 0], [.4, 0, .6, 0],
                         [.4, 0, .6, 0], [.2, 0, 0, .8]])
        hot = np.array([True, False, False, False])
        dep = np.array([0, 0, 0, 1.])
        shuffled = np.array([0, 1., 0, 0])
        result, info = pools.construct_pools(ids, mass, hot, dep, shuffled, 2)
        np.testing.assert_array_equal(result['protected_shared'], [0, 2])
        np.testing.assert_array_equal(result['protected_dependency'], [0, 3])
        np.testing.assert_array_equal(result['protected_shuffled_dependency'], [0, 2])
        self.assertTrue(all(0 in pool and len(pool) == 2 for pool in result.values()))
        self.assertEqual(info['dependency_opportunity'], 1)

    def test_no_space_means_no_dependency_effect(self):
        ids = np.array([[0, 1], [2, 0], [3, 0]])
        mass = np.array([[.8, .2, 0, 0], [.4, 0, .6, 0], [.2, 0, 0, .8]])
        result, info = pools.construct_pools(ids, mass, np.array([True, True, False]),
            np.array([0., 0, 1]), np.array([0., 0, 1]), 2)
        np.testing.assert_array_equal(result['protected_shared'], result['protected_dependency'])
        self.assertEqual(info['dependency_opportunity'], 0)

    def test_policy_constructor_cannot_see_acceptance_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dependency_fixture.npz'
            fixtures.fixture(path)
            _, _, states, records = pools.analyze_trace(path, budgets=(8, 16))
            self.assertTrue(states)
            self.assertEqual({r['step'] for r in records}, {1})
            self.assertEqual({r['policy'] for r in records}, set(pools.POLICIES))
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays['accepted'][1:] = False
            arrays['confidence'][1:] = .01
            arrays['attention'][1:] = 0
            np.savez(path, **arrays)
            _, _, changed_states, changed = pools.analyze_trace(path, budgets=(8, 16))
            self.assertEqual([r['pool_ids'] for r in records], [r['pool_ids'] for r in changed])
            self.assertEqual([r['budget_actual'] for r in states], [r['budget_actual'] for r in changed_states])

    def test_paired_comparison_rejects_capacity_mismatch(self):
        base = dict(sample='x', trace='x', block=0, step=1, layer=0, budget_requested=8,
                    **{m: 0. for m in pools.METRICS})
        rows = [dict(base, policy='protected_shared', budget_actual=8),
                dict(base, policy='protected_dependency', budget_actual=9)]
        with self.assertRaisesRegex(ValueError, 'unequal budgets'):
            pools.deltas(rows, 'protected_shared')


if __name__ == '__main__':
    unittest.main()
