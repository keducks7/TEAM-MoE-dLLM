"""NumPy matching/drift tests; CPU replay tests require server PyTorch."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis/tools'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load('route_probe', ROOT / 'opencompass/models/sdar_dependency_probe.py')
drift = load('drift', ROOT / 'analysis/tools/analyze_dependency_drift.py')
report = load('probe_report', ROOT / 'analysis/tools/analyze_dependency_probes.py')
old_tests = load('old_tests', ROOT / 'scripts/test_dependency_motivation.py')


def frame_fixture():
    t, e, k = 8, 32, 8
    ids = np.tile(np.arange(k), (t, 1))
    ids[1] = np.arange(6, 14)
    ids[4] = np.arange(14, 22)
    ids[5] = np.arange(20, 28)
    probs = np.ones((t, e)) * (.2 / (e - k))
    for i in range(t):
        probs[i, ids[i]] = np.linspace(.11, .09, k)
    attention = np.ones((t, t)) * .01
    for i in range(4):
        for j in range(4):
            if i != j:
                attention[i, j] = .2
    return dict(mask=np.ones(t, bool), positions=np.arange(t),
                confidence=np.ones(t) * .7, accepted=np.zeros(t, bool),
                router_probs=probs[None], expert_ids=ids[None], attention=attention[None])


class MatchingTests(unittest.TestCase):
    def test_disjoint_pairs_fixed_pool_top1_and_matched_loss(self):
        frame = frame_fixture()
        match = probe.select_matched_pairs(frame, 0, pool_size=12)
        self.assertIsNotNone(match)
        a, b = match
        self.assertEqual(a['stratum'], b['stratum'])
        self.assertFalse(set(a['tokens']) & set(b['tokens']))
        for pair in match:
            self.assertEqual(len(pair['pool']), 12)
            for token in pair['tokens']:
                self.assertIn(int(frame['expert_ids'][0, token, 0]), pair['pool'])
        self.assertLessEqual(max(abs(x-y) for x,y in zip(sorted(a['removed_mass']), sorted(b['removed_mass']))), .03)

    def test_no_silent_relaxation_or_fake_intervention(self):
        frame = frame_fixture()
        frame['attention'][:] = .01
        self.assertIsNone(probe.select_matched_pairs(frame, 0))
        self.assertEqual(probe.candidate_pairs(frame_fixture(), 0, pool_size=32), [])
        with self.assertRaises(ValueError):
            probe.candidate_pairs(frame_fixture(), 0, pool_size=7)

    def test_drift_does_not_cross_blocks_or_step_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'test.npz'
            old_tests.fixture(file)
            _, rows = drift.drift_records(file)
            self.assertTrue(rows)
            self.assertTrue(all(r['block'] == 1 and r['step'] == 1 for r in rows))
            old_tests.fixture(file, gap=True)
            self.assertEqual(drift.drift_records(file)[1], [])

    def test_partial_rank_short_or_constant_is_unavailable(self):
        self.assertIsNone(drift.partial_rank_correlation(np.ones(8), np.arange(8), np.zeros((8, 4))))
        self.assertIsNone(drift.partial_rank_correlation(np.ones(20), np.arange(20), np.zeros((20, 4))))

    def test_no_match_report_keeps_coverage(self):
        summary = report.summarize([dict(status='no_match', sample='a')])
        self.assertEqual(summary['attempted_states'], 1)
        self.assertEqual(summary['matched_states'], 0)
        self.assertEqual(summary['match_rate'], 0)
        self.assertIsNone(summary['effects']['other_prediction_flip_rate']['strong_minus_weak']['mean'])

    def test_cli_reports_and_matching_validation(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_tests.fixture(root / 'dependency_test.npz')
            subprocess.run([sys.executable, str(ROOT / 'analysis/tools/analyze_dependency_drift.py'),
                            '--trace-dir', str(root), '--output-dir', str(root / 'drift')],
                           check=True, capture_output=True)
            self.assertTrue((root / 'drift/report.md').exists())
            pair = probe.select_matched_pairs(frame_fixture(), 0)
            effects = {k: 0. for k in ['own_prediction_flip_rate', 'other_prediction_flip_rate',
                       'own_confidence_delta', 'other_confidence_abs_delta', 'own_accepted_lost',
                       'other_accepted_lost', 'own_accepted_gained', 'other_accepted_gained',
                       'baseline_accepted_prediction_flips', 'accepted_set_changes']}
            record = dict(schema='sdar_dependency_probe_v1', sample='test', block=0, step=0,
                          layer=0, status='matched', source_sha256='toy', pool_size=12,
                          mass_tolerance=.03, confidence_tolerance=.05,
                          sham_logit_max_abs=0.)
            for trial in pair:
                record[trial['group']] = dict(trial, effects=effects)
            (root / 'probe_test.jsonl').write_text(json.dumps(record) + '\n')
            command = [sys.executable, str(ROOT / 'analysis/tools/analyze_dependency_probes.py'),
                       '--probe-dir', str(root), '--output-dir', str(root / 'report')]
            subprocess.run(command, check=True, capture_output=True)
            self.assertTrue((root / 'report/report.md').exists())
            record['weak']['stratum'] = list(record['weak']['stratum'])
            record['weak']['stratum'][0] += 1
            (root / 'probe_test.jsonl').write_text(json.dumps(record) + '\n')
            command[-1] = str(root / 'invalid')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unmatched', result.stderr)


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None


if torch is not None:
    class ToyMoe(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 8
            self.norm_topk_prob = True
            self.gate = nn.Linear(4, 16)
            self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(16)])

        def forward(self, hidden):
            flat = hidden.reshape(-1, 4)
            logits = self.gate(flat)
            weights, ids = logits.softmax(-1).topk(8)
            weights = weights / weights.sum(-1, keepdim=True)
            output = torch.zeros_like(flat)
            for e in range(16):
                rows, slots = torch.where(ids == e)
                output.index_add_(0, rows, self.experts[e](flat[rows]) * weights[rows, slots, None])
            return output.reshape_as(hidden), logits

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 4)
            self.model = nn.Module()
            layer = nn.Module()
            layer.mlp = ToyMoe()
            self.model.layers = nn.ModuleList([layer])

        def forward(self, input_ids, **kwargs):
            torch.rand(1)  # Check restoration even when replay consumes RNG.
            return self.model.layers[0].mlp(self.embedding(input_ids))[0]


@unittest.skipIf(torch is None, 'PyTorch unavailable locally: run replay tests on server')
class ReplayTests(unittest.TestCase):
    def test_restricted_routes_preserve_other_rows_and_router_output(self):
        torch.manual_seed(9)
        module = ToyMoe().eval()
        hidden = torch.randn(1, 8, 4)
        output = module(hidden)
        routed = probe.restricted_output(module, output, hidden.reshape(-1, 4), output[1], [0, 1], list(range(12)))
        self.assertTrue(torch.equal(output[0][:, 2:], routed[0][:, 2:]))
        self.assertIs(output[1], routed[1])
        noop = probe.restricted_output(module, output, hidden.reshape(-1, 4), output[1], [0, 1], list(range(16)))
        torch.testing.assert_close(noop[0], output[0], atol=1e-6, rtol=1e-6)

    def test_replay_preserves_teacher_inputs_rng_and_cache(self):
        torch.manual_seed(9)
        model = ToyModel().eval()
        input_ids = torch.arange(8)[None]
        original_ids = input_ids.clone()
        cache = [(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))]
        original_cache = [t.clone() for t in cache[0]]
        kwargs = dict(input_ids=input_ids, past_key_values=cache, store_kv=False)
        baseline = model(**kwargs)
        mask = np.ones(8, bool)
        pred, conf, accepted = probe.greedy_decisions(baseline, mask, .95, 1)
        frame = dict(block_id=1, step=0, mask=mask, predictions=pred, confidence=conf, accepted=accepted)
        metadata = dict(prompt_sha256='toy', sequence=0, model_source_sha256='toy')
        observation = SimpleNamespace(last_record=frame, layer_ids=[0])
        trials = (dict(group='strong', tokens=[0, 1], pool=list(range(12))),
                  dict(group='weak', tokens=[4, 5], pool=list(range(12))))
        with tempfile.TemporaryDirectory() as tmp:
            runner = probe.DependencyProbe(model, tmp, layer=0)
            runner.start_sequence(metadata)
            state = torch.random.get_rng_state().clone()
            with patch.object(probe, 'select_matched_pairs', return_value=trials):
                runner.run(observation, kwargs, baseline, lambda x, _model: x, .95, 1)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(torch.equal(input_ids, original_ids))
            for a, b in zip(cache[0], original_cache):
                self.assertTrue(torch.equal(a, b))
            self.assertIsNone(runner.trial)
            record = json.loads(next(Path(tmp).glob('*.jsonl')).read_text())
            self.assertEqual(record['status'], 'matched')
            self.assertEqual(record['sham_logit_max_abs'], 0.)


if __name__ == '__main__':
    unittest.main()
