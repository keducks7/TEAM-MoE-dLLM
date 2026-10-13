"""Offline protocol tests plus CPU replay checks when PyTorch is installed."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis/tools'))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = ModuleType('frontier_test_models'); package.__path__ = []
sys.modules[package.__name__] = package
parent = load('frontier_test_models.sdar_dependency_probe', ROOT / 'opencompass/models/sdar_dependency_probe.py')
probe = load('frontier_test_models.sdar_frontier_probe', ROOT / 'opencompass/models/sdar_frontier_probe.py')
report = load('frontier_report', ROOT / 'analysis/tools/analyze_frontier_probes.py')


def fixture():
    n = 5
    probs = np.tile(np.arange(16, 0, -1, dtype=float), (n, 1))
    probs /= probs.sum(1, keepdims=True)
    attention = np.ones((1, n, n)) * .01
    attention[0, 2, 1] = .5
    attention[0, 2, 3] = .001
    accepted = np.zeros(n, bool); accepted[2] = True
    confidence = np.ones(n) * .7; confidence[2] = .99
    return dict(mask=np.ones(n, bool), accepted=accepted, confidence=confidence,
                positions=np.arange(n), block_id=0, step=1, attention=attention,
                predictions=np.zeros(n, dtype=int), router_probs=probs[None],
                expert_ids=np.tile(np.arange(8), (1, n, 1)))


class ProtocolTests(unittest.TestCase):
    def test_directed_source_selection_and_exact_rank_replacement(self):
        frame = fixture()
        candidates, reason = probe.source_candidates(frame, 0)
        self.assertIsNone(reason)
        self.assertNotIn(2, [c['token'] for c in candidates])
        a, b = probe.match_sources(candidates)
        self.assertEqual((a['token'], b['token']), (1, 3))
        self.assertEqual(a['original_pool'], list(range(8)))
        self.assertEqual(a['replacement_pool'], [*range(7), 8])
        self.assertEqual(len(set(a['replacement_pool'])), 8)
        frame['attention'][0, 1, 2] = 100  # Opposite direction must not affect score.
        altered, _ = probe.source_candidates(frame, 0)
        self.assertEqual(candidates, altered)

    def test_controls_and_no_silent_relaxation(self):
        candidates, _ = probe.source_candidates(fixture(), 0)
        next(c for c in candidates if c['token'] == 3)['replacement_mass'] += .1
        self.assertIsNone(probe.match_sources(candidates))
        selected, kind = probe.select_sources(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(kind, 'stratified_unmatched')
        frame = fixture(); frame['accepted'][:] = False
        self.assertEqual(probe.source_candidates(frame, 0), ([], 'no_frontier'))
        frame = fixture(); frame['attention'][:] = 0
        self.assertEqual(probe.source_candidates(frame, 0), ([], 'tied_dependency'))

    def test_history_never_crosses_block_or_step_gap(self):
        current = fixture(); previous = fixture(); previous['step'] = 0
        candidates, _ = probe.source_candidates(current, 0, previous)
        self.assertTrue(all(c[report.HISTORY[0]] is not None for c in candidates))
        self.assertTrue(all(c[report.HISTORY[1]] is not None for c in candidates))
        for key, value in [('block_id', 1), ('step', -1)]:
            changed = dict(previous); changed[key] = value
            candidates, _ = probe.source_candidates(current, 0, changed)
            self.assertTrue(all(c[report.HISTORY[0]] is None for c in candidates))
            self.assertTrue(all(c[report.HISTORY[1]] is None for c in candidates))

    def test_frozen_receivers_and_signed_margin(self):
        base = dict(predictions=np.array([0, 1, 2]), accepted=np.array([False, True, True]),
                    confidence=np.array([.5, .96, .98]), original_logprob=np.array([-1., -.1, -.2]),
                    original_margin=np.array([1., 2., 3.]))
        changed = {k: v.copy() for k, v in base.items()}
        changed['predictions'][1] = 3; changed['accepted'][2] = False
        changed['original_margin'][1] = -1
        effect = probe.measured_effect(base, changed, [1, 2], 0)
        self.assertEqual(effect['frontier_prediction_flip_rate'], .5)
        self.assertEqual(effect['frontier_acceptance_lost_rate'], .5)
        self.assertEqual(effect['frontier_margin_abs_delta'], 1.5)
        self.assertEqual(effect['receiver_margin_delta'], [-3., 0.])

    def test_empty_and_failed_controls_not_treated_as_zero_effects(self):
        result = report.summarize([dict(status='no_match', no_match_reason='no_frontier', sample='a'),
                                   dict(status='control_failed', sample='b', sham_logit_max_abs=0.,
                                        strong=dict(control_logit_max_abs=.1),
                                        weak=dict(control_logit_max_abs=.2))])
        self.assertEqual(result['matched_states'], 0)
        self.assertEqual(result['status_counts']['control_failed'], 1)
        self.assertEqual(result['max_control_logit_error'], .2)
        self.assertIsNone(result['effects_vs_original'][report.METRICS[0]]['strong']['mean'])
        self.assertEqual(result['predictive_checks']['current']['status'], 'insufficient_prompts')

    def test_held_out_prediction_and_prompt_splits(self):
        rng = np.random.default_rng(42)
        rows = []
        for sample in range(12):
            for _ in range(10):
                signal = rng.normal()
                rows.append(dict(sample=str(sample), nuisance=float(rng.normal()),
                                 dependency=signal, outcome=2*signal + rng.normal(scale=.01)))
        base = report.held_out_mse(rows, ['nuisance'], 'outcome')
        extra = report.held_out_mse(rows, ['nuisance', 'dependency'], 'outcome')
        self.assertEqual(len(extra), 12)
        self.assertLess(np.mean(list(extra.values())), np.mean(list(base.values())) / 10)
        # Changing labels of one held-out prompt cannot change another row's
        # prediction in that same held-out fold. Constant offset exposes leakage.
        modified = [dict(r, outcome=r['outcome']+10 if r['sample']=='0' else r['outcome']) for r in rows]
        new = report.held_out_mse(modified, ['nuisance', 'dependency'], 'outcome')
        self.assertGreater(new['0'], 90)

    def test_duplicate_and_mixed_records_rejected(self):
        row = dict(schema='sdar_frontier_probe_v1', status='no_match', sample='x',
                   layer=24, block=0, step=0, no_match_reason='no_frontier')
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'frontier_test.jsonl'
            file.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                report.load_records(Path(tmp))
            other = dict(row, step=1, layer=25)
            file.write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
            with self.assertRaisesRegex(ValueError, 'Mixed'):
                report.load_records(Path(tmp))

    def test_report_cli_and_unmatched_separation(self):
        candidates, _ = probe.source_candidates(fixture(), 0)
        a, b = probe.match_sources(candidates)
        effects = {m: 0. for m in report.METRICS}
        effects.update(receiver_prediction_flips=[False], receiver_acceptance_lost=[False],
                       receiver_logprob_delta=[0.], receiver_margin_delta=[0.])
        row = dict(schema='sdar_frontier_probe_v1', sample='x', block=0, step=0, layer=24,
                   status='observed_unmatched', selection='stratified_unmatched', frontier=[2],
                   baseline_frontier_margin=[3.], sham_logit_max_abs=0.,
                   mass_tolerance=.03, confidence_tolerance=.05)
        for side, candidate in [('strong', a), ('weak', b)]:
            row[side] = dict(candidate, control_decisions_equal=True, control_logit_max_abs=0.,
                            effects=effects, control_effects=effects, relative_to_control=effects)
        report.validate(row)
        result = report.summarize([row])
        self.assertEqual(result['matched_states'], 0)
        self.assertEqual(result['observed_states'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'frontier_test.jsonl').write_text(json.dumps(row)+'\n')
            subprocess.run([sys.executable,str(ROOT/'analysis/tools/analyze_frontier_probes.py'),
                            '--probe-dir',str(path),'--output-dir',str(path/'out')],
                           check=True,capture_output=True)
            self.assertTrue((path/'out/report.md').exists())
            result=json.loads((path/'out/summary.json').read_text())
            self.assertEqual(result['observed_states'],1)
        row['status']='matched'
        with self.assertRaisesRegex(ValueError,'Unmatched observations'):
            report.validate(row)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed locally; run on server')
class ReplayTests(unittest.TestCase):
    def test_five_replays_keep_inputs_rng_cache_and_original_generation(self):
        old = load('frontier_toy_tests', ROOT/'scripts/test_dependency_next_steps.py')
        torch.manual_seed(21)
        model = old.ToyModel().eval()
        inputs = torch.arange(8)[None]
        cache = [(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))]
        kwargs = dict(input_ids=inputs, past_key_values=cache, store_kv=False)
        with torch.no_grad():
            baseline = model(**kwargs)
            pred, conf, accepted = parent.greedy_decisions(baseline, np.ones(8, bool), .95, 1)
            logits = model.model.layers[0].mlp.gate(model.embedding(inputs).reshape(-1,4))
            probs = logits.float().softmax(-1)
            ids = probs.topk(8).indices.cpu().numpy()
            frame = dict(block_id=0, step=0, positions=np.arange(8), mask=np.ones(8,bool),
                         accepted=accepted, confidence=conf, predictions=pred,
                         attention=np.ones((1,8,8)), router_probs=probs.cpu().numpy()[None], expert_ids=ids[None])
            candidates = []
            for group, token in zip(('strong','weak'), np.flatnonzero(~accepted)[:2]):
                extra = int(next(e for e in range(16) if e not in ids[token]))
                candidates.append(dict(group=group, token=int(token), original_pool=ids[token].tolist(),
                                       replacement_pool=[*ids[token][:-1].tolist(), extra]))
            with tempfile.TemporaryDirectory() as tmp:
                runner = probe.FrontierProbe(model, tmp, layer=0)
                runner.start_sequence(dict(prompt_sha256='toy', sequence=0, model_source_sha256='toy', seed=21))
                state = torch.random.get_rng_state().clone(); copied=inputs.clone()
                with patch.object(probe,'source_candidates',return_value=(candidates,None)), patch.object(
                        probe,'select_sources',return_value=(candidates,'strict_matched')):
                    with patch.object(model, 'forward', wraps=model.forward) as forward:
                        runner.run(SimpleNamespace(last_record=frame,layer_ids=[0]),kwargs,baseline,lambda x,_:x,.95,1)
                    self.assertEqual(forward.call_count,5)
                self.assertTrue(torch.equal(torch.random.get_rng_state(),state))
                self.assertTrue(torch.equal(inputs,copied))
                self.assertTrue(all(torch.count_nonzero(t)==0 for t in cache[0]))
                self.assertIsNone(runner.trial)
                row=json.loads(next(Path(tmp).glob('*.jsonl')).read_text())
                self.assertEqual(row['status'],'matched')
                self.assertEqual(row['sham_logit_max_abs'],0)
                for side in ('strong','weak'):
                    self.assertTrue(row[side]['control_decisions_equal'])
                    self.assertIn('relative_to_control',row[side])
                # Idle hooks do not modify the normal path after all replays.
                torch.testing.assert_close(model(**kwargs),baseline)
                # Even an interrupted replay must clear trial hooks and restore RNG.
                rng = torch.random.get_rng_state().clone()
                with patch.object(probe,'source_candidates',return_value=(candidates,None)), patch.object(
                        probe,'select_sources',return_value=(candidates,'strict_matched')):
                    def fail(_output, _model):
                        raise RuntimeError('synthetic replay failure')
                    with self.assertRaisesRegex(RuntimeError,'synthetic replay failure'):
                        runner.run(SimpleNamespace(last_record=frame,layer_ids=[0]),kwargs,baseline,fail,.95,1)
                self.assertIsNone(runner.trial)
                self.assertTrue(torch.equal(torch.random.get_rng_state(),rng))


if __name__ == '__main__':
    unittest.main()
