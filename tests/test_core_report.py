"""Core workflow checks with explicit synthetic backends, never research results."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from codeguard.controls import load_reference_controls, reference_code, score_reference_controls
from codeguard.core_report import core_report, group_statistics, roc_statistics
from codeguard.evaluation import evaluate_run, load_run
from codeguard.experiments import comparison_configs
from codeguard.pipeline import generate
from codeguard.reporting import detected_correctness
from test_analysis import ANALYSIS_AVAILABLE, ScoredBackend
from test_pipeline import FakeEvaluator

ROOT = Path(__file__).resolve().parents[1]


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.configs = comparison_configs(ROOT / 'configs/pilot.yaml', self.root)
        for config in self.configs.values():
            config['dataset']['limit'] = 2
            generate(config, ScoredBackend, progress=lambda _: None)
        self.paths = {label: c['output_dir'] for label, c in self.configs.items()}
        self.controls = self.root / 'reference_controls'

    def score(self, factory=ScoredBackend):
        return score_reference_controls(self.paths['delta-1'], self.controls, ROOT / '.deps/MarkLLM',
                                        backend_factory=factory, progress=lambda _: None)

    def test_reference_reconstruction(self):
        task = {'task_id': 'test', 'prompt': 'def f():\n', 'canonical_solution': '    return 1\n'}
        self.assertEqual(reference_code(task), 'def f():\n    return 1\n')
        self.assertEqual(reference_code(dict(task, reference_code='full code')), 'full code')

    def test_resume_reference_scores_and_reuse_across_delta(self):
        calls = []
        class Interrupted(ScoredBackend):
            def __init__(self, config, load_model=True):
                if load_model:
                    raise AssertionError('Reference scoring must not load weights')
                super().__init__(config)
            def score(self, text):
                calls.append(text)
                if len(calls) == 2:
                    raise RuntimeError('interrupted')
                return super().score(text)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.score(Interrupted)
        self.assertEqual(len((self.controls / 'scores.jsonl').read_text().splitlines()), 1)
        self.assertEqual(self.score(), 2)
        def no_load(*args, **kwargs):
            raise AssertionError('A complete controls run should not reload the detector')
        self.assertEqual(self.score(no_load), 2)
        for directory in self.paths.values():
            manifest, tasks, _ = load_run(directory)
            self.assertEqual(len(load_reference_controls(self.controls, manifest, tasks)), 2)
        changed = deepcopy(manifest)
        changed['config']['watermark']['hash_key'] += 1
        with self.assertRaisesRegex(ValueError, 'differ'):
            load_reference_controls(self.controls, changed, tasks)

    def test_missing_or_modified_control_rejected(self):
        manifest, tasks, _ = load_run(self.paths['delta-1'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            load_reference_controls(self.controls, manifest, tasks)
        self.score()
        path = self.controls / 'scores.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['code_sha256'] = 'changed'
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_reference_controls(self.controls, manifest, tasks)

    def test_runtime_mismatch_rejected(self):
        class WrongEnvironment(ScoredBackend):
            metadata = {'backend': 'wrong-device'}
        with self.assertRaisesRegex(ValueError, 'environment'):
            self.score(WrongEnvironment)

    @unittest.skipUnless(ANALYSIS_AVAILABLE, 'Install .[analysis]')
    def test_complete_report_and_pending_correctness(self):
        self.score()
        output = self.root / 'analysis'
        summary = core_report(self.paths, self.controls, output)
        self.assertEqual(summary['status'], 'correctness_pending')
        self.assertEqual(summary['reference_samples'], 2)  # not six independent controls
        self.assertEqual(len(summary['groups']), 9)
        self.assertTrue(all(r['pass_at_1_change'] is None for r in summary['comparison']))
        self.assertTrue(all(r['correct_and_detected_rate'] is None for r in summary['detected_correctness']))
        self.assertIn('correctness_pending', (output / 'report.md').read_text(encoding='utf-8'))
        for path in self.paths.values():
            evaluate_run(path, FakeEvaluator, progress=lambda _: None)
        summary = core_report(self.paths, self.controls, output)
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(all(r['pass_at_1_change'] == 0 for r in summary['comparison']))
        self.assertTrue(all(r['correct_and_detected_rate'] == 0 for r in summary['detected_correctness']))
        self.assertTrue(all(r['correctness_among_detected'] is None for r in summary['detected_correctness']))
        for name in ('controls_detection.png', 'tradeoff.png', 'roc.csv', 'length_analysis.csv', 'examples.json', 'detected_correctness.csv'):
            self.assertGreater((output / name).stat().st_size, 10)


class DetectedCorrectnessTests(unittest.TestCase):
    def test_threshold_and_denominators(self):
        rows = [{'sample_id': str(i), 'code_detection': None if z is None else {'status': 'ok', 'z_score': z}}
                for i, z in enumerate((5, 6, 4, None, float('nan')))]
        evaluations = {str(i): {'status': status} for i, status in enumerate(('passed', 'failed', 'passed', 'syntax_error', 'passed'))}
        stats = detected_correctness(rows, evaluations, 4, 5)
        self.assertEqual((stats['scored'], stats['detected'], stats['correct_detected']), (3, 2, 1))
        self.assertEqual(stats['correctness_among_detected'], .5)
        self.assertEqual(stats['correct_and_detected_rate'], .2)

    def test_no_detections_and_missing_tests_are_not_zero_conditional_accuracy(self):
        rows = [{'sample_id': 'a', 'code_detection': {'status': 'ok', 'z_score': 4}}]
        stats = detected_correctness(rows, {'a': {'status': 'passed'}}, 4, 1)
        self.assertIsNone(stats['correctness_among_detected'])
        self.assertEqual(stats['correct_and_detected_rate'], 0)
        rows[0]['code_detection']['z_score'] = 5
        stats = detected_correctness(rows, {}, 4, 1)
        self.assertEqual(stats['evaluated_detected'], 0)
        self.assertIsNone(stats['correctness_among_detected'])
        self.assertIsNone(stats['correct_and_detected_rate'])
        stats = detected_correctness(rows, {'a': {'status': 'passed'}}, 4, 2)
        self.assertIsNone(stats['correctness_among_detected'])
        self.assertIsNone(stats['correct_and_detected_rate'])


@unittest.skipUnless(ANALYSIS_AVAILABLE, 'Install .[analysis]')
class CoreStatisticsTests(unittest.TestCase):
    def test_empty_scores_and_known_roc(self):
        missing = [{'code_detection': None}]
        def row(z):
            return {'code_detection': {'status': 'ok', 'z_score': z, 'scored_tokens': 10}}
        self.assertIsNone(group_statistics(missing, 4)['positive_rate'])
        self.assertEqual(group_statistics([row(4), row(5), *missing], 4)['positive_rate'], .5)
        auc, points = roc_statistics([row(0), row(1)], [row(2), row(3)])
        self.assertEqual(auc, 1)
        self.assertIsNone(points[0]['threshold'])  # infinity must not enter JSON
        self.assertEqual(roc_statistics(missing, [row(2)]), (None, []))
