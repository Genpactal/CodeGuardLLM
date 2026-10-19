"""Check matched baselines, resumability, and calibration/selection separation offline."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from codeguard.evaluation import evaluate_run, load_run
from codeguard.storage import digest, read_records
from codeguard.tuning import (analyze_suite, calibrated_threshold, candidates, controls_suite,
                             generate_suite, load_suite, prepare_suite, suite_configs)
from test_pipeline import FakeBackend, FakeEvaluator

ROOT = Path(__file__).resolve().parents[1]


class ScoredBackend(FakeBackend):
    calls_log = []
    runtime = 'same'
    fail_after = None

    def __init__(self, config, load_model=True):
        super().__init__(config)
        self.metadata = {'backend': 'test-double', 'runtime': type(self).runtime}
        self.algorithm = config['watermark']['algorithm']
        self.fail_after = type(self).fail_after

    def generate(self, *args):
        result = super().generate(*args)
        self.calls_log.append(args)
        return result

    def score(self, text):
        z = 2.0 if self.algorithm == 'SWEET' else 0.0
        return dict(status='ok', z_score=z, is_watermarked=False, scored_tokens=10)


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'tuning'
        self.config = ROOT / 'configs/pilot.yaml'
        ScoredBackend.calls_log = []
        ScoredBackend.runtime = 'same'
        ScoredBackend.fail_after = None
        prepare_suite(self.config, self.output, task_count=4, calibration_count=2)

    def test_grid_is_staged_and_split_stays_fixed(self):
        self.assertEqual(len(candidates()), 9)
        self.assertEqual(len(candidates('full')), 17)
        suite, tasks = load_suite(self.output)
        self.assertFalse(set(suite['calibration_task_ids']) & set(suite['selection_task_ids']))
        self.assertEqual(len(tasks), 4)
        self.assertEqual(prepare_suite(self.config, self.output, 4, 2), suite)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            prepare_suite(self.config, self.output, 4, 2, seeds=[17])

    def test_clean_is_generated_once_and_scored_by_each_detector(self):
        generate_suite(self.output, backend_factory=ScoredBackend)
        self.assertEqual(len(ScoredBackend.calls_log), 4 * (9 + 1))
        self.assertEqual(sum(args[-1] == 'clean' for args in ScoredBackend.calls_log), 4)
        runs = [load_run(c['output_dir']) for c in suite_configs(self.output).values()]
        first_clean = [r for r in runs[0][2] if r['condition'] == 'clean']
        last_clean = [r for r in runs[-1][2] if r['condition'] == 'clean']
        self.assertEqual([r['raw_text'] for r in first_clean], [r['raw_text'] for r in last_clean])
        self.assertNotEqual(first_clean[0]['code_detection'], last_clean[0]['code_detection'])
        generate_suite(self.output, backend_factory=ScoredBackend)
        self.assertEqual(len(ScoredBackend.calls_log), 40)

    def test_partial_run_resumes_and_rejects_another_runtime(self):
        ScoredBackend.fail_after = 1
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            generate_suite(self.output, backend_factory=ScoredBackend)
        ScoredBackend.fail_after = None
        ScoredBackend.runtime = 'different'
        with self.assertRaisesRegex(ValueError, 'environment changed'):
            generate_suite(self.output, backend_factory=ScoredBackend)
        ScoredBackend.runtime = 'same'
        generate_suite(self.output, backend_factory=ScoredBackend)
        self.assertEqual(sum(args[-1] == 'clean' for args in ScoredBackend.calls_log), 4)

    def test_cache_and_task_tampering_are_rejected(self):
        ScoredBackend.fail_after = 1
        with self.assertRaises(RuntimeError):
            generate_suite(self.output, backend_factory=ScoredBackend)
        ScoredBackend.fail_after = None
        path = self.output / 'shared_clean/generations.jsonl'
        rows = read_records(path)
        rows[0]['output']['raw_text'] = 'changed'
        path.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'modified clean cache'):
            generate_suite(self.output, backend_factory=ScoredBackend)
        (self.output / 'tasks.jsonl').write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Derived task'):
            suite_configs(self.output)

    def test_threshold_is_strict_and_satisfies_both_null_groups(self):
        for groups in ([[1] * 20, [2] * 20], [list(range(20)), list(range(40))]):
            threshold = calibrated_threshold(groups, 0.05)
            for group in groups:
                self.assertLessEqual(sum(z > threshold for z in group) / len(group), 0.05)
        self.assertEqual(calibrated_threshold([[1, 2], [2, 3]], 0.05), 3)
        with self.assertRaises(ValueError):
            calibrated_threshold([[], [1]], 0.05)

    def test_report_keeps_correctness_pending_and_excludes_calibration_outcomes(self):
        generate_suite(self.output, backend_factory=ScoredBackend)
        controls_suite(self.output, backend_factory=ScoredBackend)
        before = analyze_suite(self.output)
        self.assertEqual(before['status'], 'correctness_pending')
        self.assertFalse(before['exploratory_shortlist'])
        self.assertTrue(all(row['correct_and_detected_rate'] is None for row in before['metrics']))
        self.assertTrue(all(row['selection_samples'] == 2 for row in before['metrics']))
        # Seven detectors: KGW gamma x3 and SWEET entropy x4; strengths share controls.
        self.assertEqual(len(list((self.output / 'controls').glob('*/manifest.json'))), 7)
        for config in suite_configs(self.output).values():
            evaluate_run(config['output_dir'], FakeEvaluator, progress=lambda _: None)
        after = analyze_suite(self.output)
        self.assertEqual(after['status'], 'selection_evaluated')
        self.assertTrue(all(row['correct_and_detected_rate'] == 0 for row in after['metrics']))
        # Poison a calibration watermarked score: threshold and selection metrics must not change.
        suite, _ = load_suite(self.output)
        config = next(iter(suite_configs(self.output).values()))
        path = Path(config['output_dir']) / 'generations.jsonl'
        rows = read_records(path)
        for row in rows:
            if row['task_id'] in suite['calibration_task_ids'] and row['condition'] == 'watermarked':
                row['code_detection']['z_score'] = 10000
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        eval_path = path.with_name('evaluations.jsonl')
        evaluations = read_records(eval_path)
        lookup = {r['sample_id']: r for r in rows}
        for evaluation in evaluations:
            evaluation['generation_sha256'] = digest(lookup[evaluation['sample_id']])
        eval_path.write_text(''.join(json.dumps(r) + '\n' for r in evaluations), encoding='utf-8')
        self.assertEqual(analyze_suite(self.output)['metrics'], after['metrics'])


class TuningNotebookTests(unittest.TestCase):
    def test_all_cells_and_embedded_scripts_parse(self):
        notebook = json.loads((ROOT / 'notebooks/watermark_tuning_colab.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant) and node.args[0].value == '-c':
                    ast.parse(ast.literal_eval(node.args[1]))

    def test_upload_archive_contains_the_new_workflow_without_results(self):
        import zipfile
        from scripts.package_colab import package
        with tempfile.TemporaryDirectory() as folder:
            archive = package(ROOT, Path(folder) / 'tuning.zip', tuning=True)
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                self.assertIn('CodeGuardLLM-TUNING/src/codeguard/tuning.py', names)
                self.assertIn('CodeGuardLLM-TUNING/scripts/tune_watermarks.py', names)
                self.assertTrue(all(name.startswith('CodeGuardLLM-TUNING/') for name in names))
                self.assertFalse(any('/runs/' in name or '/.venv/' in name for name in names))
                notebook = json.loads(bundle.read('CodeGuardLLM-TUNING/notebooks/watermark_tuning_colab.ipynb'))
                self.assertTrue(all(not c.get('outputs') for c in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
