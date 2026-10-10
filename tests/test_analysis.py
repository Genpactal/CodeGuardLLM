"""Known-score calculations and reports; these do not measure model performance."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from codeguard.analysis import attribution_metrics, attribution_report, choose_key, detection_report, detection_statistics
from codeguard.experiments import comparison_configs, attribution_configs
from codeguard.pipeline import generate
from test_pipeline import FakeBackend

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_AVAILABLE = all(importlib.util.find_spec(name) for name in ('sklearn', 'matplotlib'))


class ScoredBackend(FakeBackend):
    def __init__(self, config, load_model=True):
        super().__init__(config)

    def score(self, text):
        return {'status': 'ok', 'z_score': 2.0, 'is_watermarked': False, 'scored_tokens': 10}


class AttributionTests(unittest.TestCase):
    def test_unknown_for_missing_tied_or_low_scores(self):
        self.assertEqual(choose_key({'a': 5, 'b': 1}, 4), ('a', 'a'))
        self.assertEqual(choose_key({'a': 4, 'b': 1}, 4), ('a', 'unknown'))
        for scores in ({'a': 5, 'b': 5}, {'a': 8, 'b': None}, {'a': float('nan')}):
            self.assertEqual(choose_key(scores, 4), ('unknown', 'unknown'))

    def test_unknowns_stay_in_accuracy_denominator(self):
        records = [
            dict(source='a', condition='watermarked', prediction='a', forced_prediction='a', scored=True),
            dict(source='a', condition='watermarked', prediction='unknown', forced_prediction='a', scored=True),
            dict(source='b', condition='watermarked', prediction='unknown', forced_prediction='unknown', scored=False),
            dict(source='b', condition='clean', prediction='a', forced_prediction='a', scored=True),
            dict(source='a', condition='clean', prediction='unknown', forced_prediction='b', scored=True),
        ]
        stats = attribution_metrics(records, ['a', 'b'])
        self.assertEqual(stats['forced_accuracy'], 2 / 3)
        self.assertEqual(stats['thresholded_accuracy'], 1 / 3)
        self.assertEqual(stats['clean_false_attribution_rate'], 0.5)
        self.assertEqual(stats['confusion_matrix']['b']['unknown'], 1)
        self.assertEqual(sum(sum(r.values()) for r in stats['confusion_matrix'].values()), 5)

    def test_configs_are_independent_and_float32(self):
        configs = comparison_configs(ROOT / 'configs/local.yaml', 'unused')
        before = deepcopy(configs['delta-2'])
        configs['delta-1']['watermark']['delta'] = 9
        self.assertEqual(configs['delta-2'], before)
        models = attribution_configs(ROOT / 'configs/local.yaml', 'unused')
        self.assertEqual(len({c['watermark']['hash_key'] for c in models.values()}), 3)
        self.assertTrue(all(c['model']['dtype'] == 'float32' for c in models.values()))

    def test_gpu_comparison_preserves_explicit_precision(self):
        configs = comparison_configs(ROOT / 'configs/gpu.yaml', 'unused')
        self.assertTrue(all(c['model']['device'] == 'cuda' and c['model']['dtype'] == 'bfloat16'
                            for c in configs.values()))

    def test_local_experiments_keep_cpu_for_every_model(self):
        for factory in (comparison_configs, attribution_configs):
            configs = factory(ROOT / 'configs/local.yaml', 'unused')
            self.assertTrue(all(c['model']['device'] == 'cpu' and c['model']['dtype'] == 'float32'
                                for c in configs.values()))


@unittest.skipUnless(ANALYSIS_AVAILABLE, 'Install .[analysis] for reports')
class ReportTests(unittest.TestCase):
    def test_auc_and_unscored_denominators(self):
        rows = []
        for condition, values in [('clean', [0, 1, None]), ('watermarked', [2, 3])]:
            for z in values:
                rows.append({'condition': condition, 'code_detection': None if z is None else
                             {'status': 'ok', 'z_score': z, 'scored_tokens': 10}})
        stats, auc, _ = detection_statistics(rows, 2)
        self.assertEqual(auc, 1.0)
        self.assertEqual(stats[0]['unscored'], 1)
        self.assertEqual(stats[1]['positive_rate'], 0.5)  # strictly greater than threshold
        for row in rows:
            if row['code_detection']:
                row['code_detection']['z_score'] = 0
        self.assertEqual(detection_statistics(rows, 4)[1], 0.5)

    def create_runs(self, configs):
        for config in configs.values():
            config['dataset']['limit'] = 2
            generate(config, ScoredBackend, progress=lambda _: None)
        return {name: c['output_dir'] for name, c in configs.items()}

    def test_detection_report_and_incomplete_run_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            configs = comparison_configs(ROOT / 'configs/pilot.yaml', folder)
            paths = self.create_runs(configs)
            output = Path(folder) / 'analysis'
            metrics = detection_report(paths, output)
            self.assertEqual(len(metrics), 6)
            self.assertTrue(all(row['pass_at_1'] is None for row in metrics))
            self.assertGreater((output / 'detection.png').stat().st_size, 100)
            self.assertEqual(json.loads((output / 'metrics.json').read_text()), metrics)
            generations = Path(paths['delta-1']) / 'generations.jsonl'
            generations.write_text(generations.read_text().splitlines()[0] + '\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete run'):
                detection_report(paths, output)

    def test_attribution_scores_all_candidates_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            configs = attribution_configs(ROOT / 'configs/pilot.yaml', folder)
            paths = self.create_runs(configs)
            calls = []
            class Detector(ScoredBackend):
                def __init__(self, config, load_model=True):
                    if load_model:
                        raise AssertionError('Attribution must not load model weights')
                    super().__init__(config)
                def score(self, text):
                    calls.append(text)
                    return super().score(text)
            output = Path(folder) / 'analysis'
            result = attribution_report(paths, output, ROOT / '.deps/MarkLLM', backend_factory=Detector)
            self.assertEqual(len(calls), 36)  # 12 samples x 3 candidate keys
            self.assertEqual(result['thresholded_accuracy'], 0)
            self.assertEqual(result['clean_false_attribution_rate'], 0)
            self.assertEqual(sum(sum(r.values()) for r in result['confusion_matrix'].values()), 12)
            self.assertTrue((output / 'confusion.png').is_file())
            class WrongEnvironment(Detector):
                metadata = {'backend': 'changed'}
            with self.assertRaisesRegex(ValueError, 'environment'):
                attribution_report(paths, output, ROOT, backend_factory=WrongEnvironment)


if __name__ == '__main__':
    unittest.main()
