"""Robustness accounting uses synthetic scores, not experimental claims."""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

from codeguard.attacks import ATTACKS, transform
from codeguard.config import load_config
from codeguard.evaluation import DockerEvaluator
from codeguard.pipeline import generate
from codeguard.robustness import evaluate, prepare, report, score
from codeguard.storage import read_records
from test_analysis import ANALYSIS_AVAILABLE

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = '''def answer(n=5):
    """Sum integers below n."""
    # comment containing a watermark-like sequence
    total = 0
    for index in range(n):
        total += index
    return total
'''


class TransformationTests(unittest.TestCase):
    def test_notebook_cells_compile_and_expensive_stages_are_opt_in(self):
        for filename in ('robustness.ipynb', 'kgw_robustness.ipynb'):
            with self.subTest(notebook=filename):
                nb = json.loads((ROOT / 'notebooks' / filename).read_text(encoding='utf-8'))
                cells = {c['id']: ''.join(c['source']) for c in nb['cells']}
                for cell in nb['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(cell['source'])
                    if any(line.lstrip().startswith(('%', '!')) for line in source.splitlines()):
                        try:
                            from IPython.core.inputtransformer2 import TransformerManager
                        except ImportError:
                            self.skipTest('Install the notebook extra to validate IPython syntax')
                        source = TransformerManager().transform_cell(source)
                    ast.parse(source)
                calls = []
                namespace = dict(RUN_SCORING=False, RUN_CORRECTNESS=False, RUN_DIRS=['source'],
                                 SCORE_OPTIONS=[], run=lambda *a: calls.append(a))
                exec(cells['score'], namespace)
                exec(cells['evaluate'], namespace)
                self.assertEqual(calls, [])

    def test_transformations_preserve_authored_fixture_signature_and_outputs(self):
        # Only this authored fixture is executed here; saved model code uses Docker.
        for attack in ATTACKS:
            with self.subTest(attack=attack):
                candidate = transform(EXAMPLE, attack)
                self.assertEqual(candidate['changed'], attack != 'identity')
                namespace = {}
                exec(candidate['code'], namespace)
                self.assertEqual(namespace['answer'](n=5), 10)
                self.assertEqual(namespace['answer'](0), 0)
                self.assertEqual(ast.dump(ast.parse(candidate['code']).body[0].args),
                                 ast.dump(ast.parse(EXAMPLE).body[0].args))

    def test_comments_do_not_touch_strings_or_unicode(self):
        code = 'def f():\n    value = "café # literal" # actual comment\n    return value\n'
        candidate = transform(code, 'strip_comments')['code']
        self.assertIn('café # literal', candidate)
        self.assertNotIn('actual comment', candidate)
        self.assertEqual(ast.dump(ast.parse(code)), ast.dump(ast.parse(candidate)))

    def test_docstring_only_bodies_stay_valid(self):
        result = transform('"""module"""\nclass C:\n    """class"""\n    def f(self):\n        """function"""', 'strip_docstrings')
        compile(result['code'], '<fixture>', 'exec')
        self.assertNotIn('"""', result['code'])

    def test_rename_skips_complex_scopes_and_reflection(self):
        bodies = ['    x = 1\n    return locals()\n',
                  '    x = [i for i in range(3)]\n    return x\n',
                  '    x = 1\n    def inner():\n        return x\n    return inner()\n',
                  '    global x\n    x = 1\n    return x\n',
                  '    x = 1\n    return f"{x=}"\n']
        for body in bodies:
            source = 'def f():\n' + body
            self.assertEqual(transform(source, 'rename_locals')['code'], source)

    def test_rename_avoids_collisions_and_preserves_parameters(self):
        source = 'def f(cg_local_0):\n    total = cg_local_0 + 1\n    return total\n'
        code = transform(source, 'rename_locals')['code']
        self.assertIn('def f(cg_local_0)', code)
        self.assertIn('cg_local_1', code)
        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace['f'](cg_local_0=2), 3)

    def test_unavailable_invalid_and_no_op_are_explicit(self):
        self.assertEqual(transform(None, 'identity')['status'], 'source_unavailable')
        self.assertEqual(transform('def broken(', 'reformat')['status'], 'transform_error')
        source = 'def f():\n    return 1\n'
        self.assertFalse(transform(source, 'rename_locals')['changed'])
        self.assertFalse(transform(source, 'strip_comments')['changed'])


class Backend:
    metadata = {'backend': 'robustness-test'}

    def __init__(self, config, load_model=True):
        pass

    def generate(self, prompt, task_id, seed, condition):
        name = prompt.split('def ', 1)[1].split('(', 1)[0]
        code = f'def {name}(*args):\n    """Doc."""\n    # watermark\n    result = 1\n    return result\n'
        return dict(raw_text=code, generated_token_ids=[1, 2], generated_tokens=2,
                    prompt_tokens=5, sample_seed=seed, generation_seconds=.1, finish_reason='eos')

    def score(self, text):
        if 'Doc.' not in text:
            return dict(status='no_eligible_tokens', z_score=None, scored_tokens=0)
        return dict(status='ok', z_score=6.0 if '# watermark' in text else 0.0,
                    scored_tokens=20, is_watermarked='# watermark' in text)


class Evaluator:
    metadata = {'image_id': 'synthetic-evaluator'}

    def __init__(self, settings):
        pass

    def evaluate(self, code, task):
        return dict(status='failed' if 'cg_local_' in code else 'passed', seconds=.01)


@unittest.skipUnless(ANALYSIS_AVAILABLE, 'Install .[analysis]')
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.output = self.root / 'source', self.root / 'robustness'
        config = load_config(ROOT / 'configs/pilot.yaml')
        config['dataset']['limit'] = 2
        config['output_dir'] = str(self.source)
        generate(config, Backend, progress=lambda _: None)
        self.original = (self.source / 'generations.jsonl').read_bytes()
        prepare(self.source, self.output, progress=lambda _: None)

    def score(self, factory=Backend, **kwargs):
        return score(self.source, self.output, ROOT / '.deps/MarkLLM', backend_factory=factory,
                     progress=lambda _: None, **kwargs)

    def test_end_to_end_denominators_and_correctness_preserving_evasions(self):
        self.assertEqual(report(self.source, self.output)['status'], 'pending')
        self.assertEqual(self.score(), 20)
        pending = report(self.source, self.output)
        self.assertTrue(all(g['pass_at_1'] is None for g in pending['groups']))
        self.assertEqual(evaluate(self.source, self.output, Evaluator, progress=lambda _: None), 20)
        summary = report(self.source, self.output)
        self.assertEqual(summary['status'], 'complete')
        groups = {g['attack']: g for g in summary['groups'] if g['condition'] == 'watermarked'}
        self.assertEqual(groups['identity']['detected_rate'], 1)
        self.assertEqual(groups['identity']['correct_evasion_rate'], 0)
        stripped = groups['strip_comments']
        self.assertEqual(stripped['correct_evasions'], 2)
        self.assertEqual(stripped['correct_evasion_rate'], 1)
        self.assertEqual(stripped['correctness_retention'], 1)
        self.assertEqual(stripped['mean_paired_z_change'], -6)
        self.assertEqual(groups['rename_locals']['pass_at_1'], 0)
        self.assertEqual(groups['rename_locals']['correct_evasions'], 0)
        self.assertEqual(groups['strip_docstrings']['unscored'], 2)
        self.assertEqual(groups['strip_docstrings']['correct_evasions'], 0)
        self.assertIsNone(groups['strip_docstrings']['auroc_vs_clean'])
        self.assertEqual(self.original, (self.source / 'generations.jsonl').read_bytes())
        self.assertTrue((self.output / 'report.md').exists())
        self.assertTrue((self.output / 'robustness.png').exists())

    def test_no_baseline_detections_is_unavailable_not_perfect_evasion(self):
        class Undetected(Backend):
            def score(self, text):
                return dict(status='ok', z_score=0, scored_tokens=20)
        self.score(Undetected)
        evaluate(self.source, self.output, Evaluator, progress=lambda _: None)
        summary = report(self.source, self.output)
        for group in summary['groups']:
            self.assertEqual(group['eligible_correct_detected'], 0)
            self.assertIsNone(group['correct_evasion_rate'])
            self.assertIsNone(group['detection_retention'])

    def test_evaluation_resumes_without_repeating_references_or_completed_variants(self):
        class Interrupted(Evaluator):
            calls = 0
            def evaluate(self, code, task):
                self.calls += 1
                if self.calls == 4:
                    raise RuntimeError('interrupted')
                return super().evaluate(code, task)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            evaluate(self.source, self.output, Interrupted, progress=lambda _: None)
        self.assertEqual(len(read_records(self.output / 'evaluations.jsonl')), 1)
        class Counting(Evaluator):
            calls = 0
            def evaluate(self, code, task):
                type(self).calls += 1
                return super().evaluate(code, task)
        self.assertEqual(evaluate(self.source, self.output, Counting, progress=lambda _: None), 20)
        self.assertEqual(Counting.calls, 19)

    def test_interrupted_scoring_repairs_tail_and_does_not_repeat_completed_items(self):
        class Interrupted(Backend):
            calls = 0
            def score(self, text):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('interrupted')
                return super().score(text)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.score(Interrupted)
        path = self.output / 'scores.jsonl'
        first = path.read_bytes()
        with path.open('ab') as handle:
            handle.write(b'{"incomplete":')
        self.assertEqual(self.score(), 20)
        self.assertTrue(path.read_bytes().startswith(first))
        def no_load(*args, **kwargs):
            raise AssertionError('Already scored; must not load model')
        self.assertEqual(self.score(no_load), 20)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.score(device='cpu', dtype='float32')

    def test_preparation_resume_and_source_or_variant_tampering_rejected(self):
        self.assertEqual(prepare(self.source, self.output, progress=lambda _: None), 20)
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare(self.source, self.output, ['reformat'])
        path = self.output / 'variants.jsonl'
        rows = read_records(path)
        rows[0]['code'] += '\n# tampered'
        path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.score()

    def test_detector_and_evaluator_environment_mismatch_rejected(self):
        class Interrupted(Backend):
            def score(self, text):
                raise RuntimeError('interrupt')
        with self.assertRaises(RuntimeError):
            self.score(Interrupted)
        class Different(Backend):
            metadata = {'backend': 'changed'}
        with self.assertRaisesRegex(ValueError, 'environment changed'):
            self.score(Different)
        evaluate(self.source, self.output, Evaluator, progress=lambda _: None)
        class DifferentEvaluator(Evaluator):
            metadata = {'image_id': 'changed'}
        with self.assertRaisesRegex(ValueError, 'environment changed'):
            evaluate(self.source, self.output, DifferentEvaluator)

    def test_failed_reference_stops_before_testing_candidates(self):
        class FailedReference(Evaluator):
            def evaluate(self, code, task):
                return dict(status='failed', seconds=.01)
        with self.assertRaisesRegex(RuntimeError, 'Reference solution failed'):
            evaluate(self.source, self.output, FailedReference)
        self.assertFalse((self.output / 'evaluations.jsonl').exists())

    def test_source_changes_and_foreign_stage_manifest_rejected(self):
        self.score()
        path = self.output / 'scores_manifest.json'
        data = json.loads(path.read_text())
        data['robustness_id'] = 'foreign'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Foreign'):
            report(self.source, self.output)
        path = self.source / 'generations.jsonl'
        rows = read_records(path)
        rows[0]['raw_text'] += '\n# changed'
        path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare(self.source, self.output)


@unittest.skipUnless(os.environ.get('CODEGUARD_DOCKER_TESTS') == '1', 'Opt in to Docker smoke tests')
class DockerTransformationTests(unittest.TestCase):
    def test_transformed_authored_programs_in_real_sandbox(self):
        evaluator = DockerEvaluator(load_config(ROOT / 'configs/pilot.yaml')['evaluation'])
        task = dict(entry_point='answer', test='def check(candidate):\n    assert candidate(n=5) == 10\n    assert candidate(0) == 0\n')
        for attack in ATTACKS:
            with self.subTest(attack=attack):
                self.assertEqual(evaluator.evaluate(transform(EXAMPLE, attack)['code'], task)['status'], 'passed')
