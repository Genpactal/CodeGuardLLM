"""Check notebook setup recovery and visible errors without a cloud session."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / 'notebooks/pilot_results.ipynb'
NOTEBOOK = json.loads(ARCHIVE_PATH.read_text(encoding='utf-8')) if ARCHIVE_PATH.is_file() else None


def source(identity):
    if NOTEBOOK is None:
        raise unittest.SkipTest('Archived cloud notebook is not present')
    return ''.join(next(c for c in NOTEBOOK['cells'] if c['id'] == identity)['source'])


class NotebookTests(unittest.TestCase):
    def test_colab_commands_use_headless_backend_without_changing_kernel(self):
        for experiment in ('watermark_comparison_colab', 'sweet_comparison_colab', 'watermark_tuning_colab'):
            with self.subTest(notebook=experiment), tempfile.TemporaryDirectory() as folder:
                notebook = json.loads((ROOT / f'notebooks/{experiment}.ipynb').read_text(encoding='utf-8'))
                code = ''.join(next(c['source'] for c in notebook['cells'] if c['id'] == 'runner'))
                namespace = dict(PROJECT=Path(folder), CHECKPOINT=None)
                exec(code, namespace)
                inline = 'module://matplotlib_inline.backend_inline'
                with patch.dict(os.environ, {'MPLBACKEND': inline}), redirect_stdout(io.StringIO()):
                    namespace['command']([
                        sys.executable, '-c',
                        "import os; assert os.environ['MPLBACKEND'] == 'Agg', os.environ['MPLBACKEND']",
                    ], label='headless-backend')
                    self.assertEqual(os.environ['MPLBACKEND'], inline)

    def test_local_generation_does_not_require_cuda(self):
        for experiment in ('watermark_comparison', 'attribution'):
            notebook = json.loads((ROOT / f'notebooks/{experiment}.ipynb').read_text(encoding='utf-8'))
            code = ''.join(next(c['source'] for c in notebook['cells'] if c['id'] == 'generate'))
            calls = []
            namespace = dict(READY=True, CONFIG=ROOT / 'configs/local.yaml',
                             RUN_DIR=ROOT / f'runs/local/humaneval/{experiment}', run=lambda *a, **k: calls.append(a))
            exec(code, namespace)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 'generate')
            self.assertEqual(calls[0][3], ROOT / 'configs/local.yaml')

    def test_watermark_comparison_and_attribution_cells_are_valid_python(self):
        for experiment in ('watermark_comparison', 'attribution'):
            notebook = json.loads((ROOT / f'notebooks/{experiment}.ipynb').read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))

    def test_cells_are_valid_python_and_no_drive_auth(self):
        if NOTEBOOK is None:
            self.skipTest('Archived cloud notebook is not present')
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                text = ''.join(cell['source'])
                ast.parse(text)
                self.assertNotIn('drive.mount(', text)
                self.assertNotIn('files.download(', text)

    def test_command_failure_includes_stderr_and_log(self):
        with tempfile.TemporaryDirectory() as folder:
            namespace = {'PROJECT': Path(folder)}
            exec(source('runner'), namespace)
            output = io.StringIO()
            with redirect_stdout(output):
                namespace['command']([sys.executable, '-c', "print('progress visible')"], label='success')
                with self.assertRaisesRegex(RuntimeError, 'actual failure reason'):
                    namespace['command']([sys.executable, '-c',
                        "import sys; print('actual failure reason', file=sys.stderr); sys.exit(1)"], label='failure')
            self.assertIn('progress visible', output.getvalue())
            logs = list(namespace['LOG_DIR'].glob('failure-*.log'))
            self.assertEqual(len(logs), 1)
            self.assertIn('actual failure reason', logs[0].read_text())

    def install_case(self, missing_pip, fail_gpu=False):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            logs = project / 'logs'
            logs.mkdir()
            calls = []

            def command(args, label):
                calls.append(label)
                if label == 'create-environment':
                    self.assertIn('--without-pip', args)
                    env = project / '.venv'
                    (env / 'bin').mkdir(parents=True)
                    (env / 'bin/python').touch()
                    (env / 'pyvenv.cfg').touch()

            def run(*args, label):
                calls.append(label)
                if label == 'check-gpu' and fail_gpu:
                    raise RuntimeError('No GPU')

            downloads = []
            namespace = dict(PROJECT=project, LOG_DIR=logs, Path=Path, sys=sys,
                CONFIG=project / 'configs/pilot_fp32.yaml',
                command=command, run=run,
                subprocess=SimpleNamespace(run=lambda *a, **k: SimpleNamespace(returncode=int(missing_pip))),
                urllib=SimpleNamespace(request=SimpleNamespace(urlretrieve=lambda url, path: downloads.append(url))))
            with redirect_stdout(io.StringIO()):
                if fail_gpu:
                    with self.assertRaisesRegex(RuntimeError, 'No GPU'):
                        exec(source('install'), namespace)
                else:
                    exec(source('install'), namespace)
            self.assertEqual(namespace['READY'], not fail_gpu)
            self.assertEqual('install-pip' in calls, missing_pip)
            self.assertEqual(bool(downloads), missing_pip)
            if downloads:
                self.assertEqual(downloads, ['https://bootstrap.pypa.io/get-pip.py'])
            return calls

    def test_missing_pip_repaired_before_dependencies(self):
        calls = self.install_case(missing_pip=True)
        self.assertLess(calls.index('install-pip'), calls.index('install-dependencies'))

    def test_working_pip_is_reused(self):
        self.install_case(missing_pip=False)

    def test_gpu_failure_blocks_readiness(self):
        self.install_case(missing_pip=False, fail_gpu=True)

    def test_summary_and_export_use_same_run_without_regeneration(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            project = base / 'CodeGuardLLM'
            run_dir = project / 'runs/pilot-fp32'
            run_dir.mkdir(parents=True)
            (run_dir / 'manifest.json').write_text('{}')
            (run_dir / 'generations.jsonl').write_text('{"sample": 1}\n')
            calls = []

            def run(*args, label):
                calls.append(args)
                self.assertEqual(args[2], 'summarize')
                self.assertEqual(args[-1], run_dir)
                (run_dir / 'summary.json').write_text(json.dumps(
                    {'generated_samples': 20, 'expected_samples': 20}))

            import shutil
            import zipfile
            namespace = dict(PROJECT=project, BASE=base, RUN_DIR=run_dir,
                             run=run, json=json, shutil=shutil)
            with redirect_stdout(io.StringIO()):
                exec(source('summary'), namespace)
                exec(source('export'), namespace)
            self.assertEqual(len(calls), 1)
            self.assertIsInstance(namespace['PROJECT'], Path)
            with zipfile.ZipFile(namespace['EXPORT']) as archive:
                self.assertIn('pilot-fp32/generations.jsonl', archive.namelist())
                self.assertIn('pilot-fp32/summary.json', archive.namelist())


if __name__ == '__main__':
    unittest.main()
