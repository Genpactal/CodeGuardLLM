"""Cloud checkpoint recovery and notebook failures, without a Colab account."""

from contextlib import redirect_stdout
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from codeguard.cloud import Checkpoint, extract_archive, select_dtype

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'notebooks/watermark_comparison_colab.ipynb').read_text(encoding='utf-8'))


class CloudTests(unittest.TestCase):
    def test_checkpoint_recovers_complete_records_without_locks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'session/watermark_comparison'
            (run / 'delta-1').mkdir(parents=True)
            (run / 'delta-1/generations.jsonl').write_bytes(b'{"sample":1}\n{"sample":')
            (run / 'delta-1/.lock').write_bytes(b'0')
            (run / 'delta-1/manifest.json').write_text('{"expected":2}')
            backup = root / 'drive/results.zip'
            checkpoint = Checkpoint(run, backup, root / 'scratch')
            checkpoint.save(force=True)
            restored = root / 'new-session/watermark_comparison'
            resumed = Checkpoint(restored, backup, root / 'scratch')
            self.assertTrue(resumed.restore())
            self.assertEqual((restored / 'delta-1/generations.jsonl').read_bytes(), b'{"sample":1}\n')
            self.assertFalse((restored / 'delta-1/.lock').exists())
            self.assertEqual(json.loads((restored / 'delta-1/manifest.json').read_text()), {'expected': 2})
            (restored / 'delta-1/generations.jsonl').write_bytes(b'{"sample":2}\n')
            self.assertFalse(resumed.restore())
            self.assertEqual((restored / 'delta-1/generations.jsonl').read_bytes(), b'{"sample":2}\n')

    def test_archive_rejects_escaping_members_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('watermark_comparison/ok.txt', 'ok')
                bundle.writestr('../outside.txt', 'bad')
            with self.assertRaises(ValueError):
                extract_archive(archive, root / 'destination', 'watermark_comparison')
            self.assertFalse((root / 'destination/watermark_comparison/ok.txt').exists())
            self.assertFalse((root / 'outside.txt').exists())

    def test_native_precision_selection(self):
        self.assertEqual(select_dtype(7, True), 'float32')
        self.assertEqual(select_dtype(8, False), 'float32')
        self.assertEqual(select_dtype(8, True), 'bfloat16')

    def test_notebook_syntax_and_visible_subprocess_failure_with_backup(self):
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            class Backup:
                def save(self, force=False):
                    calls.append(force)
            namespace = dict(PROJECT=Path(temp), PYTHON=sys.executable, CHECKPOINT=Backup())
            runner = ''.join(next(c['source'] for c in NOTEBOOK['cells'] if c['id'] == 'runner'))
            exec(runner, namespace)
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'real error from child'):
                    namespace['run']('-c', "import sys; print('real error from child'); sys.exit(1)")
            self.assertIn(True, calls)


if __name__ == '__main__':
    unittest.main()
