"""Exercise crash recovery and quality accounting without downloading a model."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from codeguard.config import load_config
from codeguard.data import load_tasks
from codeguard.evaluation import evaluate_run, load_run
from codeguard.extraction import extract_code
from codeguard.pipeline import generate
from codeguard.reporting import summarize
from codeguard.storage import append_record, read_records

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    metadata = {"backend": "test-double"}

    def __init__(self, config):
        self.calls = []
        self.fail_after = None

    def generate(self, prompt, task_id, seed, condition):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated interrupted session")
        self.calls.append((prompt, task_id, seed, condition))
        name = prompt.split("def ", 1)[1].split("(", 1)[0]
        raw = f"def {name}(*args):\n    return None\n" if condition == "clean" else "def broken("
        return {"raw_text": raw, "generated_token_ids": [1, 2], "generated_tokens": 2,
                "prompt_tokens": 5, "sample_seed": seed, "generation_seconds": 0.1, "finish_reason": "eos"}

    def score(self, text):
        return {"status": "ok", "z_score": 0.0, "is_watermarked": False}


class FakeEvaluator:
    metadata = {"image_id": "test-double"}

    def __init__(self, settings):
        self.calls = []

    def evaluate(self, code, task):
        self.calls.append((code, task))
        return {"status": "passed" if code == task["reference_code"] else "failed", "seconds": 0.01}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config(ROOT / "configs/week1.yaml")
        self.config["output_dir"] = self.temp.name
        self.config["dataset"]["limit"] = 2
        self.backend = FakeBackend(self.config)

    def run_generation(self):
        return generate(self.config, lambda _: self.backend, progress=lambda _: None)

    def test_interruption_resume_and_prompt_separation(self):
        self.backend.fail_after = 1
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_generation()
        records = Path(self.temp.name) / "generations.jsonl"
        with records.open("ab") as handle:
            handle.write(b'{"incomplete":')
        self.backend.fail_after = None
        self.assertEqual(self.run_generation(), 4)
        self.assertEqual(len(self.backend.calls), 4)
        self.assertEqual(len(read_records(records)), 4)
        self.assertEqual(self.run_generation(), 4)
        self.assertEqual(len(self.backend.calls), 4)
        expected_prompts = {task["prompt"] for task in load_tasks(self.config["dataset"])}
        self.assertTrue(all(call[0] in expected_prompts for call in self.backend.calls))
        self.assertTrue(list(Path(self.temp.name).glob("*.recovered-*")))

    def test_protocol_changes_rejected(self):
        self.run_generation()
        self.config["watermark"]["delta"] += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_generation()

    def test_backend_change_rejected_on_partial_run(self):
        self.backend.fail_after = 1
        with self.assertRaises(RuntimeError):
            self.run_generation()
        self.backend.fail_after = None
        self.backend.metadata = {"backend": "different-runtime"}
        with self.assertRaisesRegex(ValueError, "environment changed"):
            self.run_generation()

    def test_evaluation_resume_and_failure_denominator(self):
        self.run_generation()
        before = summarize(self.temp.name)
        self.assertIsNone(before["conditions"]["clean"]["pass_at_1"])
        evaluator = FakeEvaluator({})
        evaluate_run(self.temp.name, lambda _: evaluator, progress=lambda _: None)
        self.assertEqual(len(evaluator.calls), 4)  # two references, two syntactically valid samples
        evaluate_run(self.temp.name, lambda _: evaluator, progress=lambda _: None)
        self.assertEqual(len(evaluator.calls), 4)
        after = summarize(self.temp.name)
        self.assertEqual(after["evaluated_samples"], 4)
        self.assertEqual(after["conditions"]["watermarked"]["evaluation_statuses"], {"syntax_error": 2})
        self.assertEqual(after["conditions"]["watermarked"]["pass_at_1"], 0.0)

    def test_modified_task_snapshot_rejected(self):
        self.run_generation()
        path = Path(self.temp.name) / "tasks.json"
        tasks = json.loads(path.read_text())
        tasks[0]["test"] += "\n# changed"
        path.write_text(json.dumps(tasks))
        with self.assertRaisesRegex(ValueError, "snapshot"):
            load_run(self.temp.name)

    def test_duplicate_generation_rejected(self):
        self.run_generation()
        path = Path(self.temp.name) / "generations.jsonl"
        append_record(path, read_records(path)[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_generation()


class ExtractionTests(unittest.TestCase):
    def test_fenced_code_preserves_internal_whitespace(self):
        raw = "Here is the answer.\n```python\ndef answer():\n    return  42\n```\n"
        self.assertEqual(extract_code(raw, "answer")["code"], "def answer():\n    return  42")

    def test_ambiguous_or_unclosed_fences_rejected(self):
        for raw in ("```python\ndef answer(): pass", "```py\ndef answer(): pass\n```\n```py\nx=2\n```"):
            self.assertEqual(extract_code(raw, "answer")["status"], "extraction_error")

    def test_failures_are_explicit(self):
        for raw, status in (("", "empty"), ("def answer(", "syntax_error"), ("x=3", "missing_entrypoint")):
            self.assertEqual(extract_code(raw, "answer")["status"], status)


class StorageTests(unittest.TestCase):
    def test_complete_corrupt_line_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            path.write_bytes(b'{"ok": 1}\ninvalid\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                read_records(path, repair=True)

    def test_valid_tail_without_newline_can_be_appended(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            path.write_bytes(b'{"ok": 1}')
            read_records(path, repair=True)
            append_record(path, {"ok": 2})
            self.assertEqual(read_records(path), [{"ok": 1}, {"ok": 2}])


if __name__ == "__main__":
    unittest.main()
