"""Validate authored fixtures and configuration errors, never generated code."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from codeguard.config import load_config
from codeguard.data import load_tasks, validate_tasks

ROOT = Path(__file__).resolve().parents[1]


class DataConfigTests(unittest.TestCase):
    def test_authored_development_references_match_their_tests(self):
        config = load_config(ROOT / "configs/pilot.yaml")
        tasks = load_tasks(config["dataset"])
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            with self.subTest(task=task["task_id"]):
                # These are trusted, project-authored fixtures, not model output.
                namespace = {}
                exec(compile(task["reference_code"], task["task_id"], "exec"), namespace)
                exec(compile(task["test"], task["task_id"] + "/tests", "exec"), namespace)
                namespace["check"](namespace[task["entry_point"]])

    def test_duplicate_tasks_rejected(self):
        config = load_config(ROOT / "configs/pilot.yaml")
        task = load_tasks(config["dataset"])[0]
        with self.assertRaisesRegex(ValueError, "Duplicate task"):
            validate_tasks([task, task])

    def test_invalid_protocol_options_rejected(self):
        raw = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text())
        cases = [("watermark", "gamma", 1.0), ("generation", "seeds", [42, 42]),
                 ("model", "revision", "main"), ("generation", "temperature", 0),
                 ("watermark", "prefix_length", 4), ("evaluation", "memory_mb", -1)]
        with tempfile.TemporaryDirectory() as temp:
            for section, field, value in cases:
                with self.subTest(field=field):
                    broken = deepcopy(raw)
                    broken[section][field] = value
                    path = Path(temp) / "config.yaml"
                    path.write_text(yaml.safe_dump(broken))
                    with self.assertRaises(ValueError):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()
