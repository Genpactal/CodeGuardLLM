"""Verify container restrictions; opt in to real Docker smoke tests."""

import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from codeguard.config import load_config
from codeguard.data import load_tasks
from codeguard.evaluation import DockerEvaluator

ROOT = Path(__file__).resolve().parents[1]


class EvaluatorContractTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config(ROOT / "configs/week1.yaml")["evaluation"]

    def test_container_has_no_mounts_or_network_and_pins_image_id(self):
        with patch("codeguard.evaluation.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "sha256:abc\n", "")):
            evaluator = DockerEvaluator(self.settings)
        command = evaluator.command("codeguard-test")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertEqual(command[-1], "sha256:abc")
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--volume", command)
        self.assertNotIn("--mount", command)

    def test_missing_docker_is_an_infrastructure_error(self):
        with patch("codeguard.evaluation.subprocess.run", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "Docker evaluation is unavailable"):
                DockerEvaluator(self.settings)


@unittest.skipUnless(os.environ.get("CODEGUARD_DOCKER_TESTS") == "1", "Set CODEGUARD_DOCKER_TESTS=1 after building the evaluator image")
class DockerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/week1.yaml")
        cls.evaluator = DockerEvaluator(cls.config["evaluation"])

    def test_all_development_references(self):
        for task in load_tasks(self.config["dataset"]):
            with self.subTest(task=task["task_id"]):
                self.assertEqual(self.evaluator.evaluate(task["reference_code"], task)["status"], "passed")

    def test_failure_timeout_and_early_exit(self):
        task = {"entry_point": "answer", "test": "def check(candidate):\n    assert candidate() == 42\n"}
        for code, status in (("def answer(): return 42", "passed"),
                             ("def answer(): return 0", "failed"),
                             ("while True: pass", "timeout"),
                             ("import sys; sys.exit(0)", "failed"),
                             ("import os; os._exit(0)", "failed")):
            with self.subTest(code=code):
                self.assertEqual(self.evaluator.evaluate(code, task)["status"], status)


if __name__ == "__main__":
    unittest.main()
