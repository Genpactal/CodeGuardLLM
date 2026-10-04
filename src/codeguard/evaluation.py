"""Docker-only evaluation. There is deliberately no host-execution fallback."""

import json
from pathlib import Path
import subprocess
import uuid

from .storage import append_record, digest, read_records, run_lock, write_json
from .pipeline import sample_id


class DockerEvaluator:
    def __init__(self, settings):
        self.settings = settings
        try:
            inspected = subprocess.run(
                ["docker", "image", "inspect", settings["image"], "--format", "{{.Id}}"],
                capture_output=True, text=True, timeout=15, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Docker evaluation is unavailable. Start Docker and build: docker build -t codeguard-eval:week1 sandbox") from exc
        image_id = inspected.stdout.strip()
        if not image_id.startswith("sha256:"):
            raise RuntimeError("Could not resolve the evaluator image ID")
        self.metadata = {"image_id": image_id, "settings": settings}

    def command(self, container_name):
        return [
            "docker", "run", "--rm", "--init", "--name", container_name, "--network", "none",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--pids-limit", "64", "--cpus", "1",
            "--memory", f"{self.settings['memory_mb']}m", "--memory-swap", f"{self.settings['memory_mb']}m",
            "--ulimit", "nofile=64:64", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--log-driver", "none", "-i", self.metadata["image_id"],
        ]

    def evaluate(self, code, task):
        name = "codeguard-" + uuid.uuid4().hex
        payload = {"code": code, "test": task["test"], "entry_point": task["entry_point"],
                   "timeout_seconds": self.settings["timeout_seconds"]}
        try:
            process = subprocess.run(self.command(name), input=json.dumps(payload), capture_output=True,
                                     text=True, timeout=self.settings["timeout_seconds"] + 20, check=False)
        except subprocess.TimeoutExpired as exc:
            # Kill only the container created for this sample; never leave a runaway program.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10, check=False)
            raise RuntimeError("Container startup or worker timed out; evaluation can be retried") from exc
        if process.returncode:
            # Infrastructure failures are not counted as incorrect code.
            raise RuntimeError(f"Docker evaluator failed: {process.stderr[-2000:]}")
        try:
            result = json.loads(process.stdout)
            if result["status"] not in ("passed", "failed", "timeout"):
                raise ValueError("Unknown evaluation status")
        except (ValueError, KeyError) as exc:
            raise RuntimeError("Invalid response from Docker evaluator") from exc
        return result


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    if digest(tasks) != manifest["tasks_sha256"]:
        raise ValueError("Saved task snapshot was modified")
    protocol = {key: manifest[key] for key in ("schema_version", "codeguard_version", "markllm_revision", "config", "tasks_sha256")}
    if digest(protocol) != manifest["experiment_id"]:
        raise ValueError("Saved experiment manifest was modified")
    expected = {sample_id(task["task_id"], seed, condition)
                for task in tasks for seed in manifest["config"]["generation"]["seeds"]
                for condition in ("clean", "watermarked")}
    if manifest["expected_samples"] != len(expected):
        raise ValueError("Incorrect expected sample count")
    rows = read_records(directory / "generations.jsonl")
    seen = set()
    for row in rows:
        identity = sample_id(row["task_id"], row["seed"], row["condition"])
        if (row["sample_id"] != identity or identity not in expected or identity in seen
                or row["experiment_id"] != manifest["experiment_id"]):
            raise ValueError("Duplicate or foreign generation records")
        seen.add(row["sample_id"])
    return manifest, {task["task_id"]: task for task in tasks}, rows


def evaluate_run(directory, evaluator_factory=DockerEvaluator, progress=print):
    directory = Path(directory)
    with run_lock(directory):
        manifest, tasks, rows = load_run(directory)
        if not rows:
            raise ValueError("Generate samples before evaluation")
        evaluator = evaluator_factory(manifest["config"]["evaluation"])
        eval_manifest = {"experiment_id": manifest["experiment_id"], **evaluator.metadata}
        metadata_path = directory / "evaluation_manifest.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != eval_manifest:
                raise ValueError("Evaluator image or settings changed; use a separate copy of the run without evaluation artifacts")
        else:
            write_json(metadata_path, eval_manifest)
        references_path = directory / "references.jsonl"
        references = {row["task_id"]: row for row in read_records(references_path, repair=True)}
        # Validate the evaluator and each test suite on a known reference before reporting quality.
        for task_id in dict.fromkeys(row["task_id"] for row in rows):
            task = tasks[task_id]
            if task_id in references:
                if references[task_id]["status"] != "passed":
                    raise ValueError(f"Previously failed reference: {task_id}")
                continue
            code = task.get("reference_code")
            if code is None and "canonical_solution" in task:
                code = task["prompt"] + task["canonical_solution"]
            if code is None:
                raise ValueError(f"No reference solution for evaluator validation: {task_id}")
            result = evaluator.evaluate(code, task)
            if result["status"] != "passed":
                raise RuntimeError(f"Reference solution failed for {task_id}; fix the evaluator or test fixture before proceeding")
            append_record(references_path, {"task_id": task_id, **result})
        results_path = directory / "evaluations.jsonl"
        previous = read_records(results_path, repair=True)
        completed = {row["sample_id"]: row for row in previous}
        originals = {row["sample_id"]: row for row in rows}
        if len(completed) != len(previous) or not completed.keys() <= originals.keys():
            raise ValueError("Duplicate or unknown evaluation records")
        for identity, result in completed.items():
            if result["generation_sha256"] != digest(originals[identity]):
                raise ValueError("Generation changed after evaluation")
        for row in rows:
            identity = row["sample_id"]
            if identity in completed:
                continue
            extraction = row["extraction"]
            if extraction["status"] != "ok":
                result = {"status": extraction["status"], "seconds": 0.0}
            else:
                result = evaluator.evaluate(extraction["code"], tasks[row["task_id"]])
            append_record(results_path, {"sample_id": identity, "generation_sha256": digest(row), **result})
            completed[identity] = result
            progress(f"[{len(completed)}/{len(rows)}] {row['task_id']} {row['condition']}: {result['status']}")
        return len(completed)
