"""Paired generation, immediate detection, and strict experiment resumption."""

from datetime import datetime, timezone
import json
from pathlib import Path

from . import __version__, MARKLLM_REVISION
from .config import portable_config
from .data import load_tasks
from .extraction import extract_code
from .storage import append_record, digest, read_records, run_lock, write_json


def sample_id(task_id, seed, condition):
    return digest([task_id, seed, condition])[:24]


def generate(config, backend_factory=None, progress=print):
    if backend_factory is None:
        from .backend import HFBackend
        backend_factory = HFBackend
    tasks = load_tasks(config["dataset"])
    directory = Path(config["output_dir"])
    protocol = {"schema_version": 1, "codeguard_version": __version__, "markllm_revision": MARKLLM_REVISION,
                "config": portable_config(config), "tasks_sha256": digest(tasks)}
    experiment_id = digest(protocol)
    expected = {sample_id(task["task_id"], seed, condition)
                for task in tasks for seed in config["generation"]["seeds"] for condition in ("clean", "watermarked")}
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        records_path = directory / "generations.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if manifest and manifest["experiment_id"] != experiment_id:
            raise ValueError("Run configuration or dataset changed. Choose a new output_dir")
        if manifest and digest(json.loads((directory / "tasks.json").read_text(encoding="utf-8"))) != protocol["tasks_sha256"]:
            raise ValueError("Saved task snapshot was modified")
        records = read_records(records_path, repair=True)
        if records and manifest is None:
            raise ValueError("Generation records exist without a manifest")
        completed = set()
        for row in records:
            identity = row["sample_id"]
            actual_id = sample_id(row["task_id"], row["seed"], row["condition"])
            if row["experiment_id"] != experiment_id or identity != actual_id or identity not in expected or identity in completed:
                raise ValueError("Unexpected, foreign, or duplicate generation record")
            completed.add(identity)
        if completed == expected:
            progress(f"Already complete: {len(completed)} samples in {directory}")
            return len(completed)
        backend = backend_factory(config)
        if manifest and manifest["backend"] != backend.metadata:
            raise ValueError("Model, tokenizer, device, or dependency environment changed. Use a new output_dir")
        if manifest is None:
            write_json(directory / "tasks.json", tasks)
            manifest = {**protocol, "experiment_id": experiment_id, "backend": backend.metadata,
                        "created_at": datetime.now(timezone.utc).isoformat(), "expected_samples": len(expected)}
            write_json(manifest_path, manifest)
        for task in tasks:
            for seed in config["generation"]["seeds"]:
                for condition in ("clean", "watermarked"):
                    identity = sample_id(task["task_id"], seed, condition)
                    if identity in completed:
                        continue
                    # No tests, canonical solution, or watermark condition is included in prompt text.
                    output = backend.generate(task["prompt"], task["task_id"], seed, condition)
                    extraction = extract_code(output["raw_text"], task["entry_point"])
                    raw_score = backend.score(output["raw_text"])
                    code_score = backend.score(extraction["code"]) if extraction["code"] is not None else None
                    record = {"sample_id": identity, "experiment_id": experiment_id, "task_id": task["task_id"],
                              "seed": seed, "condition": condition, **output, "extraction": extraction,
                              "raw_detection": raw_score, "code_detection": code_score}
                    append_record(records_path, record)
                    completed.add(identity)
                    progress(f"[{len(completed)}/{len(expected)}] {task['task_id']} {condition}: {extraction['status']}")
        return len(completed)
