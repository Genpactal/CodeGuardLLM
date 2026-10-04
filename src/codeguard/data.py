"""Task loading. Only task.prompt is given to the generation backend."""

import ast
import json
from pathlib import Path


def validate_tasks(tasks):
    seen = set()
    if not tasks:
        raise ValueError("The dataset is empty")
    for task in tasks:
        for field in ("task_id", "prompt", "entry_point", "test"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"Missing or invalid task field: {field}")
        if task["task_id"] in seen:
            raise ValueError(f"Duplicate task ID: {task['task_id']}")
        seen.add(task["task_id"])
        if not task["entry_point"].isidentifier():
            raise ValueError("entry_point must be a Python identifier")
        tree = ast.parse(task["test"])
        if not any(isinstance(node, ast.FunctionDef) and node.name == "check" for node in tree.body):
            raise ValueError("Task tests must define check(candidate)")
    return tasks


def load_tasks(config):
    if config["source"] == "local":
        with Path(config["path"]).open(encoding="utf-8") as handle:
            tasks = [json.loads(line) for line in handle if line.strip()]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
        tasks = [dict(row) for row in load_dataset(config["name"], revision=config["revision"], split=config["split"])]
    validate_tasks(tasks)
    if config["limit"] > len(tasks):
        raise ValueError(f"Requested {config['limit']} tasks, but only {len(tasks)} are available")
    return tasks[:config["limit"]]
