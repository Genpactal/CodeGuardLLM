"""Resumable reference-code detection; reference code is never executed here."""

from copy import deepcopy
import json
from pathlib import Path

from .evaluation import load_run
from .storage import append_record, digest, read_records, run_lock, write_json


def reference_code(task):
    if task.get('reference_code') is not None:
        return task['reference_code']
    if task.get('canonical_solution') is not None:
        return task['prompt'] + task['canonical_solution']
    raise ValueError(f"No reference solution for {task['task_id']}")


def control_protocol(manifest):
    watermark = deepcopy(manifest['config']['watermark'])
    # Delta changes generation, not the text-only KGW/SWEET detection rule.
    watermark.pop('delta')
    return {'schema_version': 1, 'tasks_sha256': manifest['tasks_sha256'],
            'backend': manifest['backend'], 'watermark': watermark,
            'representation': 'complete reference code; HumanEval prompt + canonical_solution, including docstrings'}


def _read_controls(directory, manifest, tasks, repair=False):
    path = Path(directory)
    protocol = control_protocol(manifest)
    metadata = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if metadata != {**protocol, 'control_id': digest(protocol), 'expected_samples': len(tasks)}:
        raise ValueError('Reference controls differ from the generation dataset or detector environment')
    rows = read_records(path / 'scores.jsonl', repair=repair)
    seen = set()
    for row in rows:
        task_id = row['task_id']
        if (task_id not in tasks or task_id in seen or row['control_id'] != metadata['control_id']
                or row['condition'] != 'human_reference'
                or row['code_sha256'] != digest(reference_code(tasks[task_id]))):
            raise ValueError('Duplicate, foreign, or changed reference control')
        seen.add(task_id)
    return rows


def load_reference_controls(directory, manifest, tasks):
    if not (Path(directory) / 'manifest.json').is_file():
        raise ValueError('Reference controls are missing. Run the controls step before the full report.')
    rows = _read_controls(directory, manifest, tasks)
    if len(rows) != len(tasks):
        raise ValueError('Reference controls are incomplete; resume the controls step')
    return rows


def score_reference_controls(run_dir, output, markllm_path, backend_factory=None, progress=print):
    manifest, tasks, _ = load_run(run_dir)
    protocol = control_protocol(manifest)
    control_id = digest(protocol)
    output = Path(output)
    with run_lock(output):
        metadata_path = output / 'manifest.json'
        if not metadata_path.exists():
            if (output / 'scores.jsonl').exists():
                raise ValueError('Reference scores exist without a manifest')
            write_json(metadata_path, {**protocol, 'control_id': control_id, 'expected_samples': len(tasks)})
        rows = _read_controls(output, manifest, tasks, repair=True)
        completed = {row['task_id'] for row in rows}
        if len(completed) == len(tasks):
            progress(f'Already scored: {len(tasks)} reference solutions')
            return len(tasks)
        if backend_factory is None:
            from .backend import HFBackend
            backend_factory = HFBackend
        config = deepcopy(manifest['config'])
        config['markllm_path'] = str(Path(markllm_path).resolve())
        detector = backend_factory(config, load_model=False)
        if detector.metadata != manifest['backend']:
            raise ValueError('Reference detector environment differs from generation; use matching versions and device')
        for task_id, task in tasks.items():
            if task_id in completed:
                continue
            code = reference_code(task)
            append_record(output / 'scores.jsonl', {
                'control_id': control_id, 'task_id': task_id, 'condition': 'human_reference',
                'code_sha256': digest(code), 'code_detection': detector.score(code),
            })
            completed.add(task_id)
            progress(f'[{len(completed)}/{len(tasks)}] reference {task_id}: scored')
        return len(completed)
