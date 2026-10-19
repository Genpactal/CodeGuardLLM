"""Exploratory, matched KGW/SWEET tuning with a shared clean generation cache."""

from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import statistics
import sys

from .analysis import valid_score, write_csv
from .config import load_config, portable_config
from .controls import control_protocol, load_reference_controls, score_reference_controls
from .data import load_tasks
from .evaluation import evaluate_run, load_run, prepare_docker
from .pipeline import generate, sample_id
from .storage import append_record, digest, read_records, run_lock, write_json


def candidates(profile='pilot'):
    """A small first stage; the full grid also varies SWEET strength at each cutoff."""
    if profile not in ('pilot', 'full'):
        raise ValueError('profile must be pilot or full')
    result = []
    for gamma in (0.1, 0.25, 0.5):
        result.append(dict(label=f'kgw-g{gamma:g}-d2', algorithm='KGW', gamma=gamma, delta=2.0))
    if profile == 'full':
        for delta in (1, 3):
            result.append(dict(label=f'kgw-g0.25-d{delta}', algorithm='KGW', gamma=0.25, delta=float(delta)))
    for cutoff in (0.3, 0.6, 0.9, 1.2):
        strengths = (1, 2, 3) if profile == 'full' or cutoff == 0.9 else (2,)
        for delta in strengths:
            result.append(dict(label=f'sweet-h{cutoff:g}-d{delta}', algorithm='SWEET',
                               gamma=0.25, delta=float(delta), entropy_threshold=cutoff))
    return result


def prepare_suite(config_path, output, task_count=64, calibration_count=32,
                  split_seed=20261019, seeds=None, profile='pilot', target_fpr=0.05,
                  max_quality_loss=0.05):
    if not 0 < calibration_count < task_count:
        raise ValueError('Calibration and selection must each contain tasks')
    if not 0 < target_fpr < 1 or not 0 <= max_quality_loss <= 1:
        raise ValueError('Invalid FPR or quality-loss target')
    config = load_config(config_path)
    if seeds is not None:
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
            raise ValueError('Seeds must be distinct unsigned 32-bit integers')
        config['generation']['seeds'] = seeds
    tasks = load_tasks(config['dataset'])
    if task_count > len(tasks):
        raise ValueError('Requested more tasks than the source dataset contains')
    ordered = sorted(tasks, key=lambda task: digest([split_seed, task['task_id']]))
    chosen = ordered[:task_count]
    protocol = {
        'schema_version': 1, 'base_config': portable_config(config),
        'source_tasks_sha256': digest(tasks), 'tasks_sha256': digest(chosen),
        'split_seed': split_seed, 'profile': profile, 'candidates': candidates(profile),
        'calibration_task_ids': [t['task_id'] for t in chosen[:calibration_count]],
        'selection_task_ids': [t['task_id'] for t in chosen[calibration_count:]],
        'unused_task_ids': [t['task_id'] for t in ordered[task_count:]],
        'target_fpr': target_fpr, 'max_quality_loss': max_quality_loss,
        'scope': 'Exploratory tuning on previously studied HumanEval; unused tasks are not an independent benchmark.',
    }
    suite = {**protocol, 'suite_id': digest(protocol)}
    output = Path(output)
    with run_lock(output):
        if (output / 'suite.json').exists():
            if _read_json(output / 'suite.json') != suite:
                raise ValueError('Tuning settings changed. Choose a new output directory')
            load_suite(output)
        else:
            if (output / 'candidates').exists() or (output / 'shared_clean').exists():
                raise ValueError('Tuning records exist without a suite manifest')
            # Write the complete dataset atomically before committing its manifest.
            write_json(output / 'tasks.json', chosen)
            write_json(output / 'suite.json', suite)
        # This derived file can be recreated after a checkpoint interrupted its write.
        task_file = output / 'tasks.jsonl'
        temporary = task_file.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in chosen), encoding='utf-8')
        temporary.replace(task_file)
    count = task_count * len(config['generation']['seeds'])
    print(f"Prepared {len(suite['candidates'])} candidates; {count} shared clean + "
          f"{count * len(suite['candidates'])} watermarked completions", flush=True)
    return suite


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_suite(output):
    output = Path(output)
    suite = _read_json(output / 'suite.json')
    protocol = {k: v for k, v in suite.items() if k != 'suite_id'}
    if digest(protocol) != suite['suite_id']:
        raise ValueError('Tuning manifest was modified')
    tasks = _read_json(output / 'tasks.json')
    if digest(tasks) != suite['tasks_sha256']:
        raise ValueError('Tuning task snapshot was modified')
    calibration = set(suite['calibration_task_ids'])
    selection = set(suite['selection_task_ids'])
    if (not calibration or not selection or calibration & selection
            or calibration | selection != {t['task_id'] for t in tasks}):
        raise ValueError('Invalid calibration/selection task split')
    return suite, tasks


def suite_configs(output, markllm_path='.deps/MarkLLM'):
    output = Path(output).resolve()
    suite, tasks = load_suite(output)
    # Refuse a changed derived dataset before any generation can use it.
    if read_records(output / 'tasks.jsonl') != tasks:
        raise ValueError('Derived task file differs; rerun prepare with the original settings')
    result = {}
    for candidate in suite['candidates']:
        config = deepcopy(suite['base_config'])
        label = candidate['label']
        config.update(name=f'tuning-{label}', output_dir=str(output / 'candidates' / label),
                      markllm_path=str(Path(markllm_path).resolve()))
        config['dataset'] = dict(source='local', path=str(output / 'tasks.jsonl'), limit=len(tasks))
        config['watermark'].pop('entropy_threshold', None)
        config['watermark'].update({k: v for k, v in candidate.items() if k != 'label'})
        result[label] = config
    return result


def generation_runtime(metadata):
    # SWEET adds these detector-only fields; all generation/runtime fields must match.
    return {k: v for k, v in metadata.items() if k not in (
        'algorithm', 'detector_version', 'entropy_context', 'entropy_precision', 'entropy_chunk_size')}


class SharedCleanBackend:
    """Reuse token-identical clean outputs, while each detector scores them itself."""

    def __init__(self, backend, config, output, tasks):
        self.backend, self.metadata = backend, backend.metadata
        self.directory = Path(output) / 'shared_clean'
        self.directory.mkdir(parents=True, exist_ok=True)
        protocol = dict(model=config['model'], generation=config['generation'],
                        tasks_sha256=digest(tasks), runtime=generation_runtime(self.metadata))
        self.cache_id = digest(protocol)
        metadata = {**protocol, 'cache_id': self.cache_id}
        manifest_path = self.directory / 'manifest.json'
        self.path = self.directory / 'generations.jsonl'
        if manifest_path.exists():
            if _read_json(manifest_path) != metadata:
                raise ValueError('Shared clean generation environment changed; start a new suite')
        else:
            if self.path.exists():
                raise ValueError('Clean cache exists without its manifest')
            write_json(manifest_path, metadata)
        expected = {sample_id(t['task_id'], seed, 'clean') for t in tasks for seed in config['generation']['seeds']}
        self.cached = {}
        for row in read_records(self.path, repair=True):
            identity = sample_id(row['task_id'], row['seed'], 'clean')
            if (identity in self.cached or identity not in expected or row['sample_id'] != identity
                    or row['cache_id'] != self.cache_id or digest(row['output']) != row['output_sha256']):
                raise ValueError('Duplicate, foreign, or modified clean cache record')
            self.cached[identity] = row['output']

    def generate(self, prompt, task_id, seed, condition):
        if condition != 'clean':
            return self.backend.generate(prompt, task_id, seed, condition)
        identity = sample_id(task_id, seed, condition)
        if identity not in self.cached:
            output = self.backend.generate(prompt, task_id, seed, condition)
            append_record(self.path, dict(sample_id=identity, task_id=task_id, seed=seed,
                                          cache_id=self.cache_id, output=output, output_sha256=digest(output)))
            self.cached[identity] = output
        return deepcopy(self.cached[identity])

    def score(self, text):
        return self.backend.score(text)


def _release_gpu():
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_suite(output, markllm_path='.deps/MarkLLM', backend_factory=None):
    if backend_factory is None:
        from .backend import HFBackend
        backend_factory = HFBackend
    with run_lock(output):
        _, tasks = load_suite(output)
        for label, config in suite_configs(output, markllm_path).items():
            print(f'Candidate: {label}', flush=True)
            def factory(c):
                return SharedCleanBackend(backend_factory(c), c, output, tasks)
            generate(config, factory)
            _release_gpu()


def controls_suite(output, markllm_path='.deps/MarkLLM', backend_factory=None):
    with run_lock(output):
        for config in suite_configs(output, markllm_path).values():
            manifest, _, _ = load_run(config['output_dir'])
            _check_candidate(manifest, config)
            controls = Path(output) / 'controls' / digest(control_protocol(manifest))[:24]
            score_reference_controls(config['output_dir'], controls, markllm_path, backend_factory)
            _release_gpu()


def evaluate_suite(output):
    with run_lock(output):
        configs = suite_configs(output)
        prepare_docker(next(iter(configs.values()))['evaluation']['image'])
        for config in configs.values():
            manifest, _, _ = load_run(config['output_dir'])
            _check_candidate(manifest, config)
            evaluate_run(config['output_dir'])


def _check_candidate(manifest, config):
    if manifest['config'] != portable_config(config):
        raise ValueError('Candidate configuration differs from the tuning plan')


def calibrated_threshold(groups, target_fpr):
    """Smallest conservative strict-Z threshold satisfying each empirical null FPR."""
    if not 0 < target_fpr < 1 or not groups or any(not group for group in groups):
        raise ValueError('Calibration needs nonempty control groups and a valid target FPR')
    thresholds = []
    for group in groups:
        if any(not math.isfinite(z) for z in group):
            raise ValueError('Calibration scores must be finite')
        ordered = sorted(group, reverse=True)
        thresholds.append(ordered[math.floor(target_fpr * len(ordered))])
    return max(thresholds)


def _scores(rows):
    return [r['code_detection']['z_score'] for r in rows if valid_score(r.get('code_detection'))]


def _auc(negative, positive):
    if not negative or not positive:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([0] * len(negative) + [1] * len(positive), negative + positive))


def analyze_suite(output):
    """Calibrate only on null calibration tasks; rank only evaluated selection tasks."""
    output = Path(output)
    with run_lock(output):
        suite, _ = load_suite(output)
        metrics, provenance = [], {}
        shared_outputs, runtime_ids = {}, set()
        calibration = set(suite['calibration_task_ids'])
        selection = set(suite['selection_task_ids'])
        for label, config in suite_configs(output).items():
            directory = Path(config['output_dir'])
            manifest, tasks, rows = load_run(directory)
            _check_candidate(manifest, config)
            if len(rows) != manifest['expected_samples']:
                raise ValueError(f'Incomplete candidate {label}; resume generation first')
            runtime_ids.add(digest(generation_runtime(manifest['backend'])))
            control_dir = output / 'controls' / digest(control_protocol(manifest))[:24]
            refs = load_reference_controls(control_dir, manifest, tasks)
            originals = {r['sample_id']: r for r in rows}
            evaluations = {}
            for evaluation in read_records(directory / 'evaluations.jsonl'):
                identity = evaluation['sample_id']
                if (identity not in originals or identity in evaluations
                        or evaluation['generation_sha256'] != digest(originals[identity])):
                    raise ValueError('Duplicate, foreign, or changed evaluation record')
                evaluations[identity] = evaluation
            for row in rows:
                if row['condition'] == 'clean':
                    identity = row['sample_id']
                    value = digest([row['raw_text'], row['generated_token_ids']])
                    if identity in shared_outputs and shared_outputs[identity] != value:
                        raise ValueError('Candidates do not share identical clean outputs')
                    shared_outputs[identity] = value
            null_clean = [r for r in rows if r['condition'] == 'clean' and r['task_id'] in calibration]
            null_refs = [r for r in refs if r['task_id'] in calibration]
            clean_cal, ref_cal = _scores(null_clean), _scores(null_refs)
            threshold = calibrated_threshold([clean_cal, ref_cal], suite['target_fpr']) if clean_cal and ref_cal else None
            clean = [r for r in rows if r['condition'] == 'clean' and r['task_id'] in selection]
            marked = [r for r in rows if r['condition'] == 'watermarked' and r['task_id'] in selection]
            reference = [r for r in refs if r['task_id'] in selection]
            clean_z, marked_z, ref_z = _scores(clean), _scores(marked), _scores(reference)
            def detection_rate(values, denominator):
                return sum(z > threshold for z in values) / denominator if threshold is not None and denominator else None
            def pass_rate(group):
                return (sum(evaluations[r['sample_id']]['status'] == 'passed' for r in group) / len(group)
                        if group and all(r['sample_id'] in evaluations for r in group) else None)
            clean_pass, marked_pass = pass_rate(clean), pass_rate(marked)
            loss = clean_pass - marked_pass if clean_pass is not None and marked_pass is not None else None
            joint = (sum(valid_score(r['code_detection']) and r['code_detection']['z_score'] > threshold
                         and evaluations[r['sample_id']]['status'] == 'passed' for r in marked) / len(marked)
                     if marked_pass is not None and threshold is not None else None)
            metric = {
                'candidate': label, 'algorithm': config['watermark']['algorithm'],
                'gamma': config['watermark']['gamma'], 'delta': config['watermark']['delta'],
                'entropy_threshold': config['watermark'].get('entropy_threshold'),
                'z_threshold': threshold, 'target_calibration_fpr': suite['target_fpr'],
                'calibration_clean_scored': len(clean_cal), 'calibration_clean_total': len(null_clean),
                'calibration_reference_scored': len(ref_cal), 'calibration_reference_total': len(null_refs),
                'calibration_clean_fpr_scored': detection_rate(clean_cal, len(clean_cal)),
                'calibration_reference_fpr_scored': detection_rate(ref_cal, len(ref_cal)),
                'selection_clean_fpr_scored': detection_rate(clean_z, len(clean_z)),
                'selection_reference_fpr_scored': detection_rate(ref_z, len(ref_z)),
                'selection_samples': len(marked), 'selection_watermarked_scored': len(marked_z),
                'selection_clean_scored': len(clean_z), 'selection_reference_scored': len(ref_z),
                'detection_rate_all': detection_rate(marked_z, len(marked)),
                'auroc_vs_clean': _auc(clean_z, marked_z), 'auroc_vs_reference': _auc(ref_z, marked_z),
                'median_scored_tokens': statistics.median(r['code_detection']['scored_tokens'] for r in marked
                                                        if valid_score(r['code_detection'])) if marked_z else None,
                'clean_pass_at_1': clean_pass, 'watermarked_pass_at_1': marked_pass,
                'quality_loss': loss, 'correct_and_detected_rate': joint,
                'within_quality_budget': loss <= suite['max_quality_loss'] if loss is not None else None,
            }
            metrics.append(metric)
            provenance[label] = dict(experiment_id=manifest['experiment_id'], records_sha256=digest(rows),
                                     controls_sha256=digest(refs), evaluations_sha256=digest(list(evaluations.values())))
        if len(runtime_ids) != 1:
            raise ValueError('Candidates were generated in different environments')
        evaluated = all(m['quality_loss'] is not None for m in metrics)
        shortlist = {}
        if evaluated:
            for algorithm in ('KGW', 'SWEET'):
                eligible = [m for m in metrics if m['algorithm'] == algorithm and m['within_quality_budget']
                            and m['correct_and_detected_rate'] is not None]
                if eligible:
                    best = max(eligible, key=lambda m: (m['correct_and_detected_rate'], m['watermarked_pass_at_1'], m['candidate']))
                    shortlist[algorithm] = best['candidate']
        summary = dict(status='selection_evaluated' if evaluated else 'correctness_pending',
                       scope=suite['scope'], target_fpr=suite['target_fpr'],
                       max_quality_loss=suite['max_quality_loss'], metrics=metrics,
                       exploratory_shortlist=shortlist, suite_id=suite['suite_id'])
        destination = output / 'analysis'
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / 'summary.json', summary)
        write_json(destination / 'provenance.json', provenance)
        write_csv(destination / 'metrics.csv', metrics)
        def number(value):
            return 'pending / unavailable' if value is None else f'{value:.3f}'
        lines = ['# Matched KGW / SWEET tuning', '', f"Status: **{summary['status']}**.", '',
                 suite['scope'], '',
                 f"Calibration uses {len(calibration)} task IDs; selection uses {len(selection)} different task IDs. "
                 'All seeds for a task stay in the same group. Each detector scores the same clean outputs.', '',
                 f"Thresholds target an empirical FPR <= {suite['target_fpr']:.1%} separately on scorable clean "
                 'and reference calibration code. This small pilot does not guarantee the population FPR. '
                 'HumanEval references are not an independent human-code corpus. Unscored outputs remain explicit.', '',
                 '| Candidate | Z threshold | Selection clean FPR | Selection reference FPR | Detection / all | AUROC vs clean | Clean Pass@1 | WM Pass@1 | Correct and detected / all |',
                 '|---|---|---|---|---|---|---|---|---|']
        for row in metrics:
            fields = ('z_threshold', 'selection_clean_fpr_scored', 'selection_reference_fpr_scored',
                      'detection_rate_all', 'auroc_vs_clean', 'clean_pass_at_1', 'watermarked_pass_at_1', 'correct_and_detected_rate')
            lines.append('| ' + row['candidate'] + ' | ' + ' | '.join(number(row[f]) for f in fields) + ' |')
        lines += ['', 'Coverage, calibration counts, eligible-token counts and all metrics are in `metrics.csv`.', '',
                  'Repeated clean outputs and shared references are not additional independent observations. '
                  'Native KGW and SWEET use different green-list seeding; this compares implementations.', '',
                  f"Exploratory shortlist: {json.dumps(shortlist) if shortlist else 'unavailable; correctness pending or no candidate meets the quality budget'}.", '',
                  'Shortlisting maximizes joint correctness and detection within the quality-loss budget. '
                  'Thresholds use calibration tasks only; selection metrics are tuning results. '
                  'Confirm candidates with at least three seeds and evaluate frozen settings and thresholds on an independent benchmark. '
                  'Do not call this pilot a final held-out result.', '']
        (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
        return summary
