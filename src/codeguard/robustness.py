"""Resumable, paired robustness experiments on saved clean and watermarked code."""

from copy import deepcopy
import json
from pathlib import Path
import platform
import statistics

from .analysis import valid_score, write_csv
from .attacks import ATTACKS, VERSION, transform
from .controls import reference_code
from .evaluation import DockerEvaluator, load_run
from .extraction import extract_code
from .plotting import report_pyplot
from .storage import append_record, digest, read_records, run_lock, write_json


def _source(directory):
    manifest, tasks, rows = load_run(directory)
    if len(rows) != manifest['expected_samples']:
        raise ValueError('Complete source generation before robustness testing')
    return manifest, tasks, rows


def prepare(run_dir, output, attacks=ATTACKS, progress=print):
    """Freeze variants and their source identity; never modify the original run."""
    attacks = list(attacks)
    if not attacks or len(set(attacks)) != len(attacks) or any(a not in ATTACKS for a in attacks):
        raise ValueError('Select distinct supported attacks')
    attacks = ['identity', *[a for a in attacks if a != 'identity']]
    source, output = Path(run_dir).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError('Robustness output must be separate from source artifacts')
    manifest, tasks, rows = _source(source)
    protocol = dict(schema_version=1, transform_version=VERSION, python=platform.python_version(),
                    source_manifest_sha256=digest(manifest), source_rows_sha256=digest(rows),
                    attacks=attacks, expected_variants=len(rows) * len(attacks))
    metadata = {**protocol, 'robustness_id': digest(protocol)}
    with run_lock(output):
        path = output / 'manifest.json'
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if {k: v for k, v in previous.items() if k != 'variants_sha256'} != metadata:
                raise ValueError('Source, attacks, or transformation runtime changed; use a new output directory')
        else:
            if any(p.name != '.lock' for p in output.iterdir()):
                raise ValueError('Output directory contains artifacts without a robustness manifest')
            write_json(path, metadata)
        existing = read_records(output / 'variants.jsonl', repair=True)
        indexed = {r['variant_id']: r for r in existing}
        if len(indexed) != len(existing):
            raise ValueError('Duplicate variants')
        expected = set()
        for row in rows:
            for attack in attacks:
                variant = dict(variant_id=digest([row['sample_id'], attack]),
                               sample_id=row['sample_id'], task_id=row['task_id'],
                               condition=row['condition'], attack=attack,
                               source_sha256=digest(row), **transform(row['extraction']['code'], attack))
                expected.add(variant['variant_id'])
                if variant['variant_id'] in indexed:
                    if indexed[variant['variant_id']] != variant:
                        raise ValueError('Saved variant changed')
                else:
                    append_record(output / 'variants.jsonl', variant)
                    existing.append(variant)
        if indexed.keys() - expected:
            raise ValueError('Foreign variants')
        write_json(path, {**metadata, 'variants_sha256': digest(existing)})
    progress(f'Prepared {len(existing)} variants in {output}')
    return len(existing)


def _bundle(run_dir, output):
    source, tasks, rows = _source(run_dir)
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    protocol = {k: v for k, v in manifest.items() if k not in ('robustness_id', 'variants_sha256')}
    if (manifest['robustness_id'] != digest(protocol)
            or manifest['source_manifest_sha256'] != digest(source)
            or manifest['source_rows_sha256'] != digest(rows)):
        raise ValueError('Robustness source or manifest changed')
    variants = read_records(output / 'variants.jsonl')
    if (len(variants) != manifest['expected_variants']
            or manifest.get('variants_sha256') != digest(variants)):
        raise ValueError('Variants incomplete or changed; finish prepare first')
    expected = {digest([r['sample_id'], a]) for r in rows for a in manifest['attacks']}
    if len({r['variant_id'] for r in variants}) != len(variants) or {r['variant_id'] for r in variants} != expected:
        raise ValueError('Duplicate or foreign variants')
    for stage in ('scores', 'evaluations'):
        path = output / f'{stage}_manifest.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['robustness_id'] != manifest['robustness_id']:
            raise ValueError(f'Foreign {stage} manifest')
    return source, tasks, manifest, variants


def _results(output, name, variants, repair=False):
    originals = {v['variant_id']: v for v in variants}
    result = {}
    for row in read_records(Path(output) / f'{name}.jsonl', repair=repair):
        key = row['variant_id']
        if (key not in originals or key in result or row['variant_sha256'] != digest(originals[key])):
            raise ValueError(f'Duplicate, foreign, or changed {name} record')
        result[key] = row
    if result and not (Path(output) / f'{name}_manifest.json').exists():
        raise ValueError(f'{name} records exist without a manifest')
    return result


def _metadata(output, name, metadata):
    path = Path(output) / f'{name}_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != metadata:
            raise ValueError(f'{name} environment changed; use a separate output directory')
    else:
        write_json(path, metadata)


def score(run_dir, output, markllm_path, device=None, dtype=None, backend_factory=None, progress=print):
    output = Path(output)
    with run_lock(output):
        source, _, manifest, variants = _bundle(run_dir, output)
        config = deepcopy(source['config'])
        config['markllm_path'] = str(Path(markllm_path).resolve())
        if device is not None:
            config['model']['device'] = device
        if dtype is not None:
            config['model']['dtype'] = dtype
        if config['model']['device'] == 'cpu' and config['model']['dtype'] != 'float32':
            raise ValueError('CPU scoring requires explicit float32 precision')
        settings = {k: v for k, v in config.items() if k != 'markllm_path'}
        completed = _results(output, 'scores', variants, repair=True)
        metadata_path = output / 'scores_manifest.json'
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding='utf-8'))
            if previous['config'] != settings or previous['robustness_id'] != manifest['robustness_id']:
                raise ValueError('Scoring settings changed; use a separate output directory')
        if len(completed) == len(variants):
            return len(completed)
        if backend_factory is None:
            from .backend import HFBackend
            backend_factory = HFBackend
        detector = backend_factory(config, load_model=False)
        _metadata(output, 'scores', dict(robustness_id=manifest['robustness_id'],
                                        config=settings, backend=detector.metadata))
        # Even identity is rescored here: never mix original Colab scores with local scores.
        for variant in variants:
            if variant['variant_id'] in completed:
                continue
            code = variant['code']
            result = detector.score(code) if code is not None else None
            append_record(output / 'scores.jsonl', dict(variant_id=variant['variant_id'],
                          variant_sha256=digest(variant), code_detection=result))
            completed[variant['variant_id']] = True
            progress(f"[{len(completed)}/{len(variants)}] {variant['attack']} {variant['task_id']}: scored")
    return len(completed)


def evaluate(run_dir, output, evaluator_factory=DockerEvaluator, progress=print):
    output = Path(output)
    with run_lock(output):
        source, tasks, manifest, variants = _bundle(run_dir, output)
        completed = _results(output, 'evaluations', variants, repair=True)
        if (output / 'references.jsonl').exists() and not (output / 'evaluations_manifest.json').exists():
            raise ValueError('Reference validations exist without an evaluator manifest')
        evaluator = evaluator_factory(source['config']['evaluation'])
        _metadata(output, 'evaluations', dict(robustness_id=manifest['robustness_id'],
                                            evaluator=evaluator.metadata))
        references = read_records(output / 'references.jsonl', repair=True)
        seen = set()
        for row in references:
            key = row['task_id']
            if (key not in tasks or key in seen or row['task_sha256'] != digest(tasks[key])
                    or row['status'] != 'passed'):
                raise ValueError('Invalid reference validation record')
            seen.add(key)
        for key, task in tasks.items():
            if key in seen:
                continue
            result = evaluator.evaluate(reference_code(task), task)
            if result['status'] != 'passed':
                raise RuntimeError(f'Reference solution failed: {key}')
            append_record(output / 'references.jsonl', dict(task_id=key, task_sha256=digest(task), **result))
        for variant in variants:
            if variant['variant_id'] in completed:
                continue
            task = tasks[variant['task_id']]
            if variant['code'] is None:
                result = dict(status=variant['status'], seconds=0.0)
            else:
                extraction = extract_code(variant['code'], task['entry_point'])
                result = (evaluator.evaluate(variant['code'], task) if extraction['status'] == 'ok'
                          else dict(status=extraction['status'], seconds=0.0))
            append_record(output / 'evaluations.jsonl', dict(variant_id=variant['variant_id'],
                          variant_sha256=digest(variant), **result))
            completed[variant['variant_id']] = True
            progress(f"[{len(completed)}/{len(variants)}] {variant['attack']} {variant['task_id']}: {result['status']}")
    return len(completed)


def report(run_dir, output):
    output = Path(output)
    with run_lock(output):
        source, _, manifest, variants = _bundle(run_dir, output)
        scores = _results(output, 'scores', variants)
        evaluations = _results(output, 'evaluations', variants)
        threshold = source['config']['watermark']['z_threshold']
        baselines = {v['sample_id']: v for v in variants if v['attack'] == 'identity'}
        def z(v):
            s = scores.get(v['variant_id'], {}).get('code_detection')
            return s['z_score'] if valid_score(s) else None
        def passed(v):
            e = evaluations.get(v['variant_id'])
            return None if e is None else e['status'] == 'passed'
        groups = []
        for attack in manifest['attacks']:
            auc = None
            selected = [v for v in variants if v['attack'] == attack]
            distributions = [[z(v) for v in selected if v['condition'] == c and z(v) is not None]
                             for c in ('clean', 'watermarked')]
            if all(v['variant_id'] in scores for v in selected) and all(distributions):
                from sklearn.metrics import roc_auc_score
                neg, pos = distributions
                auc = float(roc_auc_score([0] * len(neg) + [1] * len(pos), neg + pos))
            for condition in ('clean', 'watermarked'):
                rows = [v for v in selected if v['condition'] == condition]
                pairs = [(baselines[v['sample_id']], v) for v in rows]
                score_ready = all(v['variant_id'] in scores and b['variant_id'] in scores for b, v in pairs)
                quality_ready = all(passed(v) is not None and passed(b) is not None for b, v in pairs)
                scored = [v for v in rows if z(v) is not None]
                detected = sum(z(v) > threshold for v in scored)
                correct = sum(passed(v) is True for v in rows)
                originally_detected = [(b, v) for b, v in pairs if z(b) is not None and z(b) > threshold]
                eligible = [(b, v) for b, v in originally_detected if passed(b) is True]
                evasions = sum(v['changed'] and passed(v) is True and z(v) is not None and z(v) <= threshold
                               for _, v in eligible)
                baseline_passed = [(b, v) for b, v in pairs if passed(b) is True]
                shifts = [z(v) - z(b) for b, v in pairs if z(v) is not None and z(b) is not None]
                groups.append(dict(attack=attack, condition=condition, samples=len(rows),
                    changed=sum(v['changed'] for v in rows),
                    transform_errors=sum(v['status'] in ('transform_error', 'source_unavailable') for v in rows),
                    scored=len(scored), unscored=len(rows) - len(scored) if score_ready else None,
                    scores_pending=sum(v['variant_id'] not in scores for v in rows),
                    detected=detected, detected_rate=detected / len(rows) if score_ready else None,
                    auroc_vs_clean=auc, paired_scored=len(shifts),
                    mean_paired_z_change=statistics.mean(shifts) if score_ready and shifts else None,
                    evaluated=sum(passed(v) is not None for v in rows), correct=correct,
                    pass_at_1=correct / len(rows) if quality_ready else None,
                    baseline_correct=len(baseline_passed) if quality_ready else None,
                    correctness_retention=(sum(passed(v) is True for _, v in baseline_passed) / len(baseline_passed)
                                           if quality_ready and baseline_passed else None),
                    baseline_detected=len(originally_detected) if score_ready else None,
                    detection_retention=(sum(z(v) is not None and z(v) > threshold for _, v in originally_detected)
                                         / len(originally_detected) if score_ready and originally_detected else None),
                    eligible_correct_detected=len(eligible) if score_ready and quality_ready else None,
                    correct_evasions=evasions if score_ready and quality_ready else None,
                    correct_evasion_rate=evasions / len(eligible) if score_ready and quality_ready and eligible else None))
        complete = len(scores) == len(evaluations) == len(variants)
        summary = dict(status='complete' if complete else 'pending',
                       algorithm=source['config']['watermark'].get('algorithm', 'KGW'), threshold=threshold,
                       robustness_id=manifest['robustness_id'], groups=groups,
                       provenance=dict(source_manifest_sha256=manifest['source_manifest_sha256'],
                                       source_rows_sha256=manifest['source_rows_sha256'],
                                       variants_sha256=manifest['variants_sha256'],
                                       scores_sha256=digest(list(scores.values())),
                                       evaluations_sha256=digest(list(evaluations.values()))))
        write_json(output / 'robustness_summary.json', summary)
        write_csv(output / 'robustness.csv', groups)
        def number(value):
            return 'N/A or pending' if value is None else f'{value:.3f}'
        lines = ['# Code watermark robustness', '', f"Status: **{summary['status']}**; fixed Z > {threshold:g}.", '',
                 'Identity and attacks are scored in the same detector environment. Clean code receives the same attacks. '
                 'AST attacks also normalize formatting and remove comments; compare them with reformat, not only identity.', '',
                 '| Attack | Condition | Changed / total | Scored | Detected rate | AUROC | Pass@1 | Correctness retention | Correct evasions / eligible |',
                 '|---|---|---|---|---|---|---|---|---|']
        for g in groups:
            evasion_count = ('pending' if g['correct_evasions'] is None
                             else f"{g['correct_evasions']} / {g['eligible_correct_detected']}")
            lines.append(f"| {g['attack']} | {g['condition']} | {g['changed']}/{g['samples']} | {g['scored']} | "
                         f"{number(g['detected_rate'])} | {number(g['auroc_vs_clean'])} | {number(g['pass_at_1'])} | "
                         f"{number(g['correctness_retention'])} | {evasion_count} |")
        lines += ['', 'Pass@1 and detected rate include every source sample, including failed transformations. '
                  'AUROC excludes unscored code. Missing stages remain pending. Unscored code is not a confirmed evasion. '
                  'Correct evasions require a changed candidate, passing baseline and attacked code, baseline detection, '
                  'and a valid attacked score at or below the fixed threshold. No eligible samples makes this rate unavailable.', '',
                  'Correctness retention is the fraction of baseline-passing solutions that still pass. '
                  'Detection retention includes unscored attacked outputs as not retained. '
                  'Small detected subsets are not reliable estimates. Test passing is not proof of semantic equivalence. '
                  'These fixed transformations do not measure adaptive or LLM-assisted attacks.', '']
        if complete:
            plt = report_pyplot()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, field, title in zip(axes, ('detected_rate', 'pass_at_1'), ('Detection rate', 'Pass@1')):
                for condition in ('clean', 'watermarked'):
                    values = [g for g in groups if g['condition'] == condition]
                    ax.plot([g['attack'] for g in values], [g[field] for g in values], marker='o', label=condition)
                ax.set(title=title, ylim=(0, 1))
                ax.tick_params(axis='x', rotation=25)
                ax.legend()
            fig.tight_layout()
            fig.savefig(output / 'robustness.png', dpi=150)
            plt.close(fig)
            lines += ['![Detection and correctness after transformations](robustness.png)', '']
        else:
            (output / 'robustness.png').unlink(missing_ok=True)
        (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary
