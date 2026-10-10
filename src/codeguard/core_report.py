"""Core research report: detection, reference controls, and correctness trade-offs."""

import math
from pathlib import Path
import statistics

from .analysis import complete_runs, detection_report, valid_score, write_csv
from .controls import load_reference_controls
from .reporting import summarize
from .storage import digest, write_json


def group_statistics(rows, threshold):
    scored = [r['code_detection'] for r in rows if valid_score(r['code_detection'])]
    return {'samples': len(rows), 'scored': len(scored), 'unscored': len(rows) - len(scored),
            'mean_z': statistics.mean(s['z_score'] for s in scored) if scored else None,
            'positive_rate': sum(s['z_score'] > threshold for s in scored) / len(scored) if scored else None,
            'mean_scored_tokens': statistics.mean(s['scored_tokens'] for s in scored) if scored else None}


def roc_statistics(negative, positive):
    from sklearn.metrics import roc_auc_score, roc_curve
    neg = [r['code_detection']['z_score'] for r in negative if valid_score(r['code_detection'])]
    pos = [r['code_detection']['z_score'] for r in positive if valid_score(r['code_detection'])]
    if not neg or not pos:
        return None, []
    labels, scores = [0] * len(neg) + [1] * len(pos), neg + pos
    fpr, tpr, thresholds = roc_curve(labels, scores)
    points = [{'fpr': float(f), 'tpr': float(t), 'threshold': float(z) if math.isfinite(z) else None}
              for f, t, z in zip(fpr, tpr, thresholds)]
    return float(roc_auc_score(labels, scores)), points


def core_report(run_dirs, controls_dir, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = complete_runs(run_dirs)
    controls = {}
    # Validate all inputs before writing any report artifacts.
    for label, (manifest, tasks, _) in runs.items():
        controls[label] = load_reference_controls(controls_dir, manifest, tasks)
    output = Path(output)
    detection_report(run_dirs, output)
    comparison, groups, roc_rows, lengths, sample_rows, examples = [], [], [], [], [], []
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 3.6 * len(runs)), squeeze=False)
    colors = {'clean': 'tab:blue', 'watermarked': 'tab:orange', 'human_reference': 'tab:green'}
    quality_complete = True
    for index, (label, (manifest, tasks, rows)) in enumerate(runs.items()):
        wm = manifest['config']['watermark']
        quality = summarize(run_dirs[label])
        quality_complete &= quality['evaluated_samples'] == manifest['expected_samples']
        selected = {condition: [r for r in rows if r['condition'] == condition]
                    for condition in ('clean', 'watermarked')}
        selected['human_reference'] = controls[label]
        scores = {condition: [r['code_detection']['z_score'] for r in values if valid_score(r['code_detection'])]
                  for condition, values in selected.items()}
        all_scores = [z for values in scores.values() for z in values]
        import numpy as np
        bins = np.histogram_bin_edges(all_scores, bins='auto') if all_scores else [0, 1]
        for condition, values in selected.items():
            stats = group_statistics(values, wm['z_threshold'])
            groups.append({'run': label, 'delta': wm['delta'], 'condition': condition,
                           'threshold': wm['z_threshold'], **stats})
            if scores[condition]:
                axes[index, 0].hist(scores[condition], bins=bins, histtype='step', linewidth=1.8,
                                    label=condition, color=colors[condition])
            for row in values:
                score = row['code_detection'] or {}
                sample_rows.append({'run': label, 'task_id': row['task_id'], 'seed': row.get('seed'),
                                    'condition': condition, 'status': score.get('status', 'unscored'),
                                    'scored_tokens': score.get('scored_tokens'), 'z_score': score.get('z_score')})
            for low, high, bucket in ((0, 50, '<50'), (50, 100, '50-99'), (100, math.inf, '100+')):
                bucket_rows = [r for r in values if valid_score(r['code_detection'])
                               and low <= r['code_detection']['scored_tokens'] < high]
                lengths.append({'run': label, 'condition': condition, 'scored_token_range': bucket,
                                **group_statistics(bucket_rows, wm['z_threshold'])})
        aucs = {}
        for negative in ('clean', 'human_reference'):
            auc, points = roc_statistics(selected[negative], selected['watermarked'])
            aucs[negative] = auc
            roc_rows.extend({'run': label, 'negative_control': negative, **point} for point in points)
            if points:
                axes[index, 1].plot([p['fpr'] for p in points], [p['tpr'] for p in points],
                                    label=f'vs {negative}: AUC={auc:.3f}', color=colors[negative])
        clean, watermarked = (quality['conditions'][c] for c in ('clean', 'watermarked'))
        gap = None if clean['pass_at_1'] is None or watermarked['pass_at_1'] is None else watermarked['pass_at_1'] - clean['pass_at_1']
        comparison.append({'run': label, 'delta': wm['delta'], 'auroc_vs_clean': aucs['clean'],
                           'auroc_vs_human_reference': aucs['human_reference'],
                           'clean_pass_at_1': clean['pass_at_1'], 'watermarked_pass_at_1': watermarked['pass_at_1'],
                           'pass_at_1_change': gap, 'clean_truncated': clean['truncated'],
                           'watermarked_truncated': watermarked['truncated'],
                           'clean_evaluation_statuses': clean['evaluation_statuses'],
                           'watermarked_evaluation_statuses': watermarked['evaluation_statuses']})
        # Examples are selected by detector score only, never executed or repaired.
        scored = [r for r in selected['watermarked'] if valid_score(r['code_detection'])]
        if scored:
            ordered = sorted(scored, key=lambda r: r['code_detection']['z_score'])
            for kind, row in (('lowest_watermarked_z', ordered[0]), ('highest_watermarked_z', ordered[-1])):
                examples.append({'run': label, 'selection': kind, 'task_id': row['task_id'],
                                 'score': row['code_detection'], 'code': row['extraction']['code']})
        axes[index, 0].axvline(wm['z_threshold'], linestyle=':', color='black', label='fixed threshold')
        axes[index, 0].set(title=f'{label}: score distributions', xlabel='Z-score', ylabel='Samples')
        axes[index, 1].plot([0, 1], [0, 1], 'k--', alpha=.35)
        axes[index, 1].set(title=f'{label}: scorable-code ROC', xlabel='False-positive rate', ylabel='True-positive rate')
        for ax in axes[index]:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'controls_detection.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    delta = [r['delta'] for r in comparison]
    for field, label in (('auroc_vs_clean', 'vs clean'), ('auroc_vs_human_reference', 'vs human reference')):
        axes[0].plot(delta, [r[field] if r[field] is not None else math.nan for r in comparison], marker='o', label=label)
    for field, label in (('clean_pass_at_1', 'clean'), ('watermarked_pass_at_1', 'watermarked')):
        axes[1].plot(delta, [r[field] if r[field] is not None else math.nan for r in comparison], marker='o', label=label)
    axes[0].set(xlabel='Watermark strength (delta)', ylabel='AUROC', ylim=(0, 1), title='Detection')
    axes[1].set(xlabel='Watermark strength (delta)', ylabel='Pass@1', ylim=(0, 1), title='Correctness')
    if not quality_complete:
        axes[1].text(.5, .5, 'Correctness evaluation incomplete', transform=axes[1].transAxes, ha='center')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'tradeoff.png', dpi=150)
    plt.close(fig)

    summary = {'status': 'complete' if quality_complete else 'correctness_pending',
               'generated_samples': sum(len(rows) for _, _, rows in runs.values()),
               'reference_samples': len(next(iter(controls.values()))),
               'reference_controls_shared_across_strengths': True,
               'comparison': comparison, 'groups': groups,
               'scope': 'Descriptive results at a fixed threshold; no held-out tuning or causal entropy measurement.'}
    write_json(output / 'core_summary.json', summary)
    write_json(output / 'examples.json', examples)
    write_csv(output / 'detection_groups.csv', groups)
    write_csv(output / 'detector_samples.csv', sample_rows)
    write_csv(output / 'length_analysis.csv', lengths)
    write_csv(output / 'quality_detection.csv', comparison)
    if roc_rows:
        write_csv(output / 'roc.csv', roc_rows)
    else:
        (output / 'roc.csv').write_text('run,negative_control,fpr,tpr,threshold\n', encoding='utf-8')
    write_json(output / 'core_provenance.json', {
        label: {'experiment_id': m['experiment_id'], 'generation_sha256': digest(rows),
                'reference_scores_sha256': digest(controls[label]),
                'quality_summary_sha256': digest(summarize(run_dirs[label]))}
        for label, (m, _, rows) in runs.items()})
    _write_report(output, summary, runs, lengths)
    return summary


def _write_report(output, summary, runs, lengths):
    def number(value):
        return 'pending / unavailable' if value is None else f'{value:.3f}'
    first = next(iter(runs.values()))[0]
    config = first['config']
    wm = config['watermark']
    lines = ['# CodeGuardLLM: core experiment', '',
             f"Status: **{summary['status']}**. {summary['generated_samples']} generated completions; "
             f"{summary['reference_samples']} distinct reference controls shared across strengths.", '',
             '## Method', '',
             f"Model: `{config['model']['name']}` at `{config['model']['revision']}`. "
             f"Device/precision: {config['model']['device']}/{config['model']['dtype']}.", '',
             f"Dataset: `{config['dataset'].get('name', 'project development fixtures')}`; "
             f"revision `{config['dataset'].get('revision', 'local snapshot')}`. "
             'Only task prompts entered generation. Tests and reference solutions were withheld.', '',
             f"KGW gamma={wm['gamma']}, fixed Z > {wm['z_threshold']}, key={wm['hash_key']}, "
             f"scheme={wm['f_scheme']}/{wm['window_scheme']}, repeated-bigram filtering={wm['ignore_repeated_ngrams']}.", '',
             f"Sampling: {config['generation']}. Model and runtime details are preserved in run manifests.", '',
             'Z = (green_count - gamma × T) / sqrt(T × gamma × (1 - gamma)). '
             'T counts scored distinct bigrams here; the first token is unscored. '
             'Generated code is scored without prepending its prompt. HumanEval references use the complete '
             'prompt + canonical_solution, retaining imports, signature and docstrings; this representation '
             'can differ from generated code.', '',
             '## Detection and correctness', '',
             '| Strength | AUROC vs clean | AUROC vs reference | Clean Pass@1 | Watermarked Pass@1 | Change |',
             '|---|---|---|---|---|---|']
    for r in summary['comparison']:
        lines.append(f"| {r['delta']:g} | {number(r['auroc_vs_clean'])} | {number(r['auroc_vs_human_reference'])} | "
                     f"{number(r['clean_pass_at_1'])} | {number(r['watermarked_pass_at_1'])} | {number(r['pass_at_1_change'])} |")
    lines += ['', '![Detection distributions and ROC](controls_detection.png)', '',
              '![Detection and correctness](tradeoff.png)', '',
              'AUROC uses only scorable code. Positive rates below use the fixed threshold; '
              'clean/reference rates are false alarms. Missing scores stay explicitly unscored. '
              'Pass@1 includes extraction failures, invalid syntax and timeouts as failures after evaluation; '
              'missing evaluation is pending, never zero correctness.', '',
              '| Run | Group | Scored / total | Mean Z | Positive rate |', '|---|---|---|---|---|']
    for r in summary['groups']:
        lines.append(f"| {r['run']} | {r['condition']} | {r['scored']}/{r['samples']} | {number(r['mean_z'])} | {number(r['positive_rate'])} |")
    lines += ['', '## Code length', '',
              'Scored lengths are split into <50, 50–99, and 100+ bigrams. Empty bins remain unavailable. '
              'This is a descriptive association, not a measurement of model entropy.', '',
              '| Run | Group | Length | Scored | Mean Z | Positive rate |', '|---|---|---|---|---|---|']
    for r in lengths:
        if r['scored']:
            lines.append(f"| {r['run']} | {r['condition']} | {r['scored_token_range']} | {r['scored']} | {number(r['mean_z'])} | {number(r['positive_rate'])} |")
    lines += ['', '## Findings', '']
    for r in summary['comparison']:
        lines.append(f"- Delta {r['delta']:g}: AUROC vs clean {number(r['auroc_vs_clean'])}; "
                     f"vs reference {number(r['auroc_vs_human_reference'])}; "
                     f"watermarked-minus-clean Pass@1 {number(r['pass_at_1_change'])}. "
                     f"Truncated outputs: clean {r['clean_truncated']}, watermarked {r['watermarked_truncated']}.")
    lines += ['', '## Interpretation and limits', '',
              'The original KGW work studies softly promoted green tokens in natural-language generation with OPT. '
              'Here the model, task domain, prompting, output lengths and repeated-bigram handling differ; '
              'its reported detection performance is not a numerical benchmark for this experiment. '
              'Short outputs supply fewer scored trials. Syntax constraints and boilerplate may weaken the '
              'signal, but these experiments do not directly test that causal explanation. '
              '[KGW paper](https://arxiv.org/abs/2301.10226).', '',
              'HumanEval reference solutions are public benchmark controls, not an independent human-code corpus. '
              'Model training contamination cannot be excluded. Our chat/full-function extraction protocol '
              'and container limits differ from the official completion evaluation; compare Pass@1 values cautiously. '
              '[HumanEval implementation](https://github.com/openai/human-eval).', '',
              'The Z threshold is uncalibrated on code. No guaranteed false-positive probability, '
              'confidence interval, or held-out tuning result is claimed. Clean generations repeated across '
              'strengths and shared references are not independent extra controls. A strength selected on '
              'these same tasks needs separate validation. Fingerprinting and robustness are outside this report.', '',
              '## Artifacts', '',
              '`core_summary.json`, `detection_groups.csv`, `quality_detection.csv`, `detector_samples.csv`, '
              '`roc.csv`, `length_analysis.csv`, `examples.json`, and `core_provenance.json`. '
              'Examples show the lowest/highest watermarked Z per run; they are not representative accuracy estimates.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
