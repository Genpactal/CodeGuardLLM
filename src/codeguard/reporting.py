"""Run summaries. Missing evaluations never become zero Pass@1 by accident."""

from collections import Counter
import math
from pathlib import Path
import statistics

from .evaluation import load_run
from .storage import digest, read_records, run_lock, write_json


def detected_correctness(rows, evaluations, threshold, expected):
    """Correctness conditional on detection, with a separate joint success rate."""
    scored = [row for row in rows if row.get('code_detection')
              and row['code_detection'].get('status') == 'ok'
              and isinstance(row['code_detection'].get('z_score'), (int, float))
              and math.isfinite(row['code_detection']['z_score'])]
    detected = [row for row in scored if row['code_detection']['z_score'] > threshold]
    evaluated = [evaluations[row['sample_id']] for row in detected if row['sample_id'] in evaluations]
    correct = sum(result['status'] == 'passed' for result in evaluated)
    generated_all = len(rows) == expected
    evaluated_all = generated_all and all(row['sample_id'] in evaluations for row in rows)
    return {'threshold': threshold, 'samples': len(rows), 'scored': len(scored),
            'detected': len(detected), 'evaluated_detected': len(evaluated),
            'correct_detected': correct,
            'correctness_among_detected': correct / len(detected)
                if generated_all and detected and len(evaluated) == len(detected) else None,
            'correct_and_detected_rate': correct / expected if evaluated_all and expected else None}


def summarize(directory):
    directory = Path(directory)
    with run_lock(directory):
        manifest, _, rows = load_run(directory)
        evaluations = read_records(directory / "evaluations.jsonl")
        indexed = {row["sample_id"]: row for row in evaluations}
        originals = {row["sample_id"]: row for row in rows}
        if len(indexed) != len(evaluations) or not indexed.keys() <= originals.keys():
            raise ValueError("Duplicate or unknown evaluation records")
        for identity, result in indexed.items():
            if result["generation_sha256"] != digest(originals[identity]):
                raise ValueError("Generation changed after evaluation")
        summary = {"experiment_id": manifest["experiment_id"], "expected_samples": manifest["expected_samples"],
                   "generated_samples": len(rows), "evaluated_samples": len(evaluations),
                   "scope": "Descriptive run results at a fixed threshold; no calibration or confidence intervals",
                   "conditions": {}}
        for condition in ("clean", "watermarked"):
            selected = [row for row in rows if row["condition"] == condition]
            statuses = Counter(indexed[row["sample_id"]]["status"] for row in selected if row["sample_id"] in indexed)
            scores = [row["code_detection"] for row in selected if row["code_detection"] and row["code_detection"]["status"] == "ok"]
            seconds = sum(row["generation_seconds"] for row in selected)
            expected = manifest["expected_samples"] // 2
            summary["conditions"][condition] = {
                "expected": expected, "generated": len(selected), "evaluated": sum(statuses.values()),
                "evaluation_statuses": dict(statuses),
                "pass_at_1": statuses["passed"] / expected if sum(statuses.values()) == expected else None,
                "detected_correctness": detected_correctness(
                    selected, indexed, manifest['config']['watermark']['z_threshold'], expected),
                "syntax_valid": sum(row["extraction"]["status"] in ("ok", "missing_entrypoint") for row in selected),
                "truncated": sum(row["finish_reason"] == "length" for row in selected),
                "scored_samples": len(scores), "unscored_samples": len(selected) - len(scores),
                "mean_z": statistics.mean(score["z_score"] for score in scores) if scores else None,
                "positive_rate_among_scored": sum(score["is_watermarked"] for score in scores) / len(scores) if scores else None,
                "generation_seconds": seconds,
                "generation_tokens_per_second": sum(row["generated_tokens"] for row in selected) / seconds if seconds else None,
            }
        write_json(directory / "summary.json", summary)
        return summary
