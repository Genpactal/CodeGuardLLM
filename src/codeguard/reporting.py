"""Pilot summaries. Missing evaluations never become zero Pass@1 by accident."""

from collections import Counter
from pathlib import Path
import statistics

from .evaluation import load_run
from .storage import digest, read_records, run_lock, write_json


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
                   "scope": "Descriptive pilot results; no calibration, confidence intervals, or attribution yet",
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
