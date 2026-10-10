"""Detection and key-attribution reports from saved development runs."""

from copy import deepcopy
import csv
import math
from pathlib import Path
import statistics

from .evaluation import load_run
from .reporting import summarize
from .storage import digest, write_json


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to save")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_runs(run_dirs):
    result = {}
    for label, directory in run_dirs.items():
        directory = Path(directory)
        if not (directory / "manifest.json").is_file():
            raise ValueError(f"Missing run {directory}. Generate it or restore its exported results first.")
        manifest, tasks, rows = load_run(directory)
        if len(rows) != manifest["expected_samples"]:
            raise ValueError(f"Incomplete run: {directory}; resume generation before comparing results")
        result[label] = (manifest, tasks, rows)
    if len({m["tasks_sha256"] for m, _, _ in result.values()}) != 1:
        raise ValueError("Compared experiments must use identical tasks")
    if len({digest(m["config"]["generation"]) for m, _, _ in result.values()}) != 1:
        raise ValueError("Compared experiments must use identical sampling settings")
    return result


def valid_score(score):
    return (score is not None and score.get("status") == "ok"
            and isinstance(score.get("z_score"), (int, float))
            and math.isfinite(score["z_score"]))


def detection_statistics(rows, threshold):
    from sklearn.metrics import roc_auc_score
    scores = {}
    result = []
    for condition in ("clean", "watermarked"):
        selected = [row for row in rows if row["condition"] == condition]
        scored = [row["code_detection"] for row in selected if valid_score(row["code_detection"])]
        scores[condition] = [score["z_score"] for score in scored]
        result.append({
            "condition": condition, "samples": len(selected), "scored": len(scored),
            "unscored": len(selected) - len(scored),
            "mean_z": statistics.mean(scores[condition]) if scored else None,
            "mean_scored_tokens": statistics.mean(score["scored_tokens"] for score in scored) if scored else None,
            "positive_rate": sum(z > threshold for z in scores[condition]) / len(scored) if scored else None,
        })
    auc = None
    if scores["clean"] and scores["watermarked"]:
        auc = float(roc_auc_score([0] * len(scores["clean"]) + [1] * len(scores["watermarked"]),
                                 scores["clean"] + scores["watermarked"]))
    return result, auc, scores


def detection_report(run_dirs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    runs = complete_runs(run_dirs)
    # A delta sweep must differ only in bias strength and run identity.
    protocols = []
    for manifest, _, _ in runs.values():
        config = deepcopy(manifest["config"])
        config.pop("name")
        config["watermark"].pop("delta")
        protocols.append(digest(config))
    if len(set(protocols)) != 1:
        raise ValueError("A strength comparison may vary delta only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, samples = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, (manifest, _, rows) in runs.items():
        wm = manifest["config"]["watermark"]
        stats, auc, scores = detection_statistics(rows, wm["z_threshold"])
        quality = summarize(run_dirs[label])
        for row in stats:
            metrics.append({"run": label, "delta": wm["delta"], **row,
                            "threshold": wm["z_threshold"], "auroc": auc,
                            "evaluated": quality["conditions"][row["condition"]]["evaluated"],
                            "pass_at_1": quality["conditions"][row["condition"]]["pass_at_1"]})
        for row in rows:
            score = row["code_detection"] or {}
            samples.append({"run": label, "task_id": row["task_id"], "seed": row["seed"],
                            "condition": row["condition"], "extraction": row["extraction"]["status"],
                            "scored_tokens": score.get("scored_tokens"), "z_score": score.get("z_score")})
        for condition, values in scores.items():
            if values:
                axes[0].plot(sorted(values), label=f"{label} {condition}",
                             linestyle="--" if condition == "clean" else "-", marker=".")
        if auc is not None:
            fpr, tpr, _ = roc_curve([0] * len(scores["clean"]) + [1] * len(scores["watermarked"]),
                                   scores["clean"] + scores["watermarked"])
            axes[1].plot(fpr, tpr, label=f"{label}: AUC={auc:.3f}")
    axes[0].set(xlabel="Sample rank (sorted within group)", ylabel="Z-score", title="Score distributions")
    axes[0].legend(fontsize=7)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC: scorable code only")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "detection.png", dpi=160)
    plt.close(fig)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "samples.csv", samples)
    write_json(output / "metrics.json", metrics)
    write_json(output / "provenance.json", {label: {"experiment_id": m["experiment_id"], "records_sha256": digest(rows)}
                                           for label, (m, _, rows) in runs.items()})
    return metrics


def choose_key(scores, threshold):
    """Compare all candidates; missing scores and ties return unknown."""
    if not scores or any(value is None or not math.isfinite(value) for value in scores.values()):
        return "unknown", "unknown"
    best = max(scores.values())
    winners = [key for key, value in scores.items() if value == best]
    if len(winners) != 1:
        return "unknown", "unknown"
    winner = winners[0]
    return winner, winner if best > threshold else "unknown"


def attribution_metrics(records, labels):
    positives = [r for r in records if r["condition"] == "watermarked"]
    negatives = [r for r in records if r["condition"] == "clean"]
    def rate(count, rows):
        return count / len(rows) if rows else None
    matrix = {truth: {prediction: 0 for prediction in [*labels, "unknown"]}
              for truth in [*labels, "clean"]}
    forced_matrix = deepcopy(matrix)
    for row in records:
        truth = row["source"] if row["condition"] == "watermarked" else "clean"
        matrix[truth][row["prediction"]] += 1
        forced_matrix[truth][row["forced_prediction"]] += 1
    return {
        "watermarked_samples": len(positives), "clean_samples": len(negatives),
        "scored_samples": sum(r["scored"] for r in records),
        "unscored_samples": sum(not r["scored"] for r in records),
        "forced_accuracy": rate(sum(r["forced_prediction"] == r["source"] for r in positives), positives),
        "thresholded_accuracy": rate(sum(r["prediction"] == r["source"] for r in positives), positives),
        "attribution_coverage": rate(sum(r["prediction"] != "unknown" for r in positives), positives),
        "clean_false_attribution_rate": rate(sum(r["prediction"] != "unknown" for r in negatives), negatives),
        "confusion_matrix": matrix,
        "forced_confusion_matrix": forced_matrix,
    }


def attribution_report(run_dirs, output, markllm_path, threshold=4.0, backend_factory=None):
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if backend_factory is None:
        from .backend import HFBackend
        backend_factory = HFBackend
    runs = complete_runs(run_dirs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for source, (_, _, rows) in runs.items():
        for row in rows:
            records.append({"source": source, "sample_id": row["sample_id"], "task_id": row["task_id"],
                            "condition": row["condition"], "text": row["extraction"]["code"], "scores": {}})
    for label, (manifest, _, _) in runs.items():
        config = deepcopy(manifest["config"])
        config["markllm_path"] = str(Path(markllm_path).resolve())
        detector = backend_factory(config, load_model=False)
        if detector.metadata != manifest["backend"]:
            raise ValueError(f"Detector environment for {label} differs from generation; use matching versions and device")
        print(f"Scoring every sample against {label}", flush=True)
        for row in records:
            result = detector.score(row["text"]) if row["text"] is not None else None
            row["scores"][label] = result["z_score"] if valid_score(result) else None
        del detector
    for row in records:
        row.pop("text")
        row["scored"] = all(value is not None for value in row["scores"].values())
        row["forced_prediction"], row["prediction"] = choose_key(row["scores"], threshold)
    summary = attribution_metrics(records, list(runs))
    summary.update(threshold=threshold, scope="Exploratory registered key/model attribution; uncalibrated maximum-Z threshold")
    write_json(output / "summary.json", summary)
    write_json(output / "scores.json", records)
    flattened = [{k: v for k, v in row.items() if k != "scores"} | {f"z_{key}": value for key, value in row["scores"].items()}
                 for row in records]
    write_csv(output / "scores.csv", flattened)
    write_json(output / "provenance.json", {label: {"experiment_id": m["experiment_id"], "records_sha256": digest(rows)}
                                           for label, (m, _, rows) in runs.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = [*runs, "unknown"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, field, title in zip(axes, ("forced_confusion_matrix", "confusion_matrix"),
                               ("Highest Z (no rejection threshold)", f"Highest Z > {threshold:g}")):
        matrix = summary[field]
        values = [[row[c] for c in columns] for row in matrix.values()]
        ax.imshow(values, cmap="Blues")
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(matrix)),
               yticklabels=list(matrix), xlabel="Predicted key", ylabel="True source", title=title)
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)
    return summary
