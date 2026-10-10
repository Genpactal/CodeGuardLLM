"""HumanEval watermark comparison, reference controls, Docker evaluation, and report."""

import argparse
from pathlib import Path

from codeguard.controls import score_reference_controls
from codeguard.core_report import core_report
from codeguard.evaluation import evaluate_run, prepare_docker
from codeguard.experiments import generate_runs, comparison_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "controls", "evaluate", "analyze", "run"])
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--output", default="runs/local/humaneval/watermark_comparison")
    args = parser.parse_args()
    configs = comparison_configs(args.config, args.output)
    paths = {label: c["output_dir"] for label, c in configs.items()}
    controls = Path(args.output) / 'reference_controls'
    if args.action in ("generate", "run"):
        generate_runs(configs)
    if args.action in ("controls", "run"):
        first = next(iter(configs.values()))
        score_reference_controls(first['output_dir'], controls, first['markllm_path'])
    if args.action in ("evaluate", "run"):
        prepare_docker(next(iter(configs.values()))['evaluation']['image'])
        for path in paths.values():
            evaluate_run(path)
    if args.action in ("analyze", "run"):
        destination = Path(args.output) / "analysis"
        result = core_report(paths, controls, destination)
        print(f"Core report saved: {destination / 'report.md'} ({result['status']})")


if __name__ == "__main__":
    main()
