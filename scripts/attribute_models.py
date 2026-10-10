"""Three small models, three keys, and explicit unknown attribution."""

import argparse
from pathlib import Path

from codeguard.analysis import attribution_report
from codeguard.experiments import generate_runs, attribution_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "attribute"])
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--output", default="runs/local/humaneval/attribution")
    parser.add_argument("--delta", type=float, default=2.0)
    args = parser.parse_args()
    configs = attribution_configs(args.config, args.output, args.delta)
    if args.action == "generate":
        generate_runs(configs)
    else:
        destination = Path(args.output) / "analysis"
        attribution_report(
            {label: config["output_dir"] for label, config in configs.items()},
            destination, next(iter(configs.values()))["markllm_path"],
        )
        print(f"Attribution table and plots saved: {destination}")


if __name__ == "__main__":
    main()
