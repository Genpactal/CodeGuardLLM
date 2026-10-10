"""Small command line interface usable from PowerShell, Linux, and notebooks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .data import load_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="codeguard", description="KGW code watermark experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "generate", "detect"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/local.yaml")
        if name == "validate":
            sub.add_argument("--check-markllm", action="store_true")
        if name == "generate":
            sub.add_argument("--output-dir", type=Path, help="Persistent run directory; overrides the config path")
        if name == "detect":
            sub.add_argument("--file", type=Path, required=True, help="Generated code only, without its input prompt")
    for name in ("evaluate", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, default=Path("runs/local/humaneval/baseline"))
    args = parser.parse_args(argv)
    try:
        if args.command in ("validate", "generate", "detect"):
            config = load_config(args.config)
        if args.command == "validate":
            tasks = load_tasks(config["dataset"])
            if args.check_markllm:
                from .watermark import check_markllm
                check_markllm(config["markllm_path"])
            print(json.dumps({"name": config["name"], "tasks": len(tasks),
                              "planned_completions": len(tasks) * len(config["generation"]["seeds"]) * 2,
                              "output_dir": config["output_dir"]}, indent=2))
        elif args.command == "generate":
            from .pipeline import generate
            if args.output_dir:
                config["output_dir"] = str(args.output_dir.resolve())
            generate(config)
        elif args.command == "detect":
            from .backend import HFBackend
            text = args.file.read_text(encoding="utf-8")
            backend = HFBackend(config, load_model=False)
            print(json.dumps({"score": backend.score(text), "detector": backend.metadata}, indent=2))
        elif args.command == "evaluate":
            from .evaluation import evaluate_run
            evaluate_run(args.run_dir)
        else:
            from .reporting import summarize
            print(json.dumps(summarize(args.run_dir), indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"codeguard: {exc}", file=sys.stderr)
        return 1
    return 0
