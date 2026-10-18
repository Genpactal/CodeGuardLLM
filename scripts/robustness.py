"""Transform, rescore, and test saved code without regenerating solutions."""

import argparse
from pathlib import Path

from codeguard.attacks import ATTACKS
from codeguard.evaluation import load_run, prepare_docker
from codeguard.robustness import evaluate, prepare, report, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'score', 'evaluate', 'analyze', 'run'))
    parser.add_argument('--run-dir', required=True, type=Path, help='One saved delta run containing manifest.json')
    parser.add_argument('--output', type=Path, help='Default: RUN_DIR/robustness')
    parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument('--markllm-path', type=Path, default=Path('.deps/MarkLLM'))
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'))
    args = parser.parse_args()
    output = args.output or args.run_dir / 'robustness'
    if args.action in ('prepare', 'run'):
        prepare(args.run_dir, output, args.attacks)
    if args.action in ('score', 'run'):
        score(args.run_dir, output, args.markllm_path, args.device, args.dtype)
    if args.action in ('evaluate', 'run'):
        manifest, _, _ = load_run(args.run_dir)
        prepare_docker(manifest['config']['evaluation']['image'])
        evaluate(args.run_dir, output)
    if args.action in ('analyze', 'run'):
        summary = report(args.run_dir, output)
        print(f"Robustness report: {output / 'report.md'} ({summary['status']})")


if __name__ == '__main__':
    main()
