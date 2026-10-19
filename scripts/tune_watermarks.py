"""Prepare and run a matched, resumable KGW/SWEET development sweep."""

import argparse

from codeguard.tuning import analyze_suite, controls_suite, evaluate_suite, generate_suite, prepare_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'generate', 'controls', 'analyze', 'evaluate'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/gpu.yaml', help='Used only by prepare')
    parser.add_argument('--tasks', type=int, default=64)
    parser.add_argument('--calibration-tasks', type=int, default=32)
    parser.add_argument('--split-seed', type=int, default=20261019)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--target-fpr', type=float, default=0.05)
    parser.add_argument('--max-quality-loss', type=float, default=0.05)
    parser.add_argument('--markllm-path', default='.deps/MarkLLM')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare_suite(args.config, args.output, args.tasks, args.calibration_tasks,
                      args.split_seed, args.seeds, args.profile, args.target_fpr, args.max_quality_loss)
    elif args.action == 'generate':
        generate_suite(args.output, args.markllm_path)
    elif args.action == 'controls':
        controls_suite(args.output, args.markllm_path)
    elif args.action == 'evaluate':
        evaluate_suite(args.output)
    else:
        summary = analyze_suite(args.output)
        print(f"Report: {args.output}/analysis/report.md ({summary['status']})")


if __name__ == '__main__':
    main()
