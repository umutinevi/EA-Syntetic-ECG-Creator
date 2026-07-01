import argparse
import json

from synthecg.benchmark.digitize import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run round-trip ECG digitization benchmark.")
    parser.add_argument("dataset_dir", help="Generated dataset directory containing manifest.csv")
    parser.add_argument(
        "--use-augmented",
        action="store_true",
        help="Benchmark on augmented images instead of clean pre-augmentation images.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_benchmark(
        args.dataset_dir,
        use_clean=not args.use_augmented,
        limit=args.limit,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Report written to {args.dataset_dir}/benchmark_report.json")


if __name__ == "__main__":
    main()
