import argparse
import json

from synthecg.benchmark.localize import run_localization_benchmark, write_localization_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PVC/OT-VA localization algorithms against Zheng EP labels.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Zheng dataset cache directory (default: ~/.cache/synthecg/zheng_otva).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of Zheng records.")
    parser.add_argument(
        "--algorithm",
        choices=["pvc_otva", "auto"],
        default="pvc_otva",
        help="Localization algorithm to evaluate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="localization_benchmark.json",
        help="Output JSON report path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_localization_benchmark(
        cache_dir=args.cache_dir,
        limit=args.limit,
        algorithm=args.algorithm,
    )
    write_localization_report(args.output, summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "results"},
            indent=2,
        )
    )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
