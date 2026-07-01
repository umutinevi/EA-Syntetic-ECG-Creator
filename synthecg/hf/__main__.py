"""Hugging Face publish automation CLI."""

from __future__ import annotations

import argparse
import json
import sys

from synthecg.export.huggingface import detect_layout, prepare_hf_export, push_to_hub
from synthecg.export.publish import HFPublishConfig, publish_to_hub


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate SynthECG dataset generation and Hugging Face publishing.",
    )
    sub = parser.add_subparsers(dest="hf_command")

    publish = sub.add_parser("publish", help="Generate, export, and optionally push to HF Hub.")
    publish.add_argument("--config", type=str, help="YAML publish config file.")
    publish.add_argument("--recipe", type=str, help="Dataset recipe to generate before export.")
    publish.add_argument("-o", "--output-dir", default="hf_dataset", help="Local dataset directory.")
    publish.add_argument("-d", "--dataset-dir", default=None, help="Existing dataset (skip generation).")
    publish.add_argument("--repo-id", required=False, help="HF dataset repo id (user/name).")
    publish.add_argument("-n", "--count", type=int, default=None, help="Override recipe sample count.")
    publish.add_argument("--seed", type=int, default=42)
    publish.add_argument("--workers", type=int, default=None)
    publish.add_argument("--split", choices=["all", "train", "val", "test"], default="train")
    publish.add_argument("--layout", choices=["3x4+1", "12x1"], default=None)
    publish.add_argument("--resume", action="store_true")
    publish.add_argument("--benchmark", action="store_true", help="Run digitization benchmark before export.")
    publish.add_argument("--private", action="store_true", help="Create a private HF dataset repo.")
    publish.add_argument(
        "--format",
        choices=["folder", "datasets"],
        default="folder",
        help="folder=full assets upload; datasets=HF Datasets API (metadata+images only).",
    )
    publish.add_argument("--push", action="store_true", default=False, help="Push to Hugging Face Hub.")
    publish.add_argument("--no-push", action="store_true", help="Only prepare export locally.")
    publish.add_argument("--export-dir", default=None, help="Local HF export folder path.")

    prepare = sub.add_parser("prepare", help="Prepare HF export from an existing dataset.")
    prepare.add_argument("-d", "--dataset-dir", required=True)
    prepare.add_argument("-o", "--export-dir", default=None)
    prepare.add_argument("--repo-id", default=None)
    prepare.add_argument("--layout", choices=["3x4+1", "12x1"], default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.hf_command == "publish":
            if args.config:
                config = HFPublishConfig.from_yaml(args.config)
            else:
                if not args.repo_id and args.push:
                    raise SystemExit("--repo-id is required when using --push (or use --no-push).")
                output_dir = args.dataset_dir or args.output_dir
                config = HFPublishConfig(
                    repo_id=args.repo_id or "local/synthecg-export",
                    output_dir=output_dir,
                    recipe=args.recipe,
                    count=args.count,
                    seed=args.seed,
                    workers=args.workers,
                    split=args.split,
                    layout=args.layout,
                    resume=args.resume,
                    benchmark=args.benchmark,
                    push=args.push and not args.no_push,
                    private=args.private,
                    format=args.format,
                    export_dir=args.export_dir,
                )
            result = publish_to_hub(config)
            print(json.dumps({k: v for k, v in result.items() if k != "benchmark"}, indent=2))

        elif args.hf_command == "prepare":
            layout = args.layout or detect_layout(args.dataset_dir)
            out = prepare_hf_export(
                args.dataset_dir,
                export_dir=args.export_dir,
                layout=layout,
                repo_id=args.repo_id,
            )
            print(f"HF export prepared at {out}")

        else:
            parser.print_help()
            raise SystemExit(1)

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
