import argparse
import sys

from synthecg.config import AugmentConfig, GenerationConfig, RenderConfig
from synthecg.pipeline import generate_dataset
from synthecg.recipes.builder import config_from_recipe
from synthecg.export.huggingface import prepare_hf_export, push_to_hub
from synthecg.recipes.definitions import list_recipes


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of ECG images to generate.")
    parser.add_argument(
        "-t",
        "--type",
        dest="diagnosis",
        type=str,
        default="random",
        help="Pathology SCP code (e.g. AFIB, NORM, PVC) or 'random'.",
    )
    parser.add_argument("-o", "--output-dir", type=str, default="output_ecgs", help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "val", "test"],
        default="all",
        help="PTB-XL stratified split.",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="PTB-XL metadata cache directory.")
    parser.add_argument("--unique-patients", action="store_true", help="One record per patient.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument("--resume", action="store_true", help="Skip ecg_ids already in manifest.csv.")
    parser.add_argument("--include-codes", nargs="+", default=[], help="Require all SCP codes (multi-label).")
    parser.add_argument("--exclude-codes", nargs="+", default=[], help="Exclude records with these SCP codes.")
    parser.add_argument("--renderer", choices=["opencv", "matplotlib"], default="opencv")
    parser.add_argument(
        "--layout",
        choices=["3x4+1", "12x1"],
        default="3x4+1",
        help="ECG page layout (default: 3x4+1).",
    )
    parser.add_argument("--speed", type=int, default=25, choices=[25, 50], help="Paper speed mm/s.")
    parser.add_argument("--gain", type=int, default=10, choices=[5, 10, 20], help="Voltage gain mm/mV.")
    parser.add_argument("--no-grid", action="store_true", help="Render without ECG grid lines.")
    parser.add_argument(
        "--augment-profile",
        choices=["clean", "scan", "clinical"],
        default="scan",
    )
    parser.add_argument("--bandpass", action="store_true", help="Apply 0.5-40 Hz bandpass before rendering.")
    parser.add_argument("--save-clean", action="store_true", help="Save pre-augmentation images.")
    parser.add_argument("--no-signals", action="store_true")
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--no-yolo", action="store_true")


def _config_from_generate_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        count=args.count,
        diagnosis=args.diagnosis,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        split=args.split,
        unique_patients=args.unique_patients,
        workers=max(1, args.workers),
        resume=args.resume,
        include_codes=args.include_codes,
        exclude_codes=args.exclude_codes,
        bandpass_filter=args.bandpass,
        export_signals=not args.no_signals,
        export_annotations=not args.no_annotations,
        export_masks=not args.no_masks,
        export_yolo=not args.no_yolo,
        save_clean=args.save_clean,
        render=RenderConfig(
            backend=args.renderer,
            layout=args.layout,
            speed_mm_s=args.speed,
            gain_mm_mv=args.gain,
            show_grid=not args.no_grid,
        ),
        augment=AugmentConfig(profile=args.augment_profile, seed=args.seed),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic realistic 12-lead ECG images from PTB-XL.",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Generate ECG images (default command).")
    _add_generate_args(generate_parser)

    dataset_parser = subparsers.add_parser("dataset", help="Build predefined dataset recipes.")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")

    build_parser = dataset_sub.add_parser("build", help="Build a named dataset recipe.")
    build_parser.add_argument("--recipe", required=True, help="Recipe name (see 'dataset list').")
    build_parser.add_argument("-o", "--output-dir", required=True, help="Output directory.")
    build_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    build_parser.add_argument("--workers", type=int, default=None, help="Override recipe worker count.")
    build_parser.add_argument("--resume", action="store_true", help="Resume from existing manifest.")

    list_parser = dataset_sub.add_parser("list", help="List available dataset recipes.")

    export_parser = subparsers.add_parser("export", help="Export datasets to external formats.")
    export_sub = export_parser.add_subparsers(dest="export_command")
    hf_parser = export_sub.add_parser("hf", help="Prepare or push a Hugging Face dataset.")
    hf_parser.add_argument("-d", "--dataset-dir", required=True, help="Generated dataset directory.")
    hf_parser.add_argument("-o", "--export-dir", default=None, help="Local HF export folder.")
    hf_parser.add_argument("--layout", default="3x4+1", choices=["3x4+1", "12x1"])
    hf_parser.add_argument("--repo-id", default=None, help="Hugging Face dataset repo id (user/name).")
    hf_parser.add_argument("--push", action="store_true", help="Upload to Hugging Face Hub.")

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility: `synthecg -n 5 -t NORM` without subcommand
    if argv and argv[0] not in {"generate", "dataset", "export"}:
        argv = ["generate", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate" or (args.command is None and hasattr(args, "count")):
            generate_dataset(_config_from_generate_args(args))
        elif args.command == "dataset":
            if args.dataset_command == "list":
                recipes = list_recipes()
                print("Available dataset recipes:\n")
                for name, description in recipes.items():
                    print(f"  {name:20} {description}")
            elif args.dataset_command == "build":
                config = config_from_recipe(
                    args.recipe,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    resume=args.resume,
                    workers=args.workers,
                )
                print(f"Building recipe '{args.recipe}' -> {args.output_dir}")
                generate_dataset(config)
            else:
                parser.print_help()
                raise SystemExit(1)
        elif args.command == "export":
            if args.export_command == "hf":
                if args.push:
                    if not args.repo_id:
                        raise SystemExit("--repo-id is required when using --push")
                    url = push_to_hub(
                        args.dataset_dir,
                        args.repo_id,
                        layout=args.layout,
                        export_dir=args.export_dir,
                    )
                    print(f"Uploaded to {url}")
                else:
                    out = prepare_hf_export(
                        args.dataset_dir,
                        export_dir=args.export_dir,
                        layout=args.layout,
                    )
                    print(f"HF export prepared at {out}")
            else:
                parser.print_help()
                raise SystemExit(1)
        else:
            parser.print_help()
            raise SystemExit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
