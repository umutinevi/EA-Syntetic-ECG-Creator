import argparse

from synthecg.config import AugmentConfig, GenerationConfig, RenderConfig
from synthecg.pipeline import generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic realistic 12-lead ECG images from PTB-XL.",
    )
    parser.add_argument("-n", "--count", type=int, default=5, help="Number of ECG images to generate.")
    parser.add_argument(
        "-t",
        "--type",
        dest="diagnosis",
        type=str,
        default="random",
        help="Pathology SCP code (e.g. AFIB, NORM, PVC) or 'random'.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output_ecgs",
        help="Directory to save the generated dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible record selection and augmentations.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "val", "test"],
        default="all",
        help="PTB-XL stratified split (train=folds 1-8, val=9, test=10).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching PTB-XL metadata CSV.",
    )
    parser.add_argument(
        "--unique-patients",
        action="store_true",
        help="Sample at most one record per patient.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for batch generation (default: 1).",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["opencv", "matplotlib"],
        default="opencv",
        help="Rendering backend (default: opencv).",
    )
    parser.add_argument(
        "--augment-profile",
        type=str,
        choices=["clean", "scan", "clinical"],
        default="scan",
        help="Artifact profile: clean, scan (default), or clinical (scan + perspective).",
    )
    parser.add_argument(
        "--save-clean",
        action="store_true",
        help="Also save pre-augmentation clean images to images/clean/.",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Skip exporting paired .npy signal files.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip exporting JSON annotation files.",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Skip exporting waveform segmentation masks.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Skip exporting YOLO-format label files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = GenerationConfig(
        count=args.count,
        diagnosis=args.diagnosis,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        split=args.split,
        unique_patients=args.unique_patients,
        workers=max(1, args.workers),
        export_signals=not args.no_signals,
        export_annotations=not args.no_annotations,
        export_masks=not args.no_masks,
        export_yolo=not args.no_yolo,
        save_clean=args.save_clean,
        render=RenderConfig(backend=args.renderer),
        augment=AugmentConfig(profile=args.augment_profile, seed=args.seed),
    )

    try:
        generate_dataset(config)
    except Exception as exc:
        print(f"Error during generation: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
