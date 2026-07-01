import os
import tempfile
from pathlib import Path

from synthecg.augment.paper import apply_paper_artifacts
from synthecg.config import AugmentConfig, GenerationConfig, RenderConfig
from synthecg.data.fetch import fetch_ptbxl_record
from synthecg.data.ptbxl import load_ptbxl_database, select_records
from synthecg.export.ground_truth import build_annotation, save_annotation, save_signal
from synthecg.export.manifest import ManifestWriter
from synthecg.render.plot import plot_ecg_layout


def _sample_id(diagnosis: str, ecg_id: int) -> str:
    return f"ecg_{diagnosis}_{ecg_id}"


def generate_sample(
    row,
    *,
    config: GenerationConfig,
    images_dir: Path,
    signals_dir: Path,
    annotations_dir: Path,
    sample_index: int,
) -> dict:
    """Generate one ECG image with optional ground-truth exports."""
    ecg_id = int(row.name)
    record_name = row.filename_hr
    sample_id = _sample_id(config.diagnosis, ecg_id)

    record = fetch_ptbxl_record(record_name, database=config.database)

    image_path = images_dir / f"{sample_id}.png"
    signal_path = signals_dir / f"{sample_id}.npy"
    annotation_path = annotations_dir / f"{sample_id}.json"

    render_config = config.render
    augment_config = AugmentConfig(
        profile=config.augment.profile,
        seed=(config.seed + sample_index) if config.seed is not None else None,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        clean_path = os.path.join(tmp_dir, "clean.png")
        plot_ecg_layout(record, clean_path, render_config)
        augmentations = apply_paper_artifacts(clean_path, str(image_path), augment_config)

    rel_image = str(Path("images") / image_path.name)
    rel_signal = ""
    rel_annotation = ""

    if config.export_signals:
        save_signal(record, signal_path)
        rel_signal = str(Path("signals") / signal_path.name)

    if config.export_annotations:
        annotation = build_annotation(
            ecg_id=ecg_id,
            patient_id=int(row.patient_id),
            scp_codes=row.scp_codes,
            strat_fold=int(row.strat_fold),
            record=record,
            render=render_config,
            augment=augment_config,
            augmentations=augmentations,
            image_path=rel_image,
            signal_path=rel_signal,
        )
        save_annotation(annotation, annotation_path)
        rel_annotation = str(Path("annotations") / annotation_path.name)

    print(f"Saved {image_path}")

    return {
        "sample_id": sample_id,
        "ecg_id": ecg_id,
        "patient_id": int(row.patient_id),
        "scp_codes": row.scp_codes,
        "strat_fold": int(row.strat_fold),
        "image_path": rel_image,
        "signal_path": rel_signal or None,
        "annotation_path": rel_annotation or None,
        "augmentations": augmentations,
    }


def generate_dataset(config: GenerationConfig) -> Path:
    """Generate a dataset of synthetic ECG images with manifest and ground truth."""
    output_dir = Path(config.output_dir)
    images_dir = output_dir / "images"
    signals_dir = output_dir / "signals"
    annotations_dir = output_dir / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    if config.export_signals:
        signals_dir.mkdir(parents=True, exist_ok=True)
    if config.export_annotations:
        annotations_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Preparing to generate {config.count} ECGs "
        f"(type='{config.diagnosis}', split='{config.split}', seed={config.seed}) ..."
    )

    df = load_ptbxl_database(database=config.database, cache_dir=config.cache_dir)
    selected = select_records(
        df,
        diagnosis=config.diagnosis,
        count=config.count,
        seed=config.seed,
        split=config.split,
        unique_patients=config.unique_patients,
    )

    manifest = ManifestWriter(output_dir)

    for index, (_, row) in enumerate(selected.iterrows()):
        print(f"[{index + 1}/{len(selected)}] Processing ecg_id={row.name} ...")
        result = generate_sample(
            row,
            config=config,
            images_dir=images_dir,
            signals_dir=signals_dir,
            annotations_dir=annotations_dir,
            sample_index=index,
        )
        manifest.add(
            diagnosis_query=config.diagnosis,
            **result,
        )

    manifest_path = manifest.write()
    print(f"Generation complete. Manifest: {manifest_path}")
    return manifest_path
