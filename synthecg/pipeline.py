from dataclasses import asdict
from pathlib import Path

import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

from synthecg.augment.paper import apply_paper_artifacts_to_arrays
from synthecg.config import AugmentConfig, GenerationConfig, RenderConfig
from synthecg.data.fetch import fetch_ptbxl_record
from synthecg.data.ptbxl import load_ptbxl_database, select_records
from synthecg.export.ground_truth import build_annotation, save_annotation, save_signal
from synthecg.export.manifest import ManifestWriter
from synthecg.export.masks import save_mask
from synthecg.export.yolo import write_classes_file, write_yolo_labels
from synthecg.render.opencv_renderer import render_ecg_opencv
from synthecg.render.plot import plot_ecg_layout


def _sample_id(diagnosis: str, ecg_id: int) -> str:
    return f"ecg_{diagnosis}_{ecg_id}"


def _config_from_dict(data: dict) -> GenerationConfig:
    render = RenderConfig(**data["render"])
    augment = AugmentConfig(**data["augment"])
    return GenerationConfig(
        render=render,
        augment=augment,
        **{k: v for k, v in data.items() if k not in ("render", "augment")},
    )


def _config_to_dict(config: GenerationConfig) -> dict:
    data = asdict(config)
    return data


def generate_sample(
    row,
    *,
    config: GenerationConfig,
    images_dir: Path,
    signals_dir: Path,
    annotations_dir: Path,
    masks_dir: Path | None,
    yolo_dir: Path | None,
    clean_dir: Path | None,
    sample_index: int,
) -> dict:
    """Generate one ECG image with optional ground-truth exports."""
    ecg_id = int(row.name) if hasattr(row, "name") else int(row["ecg_id"])
    record_name = row.filename_hr if hasattr(row, "filename_hr") else row["filename_hr"]
    patient_id = int(row.patient_id if hasattr(row, "patient_id") else row["patient_id"])
    scp_codes = row.scp_codes if hasattr(row, "scp_codes") else row["scp_codes"]
    strat_fold = int(row.strat_fold if hasattr(row, "strat_fold") else row["strat_fold"])

    sample_id = _sample_id(config.diagnosis, ecg_id)

    record = fetch_ptbxl_record(record_name, database=config.database)

    image_path = images_dir / f"{sample_id}.png"
    signal_path = signals_dir / f"{sample_id}.npy"
    annotation_path = annotations_dir / f"{sample_id}.json"
    mask_path = masks_dir / f"{sample_id}.png" if masks_dir else None
    yolo_path = yolo_dir / f"{sample_id}.txt" if yolo_dir else None
    clean_path = clean_dir / f"{sample_id}.png" if clean_dir else None

    render_config = config.render
    augment_config = AugmentConfig(
        profile=config.augment.profile,
        seed=(config.seed + sample_index) if config.seed is not None else None,
    )

    render_result = None
    if render_config.backend == "opencv":
        render_result = render_ecg_opencv(
            record,
            render_config,
            ecg_id=ecg_id,
            patient_id=patient_id,
            scp_codes=scp_codes,
        )
        img = render_result.image.copy()
        mask = render_result.mask.copy()

        if config.save_clean and clean_path is not None:
            cv2.imwrite(str(clean_path), img)

        img, mask, augmentations = apply_paper_artifacts_to_arrays(img, mask, augment_config)
        cv2.imwrite(str(image_path), img)

        if config.export_masks and mask_path is not None:
            save_mask(mask, mask_path)

        if config.export_yolo and yolo_path is not None and render_result is not None:
            write_yolo_labels(render_result.leads, yolo_path, img_w=render_result.width, img_h=render_result.height)
    else:
        plot_ecg_layout(record, str(image_path), render_config)
        augmentations = []
        if config.augment.profile != "clean":
            from synthecg.augment.paper import apply_paper_artifacts

            tmp = str(image_path.with_suffix(".clean.png"))
            Path(tmp).write_bytes(image_path.read_bytes())
            augmentations = apply_paper_artifacts(tmp, str(image_path), augment_config)
            Path(tmp).unlink(missing_ok=True)

    rel_image = str(Path("images") / image_path.name)
    rel_signal = ""
    rel_annotation = ""
    rel_mask = str(Path("masks") / mask_path.name) if mask_path and mask_path.exists() else None
    rel_yolo = str(Path("labels") / yolo_path.name) if yolo_path and yolo_path.exists() else None
    rel_clean = str(Path("images/clean") / clean_path.name) if clean_path and clean_path.exists() else None

    if config.export_signals:
        save_signal(record, signal_path)
        rel_signal = str(Path("signals") / signal_path.name)

    if config.export_annotations:
        annotation = build_annotation(
            ecg_id=ecg_id,
            patient_id=patient_id,
            scp_codes=scp_codes,
            strat_fold=strat_fold,
            record=record,
            render=render_config,
            augment=augment_config,
            augmentations=augmentations,
            image_path=rel_image,
            signal_path=rel_signal,
            mask_path=rel_mask,
            yolo_path=rel_yolo,
            clean_image_path=rel_clean,
            render_result=render_result,
        )
        save_annotation(annotation, annotation_path)
        rel_annotation = str(Path("annotations") / annotation_path.name)

    print(f"Saved {image_path}")

    return {
        "sample_id": sample_id,
        "ecg_id": ecg_id,
        "patient_id": patient_id,
        "scp_codes": scp_codes,
        "strat_fold": strat_fold,
        "image_path": rel_image,
        "signal_path": rel_signal or None,
        "annotation_path": rel_annotation or None,
        "mask_path": rel_mask,
        "yolo_path": rel_yolo,
        "clean_image_path": rel_clean,
        "augmentations": augmentations,
    }


def _worker_generate(payload: dict) -> dict:
    config = _config_from_dict(payload["config"])
    row = payload["row"]
    output_dir = Path(config.output_dir)

    result = generate_sample(
        row,
        config=config,
        images_dir=output_dir / "images",
        signals_dir=output_dir / "signals",
        annotations_dir=output_dir / "annotations",
        masks_dir=output_dir / "masks" if config.export_masks else None,
        yolo_dir=output_dir / "labels" if config.export_yolo else None,
        clean_dir=output_dir / "images" / "clean" if config.save_clean else None,
        sample_index=payload["sample_index"],
    )
    return {"diagnosis_query": config.diagnosis, **result}


def generate_dataset(config: GenerationConfig) -> Path:
    """Generate a dataset of synthetic ECG images with manifest and ground truth."""
    output_dir = Path(config.output_dir)
    images_dir = output_dir / "images"
    signals_dir = output_dir / "signals"
    annotations_dir = output_dir / "annotations"
    masks_dir = output_dir / "masks" if config.export_masks else None
    yolo_dir = output_dir / "labels" if config.export_yolo else None
    clean_dir = output_dir / "images" / "clean" if config.save_clean else None

    images_dir.mkdir(parents=True, exist_ok=True)
    if config.export_signals:
        signals_dir.mkdir(parents=True, exist_ok=True)
    if config.export_annotations:
        annotations_dir.mkdir(parents=True, exist_ok=True)
    if masks_dir:
        masks_dir.mkdir(parents=True, exist_ok=True)
    if yolo_dir:
        yolo_dir.mkdir(parents=True, exist_ok=True)
        write_classes_file(yolo_dir)
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Preparing to generate {config.count} ECGs "
        f"(type='{config.diagnosis}', split='{config.split}', seed={config.seed}, "
        f"workers={config.workers}, renderer={config.render.backend}) ..."
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
    rows = []
    for index, (_, row) in enumerate(selected.iterrows()):
        rows.append(
            {
                "row": {
                    "ecg_id": int(row.name),
                    "patient_id": int(row.patient_id),
                    "filename_hr": row.filename_hr,
                    "scp_codes": row.scp_codes,
                    "strat_fold": int(row.strat_fold),
                },
                "sample_index": index,
            }
        )

    config_dict = _config_to_dict(config)
    results: list[dict] = []

    if config.workers <= 1:
        for index, item in enumerate(rows):
            print(f"[{index + 1}/{len(rows)}] Processing ecg_id={item['row']['ecg_id']} ...")
            row_series = selected.iloc[index]
            result = generate_sample(
                row_series,
                config=config,
                images_dir=images_dir,
                signals_dir=signals_dir,
                annotations_dir=annotations_dir,
                masks_dir=masks_dir,
                yolo_dir=yolo_dir,
                clean_dir=clean_dir,
                sample_index=item["sample_index"],
            )
            results.append({"diagnosis_query": config.diagnosis, **result})
    else:
        payloads = [{"config": config_dict, **item} for item in rows]
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {executor.submit(_worker_generate, payload): payload for payload in payloads}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                payload = futures[future]
                print(f"[{completed}/{len(rows)}] Finished ecg_id={payload['row']['ecg_id']}")
                results.append(future.result())

    for result in sorted(results, key=lambda r: r["sample_id"]):
        manifest.add(**result)

    manifest_path = manifest.write()
    print(f"Generation complete. Manifest: {manifest_path}")
    return manifest_path
