from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

from synthecg.augment.paper import apply_paper_artifacts_to_arrays
from synthecg.config import AugmentConfig, GenerationConfig, RenderConfig, is_zheng_database
from synthecg.data.fetch import fetch_ptbxl_record
from synthecg.data.preprocess import preprocess_record
from synthecg.data.ptbxl import load_completed_ecg_ids, load_ptbxl_database, select_balanced_records, select_records
from synthecg.data.zheng_otva import (
    DATABASE_ID as ZHENG_DATABASE_ID,
    fetch_zheng_record,
    load_completed_hospital_ids,
    load_zheng_database,
    localization_from_row,
    select_zheng_records,
    zheng_metadata_row,
)
from synthecg.export.ground_truth import build_annotation, save_annotation, save_signal
from synthecg.export.manifest import ManifestWriter
from synthecg.export.masks import save_mask
from synthecg.export.yolo import write_classes_file, write_yolo_labels
from synthecg.localization.inference import infer_localization_from_record
from synthecg.render.opencv_renderer import render_ecg_opencv
from synthecg.render.plot import plot_ecg_layout


def _sample_id(diagnosis: str, ecg_id: int) -> str:
    return f"ecg_{diagnosis}_{ecg_id}"


def _diagnosis_query(row, config: GenerationConfig) -> str:
    if hasattr(row, "diagnosis_query") and getattr(row, "diagnosis_query", None):
        return str(row.diagnosis_query)
    if isinstance(row, dict) and row.get("diagnosis_query"):
        return str(row["diagnosis_query"])
    return config.diagnosis


def _config_from_dict(data: dict) -> GenerationConfig:
    render_data = data["render"]
    if "grid_color" in render_data and isinstance(render_data["grid_color"], list):
        render_data["grid_color"] = tuple(render_data["grid_color"])
    if "grid_minor_color" in render_data and isinstance(render_data["grid_minor_color"], list):
        render_data["grid_minor_color"] = tuple(render_data["grid_minor_color"])
    if "canvas_size" in render_data and isinstance(render_data["canvas_size"], list):
        render_data["canvas_size"] = tuple(render_data["canvas_size"])

    render = RenderConfig(**render_data)
    augment = AugmentConfig(**data["augment"])
    skip = {"render", "augment"}
    return GenerationConfig(
        render=render,
        augment=augment,
        **{k: v for k, v in data.items() if k not in skip},
    )


def _config_to_dict(config: GenerationConfig) -> dict:
    return asdict(config)


def _coerce_row(row):
    if isinstance(row, dict):
        obj = SimpleNamespace(**row)
        obj.name = row.get("ecg_id")
        return obj
    return row


def _resolve_localization(row, record, scp_codes: dict, config: GenerationConfig) -> dict | None:
    row = _coerce_row(row)
    if getattr(row, "localization", None):
        return row.localization
    if getattr(row, "site", None):
        return localization_from_row(row).to_dict()
    if not config.infer_localization:
        return None
    info = infer_localization_from_record(record, scp_codes)
    return info.to_dict() if info else None


def _fetch_record(row, config: GenerationConfig):
    row = _coerce_row(row)
    if is_zheng_database(config.database):
        hospital_id = int(row.name) if hasattr(row, "name") else int(row["ecg_id"])
        return fetch_zheng_record(hospital_id, cache_dir=config.cache_dir)
    record_name = row.filename_hr if hasattr(row, "filename_hr") else row["filename_hr"]
    return fetch_ptbxl_record(record_name, database=config.database)


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
    row = _coerce_row(row)
    ecg_id = int(row.name) if hasattr(row, "name") and row.name is not None else int(row.ecg_id)
    patient_id = int(getattr(row, "patient_id"))
    scp_codes = getattr(row, "scp_codes")
    strat_fold = int(getattr(row, "strat_fold"))
    query = _diagnosis_query(row, config)

    sample_id = _sample_id(query, ecg_id)

    record = _fetch_record(row, config)
    record = preprocess_record(
        record,
        bandpass=config.bandpass_filter,
        low_hz=config.bandpass_low_hz,
        high_hz=config.bandpass_high_hz,
    )
    localization = _resolve_localization(row, record, scp_codes, config)

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
            localization=localization,
            database=config.database if is_zheng_database(config.database) else config.database,
        )
        if config.bandpass_filter:
            annotation["preprocess"] = {
                "bandpass_filter": True,
                "low_hz": config.bandpass_low_hz,
                "high_hz": config.bandpass_high_hz,
            }
        save_annotation(annotation, annotation_path)
        rel_annotation = str(Path("annotations") / annotation_path.name)

    print(f"Saved {image_path}")

    result = {
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
        "diagnosis_query": query,
    }
    if localization is not None:
        result["localization"] = localization
    return result


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
    return result


def _load_completed_ids(config: GenerationConfig) -> set[int]:
    if config.resume:
        if is_zheng_database(config.database):
            return load_completed_hospital_ids(config.output_dir)
        return load_completed_ecg_ids(config.output_dir)
    return set()


def _load_database(config: GenerationConfig):
    if is_zheng_database(config.database):
        return load_zheng_database(config.cache_dir, download_ecg=False)
    return load_ptbxl_database(database=config.database, cache_dir=config.cache_dir)


def _select_dataset_records(df, config: GenerationConfig, exclude_ecg_ids: set[int]):
    if is_zheng_database(config.database):
        if config.balanced_sites:
            count_per_site = config.count_per_site or max(1, config.count // len(config.balanced_sites))
            return select_zheng_records(
                df,
                count=config.count,
                seed=config.seed,
                balanced_sites=config.balanced_sites,
                count_per_site=count_per_site,
                exclude_hospital_ids=exclude_ecg_ids,
            )
        site = None if config.diagnosis.lower() == "random" else config.diagnosis
        return select_zheng_records(
            df,
            count=config.count,
            seed=config.seed,
            site=site,
            exclude_hospital_ids=exclude_ecg_ids,
        )

    if config.balanced_codes:
        count_per_code = config.count_per_code or max(1, config.count // len(config.balanced_codes))
        return select_balanced_records(
            df,
            codes=config.balanced_codes,
            count_per_code=count_per_code,
            seed=config.seed,
            split=config.split,
            unique_patients=config.unique_patients,
            exclude_ecg_ids=exclude_ecg_ids,
        )

    return select_records(
        df,
        diagnosis=config.diagnosis,
        count=config.count,
        seed=config.seed,
        split=config.split,
        unique_patients=config.unique_patients,
        include_codes=config.include_codes,
        exclude_codes=config.exclude_codes,
        exclude_ecg_ids=exclude_ecg_ids,
    )


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

    exclude_ecg_ids = _load_completed_ids(config)
    if exclude_ecg_ids:
        print(f"Resume: skipping {len(exclude_ecg_ids)} already-generated ecg_ids.")

    if is_zheng_database(config.database):
        config = GenerationConfig(**{**asdict(config), "database": ZHENG_DATABASE_ID})

    mode = (
        "balanced_sites"
        if config.balanced_sites
        else ("balanced" if config.balanced_codes else config.diagnosis)
    )
    print(
        f"Preparing to generate records "
        f"(database='{config.database}', mode='{mode}', split='{config.split}', seed={config.seed}, "
        f"workers={config.workers}, renderer={config.render.backend}, "
        f"bandpass={config.bandpass_filter}) ..."
    )

    df = _load_database(config)
    selected = _select_dataset_records(df, config, exclude_ecg_ids)

    if selected.empty:
        print("No new records to generate.")
        manifest = ManifestWriter(output_dir, resume=config.resume)
        return manifest.write()

    rows = []
    for index, (_, row) in enumerate(selected.iterrows()):
        if is_zheng_database(config.database):
            row_dict = zheng_metadata_row(row)
        else:
            row_dict = {
                "ecg_id": int(row.name),
                "patient_id": int(row.patient_id),
                "filename_hr": row.filename_hr,
                "scp_codes": row.scp_codes,
                "strat_fold": int(row.strat_fold),
            }
            if "diagnosis_query" in row:
                row_dict["diagnosis_query"] = row.diagnosis_query
        rows.append({"row": row_dict, "sample_index": index})

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
            results.append(result)
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

    manifest = ManifestWriter(output_dir, resume=config.resume)
    for result in sorted(results, key=lambda r: r["sample_id"]):
        manifest.add(**result)

    manifest_path = manifest.write()
    print(f"Generation complete. Manifest: {manifest_path} ({len(manifest._rows)} total entries)")
    return manifest_path
