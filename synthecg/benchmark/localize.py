"""Benchmark literature localization algorithms against Zheng EP labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthecg.data.zheng_otva import fetch_zheng_record, load_zheng_database, localization_from_row
from synthecg.localization.inference import infer_localization_from_signal
from synthecg.localization.pvc_otva import infer_pvc_localization


def _region_accuracy(truth: dict, predicted: dict | None) -> bool:
    if predicted is None:
        return False
    return truth.get("region") == predicted.get("region")


def _site_accuracy(truth: dict, predicted: dict | None) -> bool:
    if predicted is None:
        return False
    return truth.get("site") == predicted.get("site")


def run_localization_benchmark(
    *,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    algorithm: str = "pvc_otva",
) -> dict:
    """Evaluate localization algorithms on Zheng OT-VA EP-validated records."""
    df = load_zheng_database(cache_dir, download_ecg=False)
    if limit is not None:
        df = df.sample(n=min(limit, len(df)), random_state=0).sort_index()

    region_correct = 0
    site_correct = 0
    evaluated = 0
    skipped = 0
    results: list[dict] = []

    for hospital_id, row in df.iterrows():
        truth = localization_from_row(row).to_dict()
        try:
            record = fetch_zheng_record(int(hospital_id), cache_dir=cache_dir)
        except FileNotFoundError:
            skipped += 1
            continue

        signal = record.p_signal.T if record.p_signal.shape[1] == 12 else record.p_signal
        if algorithm == "pvc_otva":
            predicted_info = infer_pvc_localization(signal, float(record.fs))
        else:
            predicted_info = infer_localization_from_signal(signal, float(record.fs), {"PVC": 100.0})

        predicted = predicted_info.to_dict() if predicted_info else None
        region_ok = _region_accuracy(truth, predicted)
        site_ok = _site_accuracy(truth, predicted)
        region_correct += int(region_ok)
        site_correct += int(site_ok)
        evaluated += 1
        results.append(
            {
                "hospital_id": int(hospital_id),
                "truth": truth,
                "predicted": predicted,
                "region_correct": region_ok,
                "site_correct": site_ok,
            }
        )

    summary = {
        "algorithm": algorithm,
        "evaluated": evaluated,
        "skipped_missing_ecg": skipped,
        "region_accuracy": region_correct / evaluated if evaluated else 0.0,
        "site_accuracy": site_correct / evaluated if evaluated else 0.0,
        "results": results,
    }
    return summary


def write_localization_report(output_path: str | Path, summary: dict) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path
