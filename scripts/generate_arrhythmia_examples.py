#!/usr/bin/env python3
"""Generate curated arrhythmia example ECG images for the examples/ folder."""

from __future__ import annotations

import json
from pathlib import Path

import ast
import cv2
import pandas as pd

from synthecg.config import RenderConfig
from synthecg.data.fetch import fetch_ptbxl_record
from synthecg.data.preprocess import preprocess_record
from synthecg.render.opencv_renderer import render_ecg_opencv
from synthecg.render.overlay import add_clinical_overlay

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Curated examples: closest PTB-XL match + clinical overlay for anatomy not in SCP codes.
ARRHYTHMIA_EXAMPLES = [
    {
        "filename": "afib_atrial_fibrillation.png",
        "ecg_id": 351,
        "scp_filter": "AFIB",
        "primary_label": "Atrial Fibrillation (AFIB)",
        "secondary_label": "Irregularly irregular rhythm without discrete P waves",
    },
    {
        "filename": "pvc_left_coronary_cusp.png",
        "ecg_id": 255,
        "scp_filter": "PVC",
        "primary_label": "Premature Ventricular Contraction (PVC)",
        "secondary_label": "Anatomic localization: Left site, Left Coronary Cusp region",
    },
    {
        "filename": "avnrt_nodal_reentrant_tachycardia.png",
        "ecg_id": 1299,
        "scp_filter": "PSVT",
        "primary_label": "AVNRT — Atrioventricular Nodal Reentrant Tachycardia",
        "secondary_label": "PTB-XL proxy: PSVT (paroxysmal supraventricular tachycardia)",
    },
    {
        "filename": "wpw_right_free_wall.png",
        "ecg_id": 2145,
        "scp_filter": "WPW",
        "primary_label": "Wolff-Parkinson-White (WPW) Syndrome",
        "secondary_label": "Accessory pathway localization: Right Free Wall",
    },
]


def _load_metadata() -> pd.DataFrame:
    url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
    cache = Path.home() / ".cache" / "synthecg" / "ptbxl_database.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col="ecg_id")
    else:
        df = pd.read_csv(url, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def generate_example(spec: dict, df: pd.DataFrame, output_dir: Path) -> dict:
    ecg_id = spec["ecg_id"]
    row = df.loc[ecg_id]
    scp_codes = row.scp_codes

    if spec["scp_filter"] not in scp_codes:
        raise ValueError(f"ecg_id {ecg_id} missing expected code {spec['scp_filter']}: {scp_codes}")

    record = fetch_ptbxl_record(row.filename_hr)
    record = preprocess_record(record, bandpass=True)

    render_config = RenderConfig(show_grid=True)
    result = render_ecg_opencv(
        record,
        render_config,
        ecg_id=int(ecg_id),
        patient_id=int(row.patient_id),
        scp_codes=scp_codes,
    )

    scp_summary = ", ".join(f"{k}({v})" for k, v in scp_codes.items() if v > 0)
    img = add_clinical_overlay(
        result.image,
        primary_label=spec["primary_label"],
        secondary_label=spec["secondary_label"],
        scp_codes=scp_summary,
    )

    out_path = output_dir / spec["filename"]
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    meta = {
        "filename": spec["filename"],
        "ecg_id": int(ecg_id),
        "patient_id": int(row.patient_id),
        "scp_codes": scp_codes,
        "scp_filter": spec["scp_filter"],
        "primary_label": spec["primary_label"],
        "secondary_label": spec["secondary_label"],
        "report": str(row.report) if isinstance(row.report, str) else "",
        "filename_hr": row.filename_hr,
    }
    return meta


def main() -> None:
    output_dir = EXAMPLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_metadata()
    manifest: list[dict] = []

    print(f"Generating {len(ARRHYTHMIA_EXAMPLES)} arrhythmia examples -> {output_dir}")
    for spec in ARRHYTHMIA_EXAMPLES:
        print(f"  {spec['filename']} (ecg_id={spec['ecg_id']}) ...")
        manifest.append(generate_example(spec, df, output_dir))

    index_path = output_dir / "arrhythmia_index.json"
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Index: {index_path}")


if __name__ == "__main__":
    main()
