#!/usr/bin/env python3
"""Generate curated arrhythmia example ECG images with provenance-aware localization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

from synthecg.config import RenderConfig
from synthecg.data.fetch import fetch_ptbxl_record
from synthecg.data.preprocess import preprocess_record
from synthecg.data.ptbxl import load_ptbxl_database
from synthecg.data.zheng_otva import fetch_zheng_record, load_zheng_database, localization_from_row
from synthecg.localization.inference import infer_localization_from_record
from synthecg.localization.taxonomy import site_label
from synthecg.render.opencv_renderer import render_ecg_opencv
from synthecg.render.overlay import add_clinical_overlay

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

# Fixed PTB-XL WPW teaching case — localization from literature algorithm only (unverified).
WPW_EXAMPLE_ECG_ID = 4825


def _overlay_secondary(localization: dict | None, fallback: str) -> str:
    if not localization:
        return fallback
    source = localization.get("source", "unknown")
    verified = localization.get("verified", False)
    region = localization.get("region")
    site = localization.get("site_label") or site_label(localization.get("site") or "")
    conf = localization.get("confidence", 0)
    status = "EP-verified" if verified else "unverified algorithm"

    if localization.get("level") == "mechanism":
        return f"Mechanism: {site} ({status}, source={source})"
    if localization.get("level") == "none":
        return fallback
    if region:
        return f"Localization: {region} / {site} ({status}, conf={conf:.2f})"
    return fallback


def _render_example(
    record,
    *,
    ecg_id: int,
    patient_id: int,
    scp_codes: dict,
    primary_label: str,
    secondary_label: str,
    output_path: Path,
) -> None:
    render_config = RenderConfig(show_grid=True)
    result = render_ecg_opencv(
        record,
        render_config,
        ecg_id=ecg_id,
        patient_id=patient_id,
        scp_codes=scp_codes,
    )
    scp_summary = ", ".join(f"{k}({v})" for k, v in scp_codes.items() if v > 0)
    img = add_clinical_overlay(
        result.image,
        primary_label=primary_label,
        secondary_label=secondary_label,
        scp_codes=scp_summary,
    )
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def build_examples() -> list[dict]:
    ptbxl = load_ptbxl_database()
    manifest: list[dict] = []

    # AFIB — no anatomic localization
    afib_id = 351
    afib_row = ptbxl.loc[afib_id]
    afib_record = preprocess_record(fetch_ptbxl_record(afib_row.filename_hr), bandpass=True)
    afib_loc = infer_localization_from_record(afib_record, afib_row.scp_codes)
    afib_loc_dict = afib_loc.to_dict() if afib_loc else None
    afib_secondary = _overlay_secondary(
        afib_loc_dict,
        "Irregularly irregular rhythm without discrete P waves",
    )
    _render_example(
        afib_record,
        ecg_id=afib_id,
        patient_id=int(afib_row.patient_id),
        scp_codes=afib_row.scp_codes,
        primary_label="Atrial Fibrillation (AFIB)",
        secondary_label=afib_secondary,
        output_path=EXAMPLES_DIR / "afib_atrial_fibrillation.png",
    )
    manifest.append(
        {
            "filename": "afib_atrial_fibrillation.png",
            "database": "ptb-xl/1.0.3",
            "ecg_id": afib_id,
            "patient_id": int(afib_row.patient_id),
            "scp_codes": afib_row.scp_codes,
            "scp_filter": "AFIB",
            "primary_label": "Atrial Fibrillation (AFIB)",
            "secondary_label": afib_secondary,
            "report": str(afib_row.report),
            "filename_hr": afib_row.filename_hr,
            "localization": afib_loc_dict,
        }
    )

    # PVC LCC — Zheng EP-validated record (verified ground truth)
    zheng_df = load_zheng_database(download_ecg=False)
    lcc_rows = zheng_df[zheng_df.site == "LCC"]
    if lcc_rows.empty:
        raise RuntimeError("No LCC records found in Zheng diagnosis table.")
    lcc_row = lcc_rows.iloc[0]
    hospital_id = int(lcc_row.name)
    lcc_record = preprocess_record(fetch_zheng_record(hospital_id), bandpass=True)
    lcc_loc = localization_from_row(lcc_row).to_dict()
    lcc_secondary = _overlay_secondary(lcc_loc, "Anatomic localization: Left Coronary Cusp")
    _render_example(
        lcc_record,
        ecg_id=hospital_id,
        patient_id=hospital_id,
        scp_codes={"PVC": 100.0},
        primary_label="Premature Ventricular Contraction (PVC)",
        secondary_label=lcc_secondary,
        output_path=EXAMPLES_DIR / "pvc_left_coronary_cusp.png",
    )
    manifest.append(
        {
            "filename": "pvc_left_coronary_cusp.png",
            "database": "zheng-otva",
            "ecg_id": hospital_id,
            "patient_id": hospital_id,
            "scp_codes": {"PVC": 100.0},
            "scp_filter": "PVC",
            "primary_label": "Premature Ventricular Contraction (PVC)",
            "secondary_label": lcc_secondary,
            "report": "Zheng OT-VA EP-validated LCC origin (ablation mapping confirmed).",
            "filename_hr": str(hospital_id),
            "localization": lcc_loc,
        }
    )

    # AVNRT proxy via PSVT
    avnrt_id = 1299
    avnrt_row = ptbxl.loc[avnrt_id]
    avnrt_record = preprocess_record(fetch_ptbxl_record(avnrt_row.filename_hr), bandpass=True)
    avnrt_loc = infer_localization_from_record(avnrt_record, {"PSVT": 100.0, **avnrt_row.scp_codes})
    avnrt_loc_dict = avnrt_loc.to_dict() if avnrt_loc else None
    avnrt_secondary = _overlay_secondary(
        avnrt_loc_dict,
        "PTB-XL proxy: PSVT (paroxysmal supraventricular tachycardia)",
    )
    _render_example(
        avnrt_record,
        ecg_id=avnrt_id,
        patient_id=int(avnrt_row.patient_id),
        scp_codes=avnrt_row.scp_codes,
        primary_label="AVNRT — Atrioventricular Nodal Reentrant Tachycardia",
        secondary_label=avnrt_secondary,
        output_path=EXAMPLES_DIR / "avnrt_nodal_reentrant_tachycardia.png",
    )
    manifest.append(
        {
            "filename": "avnrt_nodal_reentrant_tachycardia.png",
            "database": "ptb-xl/1.0.3",
            "ecg_id": avnrt_id,
            "patient_id": int(avnrt_row.patient_id),
            "scp_codes": avnrt_row.scp_codes,
            "scp_filter": "PSVT",
            "primary_label": "AVNRT — Atrioventricular Nodal Reentrant Tachycardia",
            "secondary_label": avnrt_secondary,
            "report": str(avnrt_row.report),
            "filename_hr": avnrt_row.filename_hr,
            "localization": avnrt_loc_dict,
        }
    )

    # WPW — literature algorithm only; do NOT claim EP-verified pathway location
    wpw_id = WPW_EXAMPLE_ECG_ID
    wpw_row = ptbxl.loc[wpw_id]
    wpw_record = preprocess_record(fetch_ptbxl_record(wpw_row.filename_hr), bandpass=True)
    wpw_loc_obj = infer_localization_from_record(wpw_record, wpw_row.scp_codes)
    wpw_loc = wpw_loc_obj.to_dict() if wpw_loc_obj else None

    predicted_site = wpw_loc.get("site_label", "unknown") if wpw_loc else "unknown"
    wpw_secondary = _overlay_secondary(
        wpw_loc,
        "WPW — accessory pathway localization requires EP confirmation",
    )
    wpw_primary = "Wolff-Parkinson-White (WPW) — Literature Algorithm (Unverified)"

    out_name = "wpw_accessory_pathway.png"
    legacy_name = EXAMPLES_DIR / "wpw_right_free_wall.png"
    if legacy_name.exists():
        legacy_name.unlink()

    _render_example(
        wpw_record,
        ecg_id=wpw_id,
        patient_id=int(wpw_row.patient_id),
        scp_codes=wpw_row.scp_codes,
        primary_label=wpw_primary,
        secondary_label=wpw_secondary,
        output_path=EXAMPLES_DIR / out_name,
    )
    manifest.append(
        {
            "filename": out_name,
            "database": "ptb-xl/1.0.3",
            "ecg_id": wpw_id,
            "patient_id": int(wpw_row.patient_id),
            "scp_codes": wpw_row.scp_codes,
            "scp_filter": "WPW",
            "primary_label": wpw_primary,
            "secondary_label": wpw_secondary,
            "report": str(wpw_row.report),
            "filename_hr": wpw_row.filename_hr,
            "localization": wpw_loc,
            "clinical_note": (
                f"Literature algorithm predicts {predicted_site}. "
                "Independent EP review may disagree (e.g. CS OS vs free wall). "
                "Not ablation ground truth."
            ),
        }
    )

    return manifest


def main() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating arrhythmia examples -> {EXAMPLES_DIR}")
    manifest = build_examples()
    index_path = EXAMPLES_DIR / "arrhythmia_index.json"
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Index: {index_path}")


if __name__ == "__main__":
    main()
