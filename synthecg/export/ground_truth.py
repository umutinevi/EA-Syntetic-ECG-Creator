import json
from pathlib import Path

import numpy as np

from synthecg.config import LEAD_NAMES, AugmentConfig, RenderConfig
from synthecg.render.types import LeadRegion, RenderResult


def save_signal(record, output_path: str | Path) -> Path:
    """Save the 12-lead signal as a NumPy array with shape (12, samples)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = record.p_signal.T.astype(np.float32)
    np.save(output_path, signal)
    return output_path


def _lead_to_dict(lead: LeadRegion, render: RenderResult) -> dict:
    return {
        "name": lead.name,
        "lead_idx": lead.lead_idx,
        "bbox": list(lead.bbox),
        "label_bbox": list(lead.label_bbox),
        "plot_bbox": list(lead.plot_bbox),
        "waveform_bbox": list(lead.waveform_bbox),
        "t_start": lead.t_start,
        "t_end": lead.t_end,
        "baseline_y": lead.baseline_y,
    }


def build_annotation(
    *,
    ecg_id: int,
    patient_id: int,
    scp_codes: dict,
    strat_fold: int,
    record,
    render: RenderConfig,
    augment: AugmentConfig,
    augmentations: list[str],
    image_path: str,
    signal_path: str,
    mask_path: str | None = None,
    yolo_path: str | None = None,
    clean_image_path: str | None = None,
    render_result: RenderResult | None = None,
) -> dict:
    """Build a JSON-serializable annotation document for one sample."""
    annotation = {
        "ecg_id": ecg_id,
        "patient_id": int(patient_id),
        "scp_codes": scp_codes,
        "strat_fold": int(strat_fold),
        "render": {
            "layout": render.layout,
            "backend": render.backend,
            "speed_mm_s": render.speed_mm_s,
            "gain_mm_mv": render.gain_mm_mv,
            "dpi": render.dpi,
            "show_grid": render.show_grid,
            "canvas_size": list(render.canvas_size),
        },
        "signal": {
            "fs": float(record.fs),
            "n_samples": int(record.sig_len),
            "lead_names": LEAD_NAMES,
            "units": "mV",
        },
        "augment": {
            "profile": augment.profile,
            "applied": augmentations,
        },
        "paths": {
            "image": image_path,
            "signal": signal_path,
            "mask": mask_path,
            "yolo": yolo_path,
            "clean_image": clean_image_path,
        },
    }

    if render_result is not None:
        annotation["render"].update(
            {
                "px_per_mm": render_result.px_per_mm,
                "px_per_second": render_result.px_per_second,
                "px_per_mv": render_result.px_per_mv,
                "width": render_result.width,
                "height": render_result.height,
            }
        )
        annotation["leads"] = [_lead_to_dict(lead, render_result) for lead in render_result.leads]

    return annotation


def save_annotation(annotation: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle, indent=2)
    return output_path
