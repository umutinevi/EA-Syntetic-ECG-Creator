import json
from pathlib import Path

import numpy as np

from synthecg.config import LEAD_NAMES, AugmentConfig, RenderConfig


def save_signal(record, output_path: str | Path) -> Path:
    """Save the 12-lead signal as a NumPy array with shape (12, samples)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = record.p_signal.T.astype(np.float32)
    np.save(output_path, signal)
    return output_path


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
) -> dict:
    """Build a JSON-serializable annotation document for one sample."""
    return {
        "ecg_id": ecg_id,
        "patient_id": int(patient_id),
        "scp_codes": scp_codes,
        "strat_fold": int(strat_fold),
        "render": {
            "layout": render.layout,
            "speed_mm_s": render.speed_mm_s,
            "gain_mm_mv": render.gain_mm_mv,
            "dpi": render.dpi,
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
        },
    }


def save_annotation(annotation: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle, indent=2)
    return output_path
