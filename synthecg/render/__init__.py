from pathlib import Path

import cv2
import numpy as np

from synthecg.config import RenderConfig
from synthecg.render.opencv_renderer import render_ecg_opencv, save_render_result
from synthecg.render.plot import plot_ecg_layout
from synthecg.render.types import RenderResult


def render_ecg(
    record,
    output_path: str,
    config: RenderConfig | None = None,
    *,
    ecg_id: int = 0,
    patient_id: int = 0,
    scp_codes: dict | None = None,
) -> RenderResult | None:
    """Render an ECG to disk using the configured backend."""
    config = config or RenderConfig()

    if config.backend == "opencv":
        result = render_ecg_opencv(
            record,
            config,
            ecg_id=ecg_id,
            patient_id=patient_id,
            scp_codes=scp_codes,
        )
        save_render_result(result, output_path)
        return result

    plot_ecg_layout(record, output_path, config)
    return None
