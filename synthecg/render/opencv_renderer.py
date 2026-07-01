import cv2

from synthecg.config import RenderConfig
from synthecg.render.layouts import render_layout
from synthecg.render.types import RenderResult


def render_ecg_opencv(
    record,
    config: RenderConfig | None = None,
    *,
    ecg_id: int = 0,
    patient_id: int = 0,
    scp_codes: dict | None = None,
) -> RenderResult:
    """Render an ECG using OpenCV with the configured layout."""
    config = config or RenderConfig()
    return render_layout(
        record,
        config,
        ecg_id=ecg_id,
        patient_id=patient_id,
        scp_codes=scp_codes or {},
    )


def save_render_result(result: RenderResult, output_path: str) -> None:
    cv2.imwrite(output_path, result.image)
