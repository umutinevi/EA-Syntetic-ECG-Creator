import cv2
import numpy as np

from synthecg.config import RenderConfig


def draw_header(
    canvas: np.ndarray,
    *,
    ecg_id: int,
    patient_id: int,
    config: RenderConfig,
) -> None:
    title = f"SynthECG  |  PTB-XL #{ecg_id}  |  Patient {patient_id}"
    settings = (
        f"{config.speed_mm_s} mm/s   {config.gain_mm_mv} mm/mV   "
        f"Filter: 0.05-150 Hz   50 Hz"
    )
    cv2.putText(canvas, title, (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, settings, (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def draw_footer(canvas: np.ndarray, *, scp_summary: str, layout_label: str, width: int, height: int) -> None:
    cv2.putText(
        canvas,
        scp_summary,
        (80, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"12-Lead ECG  |  {layout_label}",
        (width - 520, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def plot_waveform(
    canvas: np.ndarray,
    mask: np.ndarray,
    signal: np.ndarray,
    x0: int,
    baseline_y: int,
    plot_width: int,
    *,
    px_per_mv: float,
    color: tuple[int, int, int] = (0, 0, 40),
    thickness: int = 2,
) -> None:
    n = len(signal)
    if n < 2:
        return

    xs = np.linspace(0, plot_width - 1, n).astype(np.int32)
    ys = baseline_y - np.round(signal * px_per_mv).astype(np.int32)
    points = np.stack([x0 + xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, color, thickness, lineType=cv2.LINE_AA)
    cv2.polylines(mask, [points], False, 255, thickness, lineType=cv2.LINE_AA)
