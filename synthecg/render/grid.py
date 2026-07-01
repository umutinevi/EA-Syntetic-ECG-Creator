import cv2
import numpy as np


def px_per_mm(dpi: int) -> float:
    return dpi / 25.4


def draw_ecg_grid(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    px_per_mm_val: float,
    major_color: tuple[int, int, int],
    minor_color: tuple[int, int, int],
) -> None:
    """Draw 1 mm minor and 5 mm major ECG grid lines on a region."""
    mm_x = max(1, int(round(px_per_mm_val)))
    mm_y = mm_x

    for offset in range(0, width + 1, mm_x):
        color = major_color if offset % (5 * mm_x) == 0 else minor_color
        cv2.line(canvas, (x + offset, y), (x + offset, y + height), color, 1)

    for offset in range(0, height + 1, mm_y):
        color = major_color if offset % (5 * mm_y) == 0 else minor_color
        cv2.line(canvas, (x, y + offset), (x + width, y + offset), color, 1)


def draw_calibration_pulse(
    canvas: np.ndarray,
    mask: np.ndarray,
    x0: int,
    baseline_y: int,
    *,
    px_per_second: float,
    px_per_mv: float,
    color: tuple[int, int, int] = (0, 0, 40),
    thickness: int = 2,
) -> None:
    """Draw a standard 0.2 s, 1 mV calibration pulse."""
    step = int(round(0.1 * px_per_second))
    height = int(round(1.0 * px_per_mv))
    points = [
        (x0, baseline_y),
        (x0 + step, baseline_y),
        (x0 + step, baseline_y - height),
        (x0 + 2 * step, baseline_y - height),
        (x0 + 2 * step, baseline_y),
    ]
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)
    cv2.polylines(mask, [pts], False, 255, thickness, lineType=cv2.LINE_AA)
