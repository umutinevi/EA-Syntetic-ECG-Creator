import cv2
import numpy as np

from synthecg.config import LEAD_NAMES, RenderConfig
from synthecg.render.grid import draw_calibration_pulse, draw_ecg_grid, px_per_mm
from synthecg.render.types import LeadRegion, RenderResult


def _draw_header(
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
    cv2.putText(canvas, title, (80, 55), cv2.FONT_HERSHEY_SIMPLEX,  1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, settings, (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def _draw_footer(canvas: np.ndarray, *, scp_summary: str, width: int, height: int) -> None:
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
        "12-Lead ECG  |  3x4 + Rhythm II",
        (width - 520, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def _plot_waveform(
    canvas: np.ndarray,
    mask: np.ndarray,
    signal: np.ndarray,
    fs: float,
    x0: int,
    baseline_y: int,
    plot_width: int,
    *,
    px_per_second: float,
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


def render_ecg_opencv(
    record,
    config: RenderConfig | None = None,
    *,
    ecg_id: int = 0,
    patient_id: int = 0,
    scp_codes: dict | None = None,
) -> RenderResult:
    """Render a 3x4 + rhythm ECG using OpenCV with masks and layout metadata."""
    config = config or RenderConfig()
    scp_codes = scp_codes or {}

    width, height = config.canvas_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    ppm = px_per_mm(config.dpi)
    px_sec = config.speed_mm_s * ppm
    px_mv = config.gain_mm_mv * ppm

    margin_x, margin_top, margin_bottom = 80, 120, 80
    content_x = margin_x
    content_y = margin_top
    content_w = width - 2 * margin_x
    content_h = height - margin_top - margin_bottom

    col_duration = 2.5
    col_width = int(round(col_duration * px_sec))
    row_height = content_h // 4

    _draw_header(canvas, ecg_id=ecg_id, patient_id=patient_id, config=config)
    scp_summary = "SCP: " + ", ".join(scp_codes.keys()) if scp_codes else "SCP: N/A"
    _draw_footer(canvas, scp_summary=scp_summary, width=width, height=height)

    signals = record.p_signal.T
    fs = record.fs
    leads: list[LeadRegion] = []

    for row in range(3):
        for col in range(4):
            lead_idx = col * 3 + row
            if lead_idx >= 12:
                continue

            region_x = content_x + col * col_width
            region_y = content_y + row * row_height
            plot_x = region_x + int(round(0.25 * px_sec))
            plot_y = region_y + 20
            plot_w = col_width - int(round(0.35 * px_sec))
            plot_h = row_height - 40
            baseline_y = plot_y + plot_h // 2

            if config.show_grid:
                draw_ecg_grid(
                    canvas,
                    plot_x,
                    plot_y,
                    plot_w,
                    plot_h,
                    px_per_mm_val=ppm,
                    major_color=config.grid_color,
                    minor_color=config.grid_minor_color,
                )

            t_start = col * col_duration
            t_end = t_start + col_duration
            idx_start = int(t_start * fs)
            idx_end = int(t_end * fs)
            segment = signals[lead_idx, idx_start:idx_end]

            draw_calibration_pulse(
                canvas,
                mask,
                plot_x,
                baseline_y,
                px_per_second=px_sec,
                px_per_mv=px_mv,
            )
            cal_offset = int(round(0.22 * px_sec))
            waveform_x = plot_x + cal_offset
            waveform_w = plot_w - cal_offset
            waveform_bbox = (waveform_x, plot_y, waveform_w, plot_h)

            _plot_waveform(
                canvas,
                mask,
                segment,
                fs,
                waveform_x,
                baseline_y,
                waveform_w,
                px_per_second=px_sec,
                px_per_mv=px_mv,
            )

            label = LEAD_NAMES[lead_idx]
            label_x, label_y = plot_x + 8, plot_y + 28
            cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            region_bbox = (region_x, region_y, col_width, row_height)
            label_bbox = (label_x - 4, label_y - text_h - 4, text_w + 8, text_h + 8)
            plot_bbox = (plot_x, plot_y, plot_w, plot_h)

            leads.append(
                LeadRegion(
                    name=label,
                    lead_idx=lead_idx,
                    bbox=region_bbox,
                    label_bbox=label_bbox,
                    plot_bbox=plot_bbox,
                    waveform_bbox=waveform_bbox,
                    t_start=t_start,
                    t_end=t_end,
                    baseline_y=baseline_y,
                )
            )

    rhythm_row = 3
    rhythm_x = content_x
    rhythm_y = content_y + rhythm_row * row_height
    rhythm_w = 4 * col_width
    rhythm_h = row_height
    plot_x = rhythm_x + int(round(0.15 * px_sec))
    plot_y = rhythm_y + 20
    plot_w = rhythm_w - int(round(0.2 * px_sec))
    plot_h = rhythm_h - 40
    baseline_y = plot_y + plot_h // 2

    if config.show_grid:
        draw_ecg_grid(
            canvas,
            plot_x,
            plot_y,
            plot_w,
            plot_h,
            px_per_mm_val=ppm,
            major_color=config.grid_color,
            minor_color=config.grid_minor_color,
        )

    rhythm_signal = signals[1, : int(10 * fs)]
    cal_offset = int(round(0.22 * px_sec))
    waveform_x = plot_x + cal_offset
    waveform_w = plot_w - cal_offset
    waveform_bbox = (waveform_x, plot_y, waveform_w, plot_h)

    draw_calibration_pulse(canvas, mask, plot_x, baseline_y, px_per_second=px_sec, px_per_mv=px_mv)
    _plot_waveform(
        canvas,
        mask,
        rhythm_signal,
        fs,
        waveform_x,
        baseline_y,
        waveform_w,
        px_per_second=px_sec,
        px_per_mv=px_mv,
    )

    label = "II"
    label_x, label_y = plot_x + 8, plot_y + 28
    cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    leads.append(
        LeadRegion(
            name="II",
            lead_idx=1,
            bbox=(rhythm_x, rhythm_y, rhythm_w, rhythm_h),
            label_bbox=(label_x - 4, label_y - text_h - 4, text_w + 8, text_h + 8),
            plot_bbox=(plot_x, plot_y, plot_w, plot_h),
            waveform_bbox=waveform_bbox,
            t_start=0.0,
            t_end=10.0,
            baseline_y=baseline_y,
        )
    )

    return RenderResult(
        image=canvas,
        mask=mask,
        width=width,
        height=height,
        leads=leads,
        px_per_mm=ppm,
        px_per_second=px_sec,
        px_per_mv=px_mv,
    )


def save_render_result(result: RenderResult, output_path: str) -> None:
    cv2.imwrite(output_path, result.image)
