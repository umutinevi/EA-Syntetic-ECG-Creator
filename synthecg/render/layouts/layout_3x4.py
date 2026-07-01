import cv2
import numpy as np

from synthecg.config import LEAD_NAMES, RenderConfig
from synthecg.render.grid import draw_calibration_pulse, draw_ecg_grid, px_per_mm
from synthecg.render.layouts.common import draw_footer, draw_header, plot_waveform
from synthecg.render.types import LeadRegion, RenderResult


def render_layout_3x4(
    record,
    config: RenderConfig,
    *,
    ecg_id: int,
    patient_id: int,
    scp_codes: dict,
) -> RenderResult:
    width, height = config.canvas_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    ppm = px_per_mm(config.dpi)
    px_sec = config.speed_mm_s * ppm
    px_mv = config.gain_mm_mv * ppm

    margin_x, margin_top, margin_bottom = 80, 120, 80
    content_x = margin_x
    content_y = margin_top
    col_duration = 2.5
    col_width = int(round(col_duration * px_sec))
    row_height = (height - margin_top - margin_bottom) // 4

    draw_header(canvas, ecg_id=ecg_id, patient_id=patient_id, config=config)
    scp_summary = "SCP: " + ", ".join(scp_codes.keys()) if scp_codes else "SCP: N/A"
    draw_footer(canvas, scp_summary=scp_summary, layout_label="3x4 + Rhythm II", width=width, height=height)

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
                    canvas, plot_x, plot_y, plot_w, plot_h,
                    px_per_mm_val=ppm,
                    major_color=config.grid_color,
                    minor_color=config.grid_minor_color,
                )

            t_start = col * col_duration
            t_end = t_start + col_duration
            segment = signals[lead_idx, int(t_start * fs): int(t_end * fs)]

            draw_calibration_pulse(canvas, mask, plot_x, baseline_y, px_per_second=px_sec, px_per_mv=px_mv)
            cal_offset = int(round(0.22 * px_sec))
            waveform_x = plot_x + cal_offset
            waveform_w = plot_w - cal_offset
            plot_waveform(canvas, mask, segment, waveform_x, baseline_y, waveform_w, px_per_mv=px_mv)

            label = LEAD_NAMES[lead_idx]
            label_x, label_y = plot_x + 8, plot_y + 28
            cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            leads.append(
                LeadRegion(
                    name=label,
                    lead_idx=lead_idx,
                    bbox=(region_x, region_y, col_width, row_height),
                    label_bbox=(label_x - 4, label_y - text_h - 4, text_w + 8, text_h + 8),
                    plot_bbox=(plot_x, plot_y, plot_w, plot_h),
                    waveform_bbox=(waveform_x, plot_y, waveform_w, plot_h),
                    t_start=t_start,
                    t_end=t_end,
                    baseline_y=baseline_y,
                )
            )

    rhythm_row = 3
    rhythm_y = content_y + rhythm_row * row_height
    rhythm_w = 4 * col_width
    rhythm_h = row_height
    plot_x = content_x + int(round(0.15 * px_sec))
    plot_y = rhythm_y + 20
    plot_w = rhythm_w - int(round(0.2 * px_sec))
    plot_h = rhythm_h - 40
    baseline_y = plot_y + plot_h // 2

    if config.show_grid:
        draw_ecg_grid(
            canvas, plot_x, plot_y, plot_w, plot_h,
            px_per_mm_val=ppm,
            major_color=config.grid_color,
            minor_color=config.grid_minor_color,
        )

    rhythm_signal = signals[1, : int(10 * fs)]
    draw_calibration_pulse(canvas, mask, plot_x, baseline_y, px_per_second=px_sec, px_per_mv=px_mv)
    cal_offset = int(round(0.22 * px_sec))
    waveform_x = plot_x + cal_offset
    waveform_w = plot_w - cal_offset
    plot_waveform(canvas, mask, rhythm_signal, waveform_x, baseline_y, waveform_w, px_per_mv=px_mv)

    label = "II"
    label_x, label_y = plot_x + 8, plot_y + 28
    cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    leads.append(
        LeadRegion(
            name="II",
            lead_idx=1,
            bbox=(content_x, rhythm_y, rhythm_w, rhythm_h),
            label_bbox=(label_x - 4, label_y - text_h - 4, text_w + 8, text_h + 8),
            plot_bbox=(plot_x, plot_y, plot_w, plot_h),
            waveform_bbox=(waveform_x, plot_y, waveform_w, plot_h),
            t_start=0.0,
            t_end=10.0,
            baseline_y=baseline_y,
        )
    )

    return RenderResult(
        image=canvas, mask=mask, width=width, height=height,
        leads=leads, px_per_mm=ppm, px_per_second=px_sec, px_per_mv=px_mv,
    )
