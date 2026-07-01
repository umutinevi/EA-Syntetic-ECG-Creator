"""Clinical label overlays for ECG example images."""

from __future__ import annotations

import cv2
import numpy as np


def add_clinical_overlay(
    image: np.ndarray,
    *,
    primary_label: str,
    secondary_label: str | None = None,
    scp_codes: str | None = None,
) -> np.ndarray:
    """Add a clinical diagnosis banner to the bottom of an ECG image."""
    img = image.copy()
    height, width = img.shape[:2]

    banner_h = 90 if secondary_label else 70
    y0 = height - banner_h - 10
    overlay = img.copy()
    cv2.rectangle(overlay, (60, y0), (width - 60, height - 10), (255, 255, 240), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (60, y0), (width - 60, height - 10), (0, 0, 80), 2)

    cv2.putText(
        img,
        primary_label,
        (80, y0 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 120),
        2,
        cv2.LINE_AA,
    )

    line_y = y0 + 58
    if secondary_label:
        cv2.putText(
            img,
            secondary_label,
            (80, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        line_y += 22

    if scp_codes:
        cv2.putText(
            img,
            f"PTB-XL SCP: {scp_codes}",
            (80, line_y if secondary_label else y0 + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    return img
