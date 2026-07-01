"""Literature-based WPW accessory pathway localization (simplified Arruda-style)."""

from __future__ import annotations

import numpy as np

from synthecg.localization.features import (
    LEAD_AVF,
    LEAD_I,
    LEAD_II,
    LEAD_III,
    LEAD_V1,
    average_beat_qrs,
    delta_polarity,
)
from synthecg.localization.types import LocalizationInfo

ALGORITHM_NAME = "wpw_arruda_simplified_v1"


def _rs_ratio(segment: np.ndarray) -> float:
    r = float(np.max(segment))
    s = float(-np.min(segment))
    if r + s <= 1e-6:
        return 0.0
    return r / (r + s)


def infer_wpw_localization(signal: np.ndarray, fs: float) -> LocalizationInfo | None:
    """
    Simplified delta-wave algorithm inspired by Arruda et al.

    Uses polarity of the first 20 ms of QRS in I, II, aVF, V1 and R/S in III/V1.
    """
    qrs = average_beat_qrs(signal, fs)
    decision_path: list[str] = []
    features: dict = {}

    pol = {
        "I": delta_polarity(qrs[LEAD_I], fs),
        "II": delta_polarity(qrs[LEAD_II], fs),
        "aVF": delta_polarity(qrs[LEAD_AVF], fs),
        "V1": delta_polarity(qrs[LEAD_V1], fs),
    }
    features["delta_polarity"] = pol
    decision_path.append(f"delta_polarity={pol}")

    rs_v1 = _rs_ratio(qrs[LEAD_V1])
    rs_iii = _rs_ratio(qrs[LEAD_III])
    features["rs_v1"] = rs_v1
    features["rs_iii"] = rs_iii
    decision_path.append(f"rs_v1={rs_v1:.3f} rs_iii={rs_iii:.3f}")

    right_sided = pol["II"] == "neg" or pol["aVF"] == "neg" or pol["I"] == "neg"
    left_sided = pol["I"] == "pos" and pol["II"] in {"pos", "iso"} and not right_sided

    if right_sided:
        region = "AP"
        if pol["V1"] == "neg" and rs_v1 < 0.5:
            site = "right_free_wall"
            confidence = 0.82
            decision_path.append("right_free_wall=V1_neg_low_rs")
        elif pol["V1"] == "pos":
            site = "right_lateral"
            confidence = 0.78
            decision_path.append("right_lateral=V1_pos")
        else:
            site = "posteroseptal"
            confidence = 0.72
            decision_path.append("fallback=posteroseptal")
    elif left_sided:
        region = "AP"
        if pol["V1"] == "neg":
            site = "left_lateral"
            confidence = 0.8
            decision_path.append("left_lateral=V1_neg")
        else:
            site = "left_septal"
            confidence = 0.75
            decision_path.append("left_septal=V1_pos")
    else:
        region = "AP"
        if rs_iii >= 0.5:
            site = "anteroseptal"
            confidence = 0.65
        else:
            site = "posteroseptal"
            confidence = 0.6
        decision_path.append("indeterminate_axis=posterior_fallback")

    return LocalizationInfo.from_algorithm(
        region=region,
        site=site,
        algorithm=ALGORITHM_NAME,
        confidence=confidence,
        features=features,
        decision_path=decision_path,
    )
