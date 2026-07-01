"""Literature-based PVC / OT-VA localization algorithms."""

from __future__ import annotations

import numpy as np

from synthecg.localization.features import (
    beat_segments,
    isolate_pvc_beat,
    v1_r_wave_amplitude,
    v2_transition_ratio,
)
from synthecg.localization.taxonomy import region_for_site
from synthecg.localization.types import LocalizationInfo

ALGORITHM_NAME = "pvc_otva_v1"


def infer_pvc_localization(signal: np.ndarray, fs: float) -> LocalizationInfo | None:
    """
    Infer OT-VA origin from 12-lead morphology.

    Steps:
    1. Isolate ectopic vs sinus beat
    2. RVOT vs LVOT via V2 transition ratio (Betensky threshold 0.6)
    3. Refine LVOT to LCC when V1 shows a small r wave during the ectopic beat
    """
    decision_path: list[str] = []
    features: dict = {}

    sr_peak, pvc_peak = isolate_pvc_beat(signal, fs)
    if sr_peak is None or pvc_peak is None:
        return None

    decision_path.append(f"isolated_beats sr_peak={sr_peak} pvc_peak={pvc_peak}")
    sr, pvc = beat_segments(signal, fs, sr_peak, pvc_peak)

    ratio = v2_transition_ratio(sr, pvc)
    features["v2_transition_ratio"] = ratio
    if ratio is None:
        return None

    decision_path.append(f"v2_transition_ratio={ratio:.3f}")

    if ratio >= 0.6:
        region = "LVOT"
        confidence = min(0.95, 0.55 + (ratio - 0.6) * 0.5)
        v1_r = v1_r_wave_amplitude(pvc)
        features["v1_r_amplitude_mv"] = v1_r
        decision_path.append(f"v1_r_amplitude={v1_r:.4f}")

        if v1_r >= 0.02:
            site = "LCC"
            decision_path.append("LCC_hint=v1_r_wave_present")
            confidence = min(0.92, confidence + 0.1)
        elif ratio >= 1.0:
            site = "AMC"
            decision_path.append("fallback_site=AMC")
        else:
            site = "Summit"
            decision_path.append("fallback_site=Summit")
    else:
        region = "RVOT"
        confidence = min(0.95, 0.55 + (0.6 - ratio) * 0.5)
        lead_ii_pvc = pvc[1]
        if float(np.max(lead_ii_pvc)) > float(-np.min(lead_ii_pvc)) * 1.2:
            site = "free_wall"
            decision_path.append("rvot_morphology=free_wall_pattern")
        else:
            site = "posterior_septal"
            decision_path.append("rvot_morphology=septal_pattern")

    assert region_for_site(site) == region
    return LocalizationInfo.from_algorithm(
        region=region,
        site=site,
        algorithm=ALGORITHM_NAME,
        confidence=confidence,
        features=features,
        decision_path=decision_path,
    )
