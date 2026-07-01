"""Literature-based PVC / OT-VA localization (Betensky, Yoshida, Di)."""

from __future__ import annotations

from synthecg.localization.features import (
    beat_segments,
    isolate_pvc_beat,
    precordial_transition_lead,
    rs_ratio,
    v1_r_wave_amplitude,
    v1_v3_transition_index,
    v2_transition_ratio,
    v2s_v3r_index,
    LEAD_V1,
)
from synthecg.localization.literature import (
    PVC_ALGORITHM_CONFIDENCE_CAP,
    PVC_BETENSKY_2011,
    PVC_DI_V1V3_2021,
    PVC_LCC_MORPHOLOGY,
    PVC_YOSHIDA_TRANSITION_2011,
    PVC_YOSHIDA_V2S_V3R_2014,
)
from synthecg.localization.taxonomy import region_for_site
from synthecg.localization.types import LocalizationInfo

ALGORITHM_NAME = "pvc_otva_literature_v2"
LITERATURE_REFS = [
    PVC_BETENSKY_2011,
    PVC_YOSHIDA_V2S_V3R_2014,
    PVC_YOSHIDA_TRANSITION_2011,
    PVC_DI_V1V3_2021,
    PVC_LCC_MORPHOLOGY,
]

BETENSKY_LVOT_THRESHOLD = 0.6
YOSHIDA_LVOT_V2S_V3R_THRESHOLD = 1.5
DI_RVOT_V1V3_THRESHOLD = -1.60
LCC_V1_R_THRESHOLD_MV = 0.02


def infer_pvc_localization(signal, fs: float) -> LocalizationInfo | None:
    """
    Multi-step OT-VA localization from published ECG criteria.

    1. Betensky V2 transition ratio (>= 0.6 -> LVOT)
    2. Yoshida V2S/V3R index confirmation when available
    3. LVOT sub-sites: LCC when V1 shows a small r wave (cusp morphology)
    4. RVOT sub-sites: Di V1-V3 transition index for septal vs free wall
    """
    decision_path: list[str] = []
    features: dict = {}

    sr_peak, pvc_peak = isolate_pvc_beat(signal, fs)
    if sr_peak is None or pvc_peak is None:
        return None

    decision_path.append(f"isolated_beats sr_peak={sr_peak} pvc_peak={pvc_peak}")
    sr, pvc = beat_segments(signal, fs, sr_peak, pvc_peak)

    v2_ratio = v2_transition_ratio(sr, pvc)
    features["v2_transition_ratio"] = v2_ratio
    if v2_ratio is None:
        return None
    decision_path.append(f"betensky_v2_transition_ratio={v2_ratio:.3f}")

    v2s_v3r = v2s_v3r_index(pvc)
    features["v2s_v3r_index"] = v2s_v3r
    if v2s_v3r is not None:
        decision_path.append(f"yoshida_v2s_v3r={v2s_v3r:.3f}")

    transition_lead = precordial_transition_lead(pvc)
    features["precordial_transition_lead"] = transition_lead
    if transition_lead is not None:
        decision_path.append(f"precordial_transition=V{transition_lead}")

    lvot_votes = 0
    if v2_ratio >= BETENSKY_LVOT_THRESHOLD:
        lvot_votes += 1
        decision_path.append("betensky_vote=LVOT")
    if v2s_v3r is not None and v2s_v3r < YOSHIDA_LVOT_V2S_V3R_THRESHOLD:
        lvot_votes += 1
        decision_path.append("yoshida_vote=LVOT")

    is_lvot = lvot_votes >= 1 or v2_ratio >= BETENSKY_LVOT_THRESHOLD

    if is_lvot:
        region = "LVOT"
        confidence = 0.55 + min(0.15, max(0.0, v2_ratio - BETENSKY_LVOT_THRESHOLD))
        v1_r = v1_r_wave_amplitude(pvc)
        features["v1_r_amplitude_mv"] = v1_r
        decision_path.append(f"v1_r_amplitude={v1_r:.4f}")

        if v1_r >= LCC_V1_R_THRESHOLD_MV:
            site = "LCC"
            decision_path.append("lcc_morphology=v1_small_r_wave")
            confidence += 0.08
        elif v2_ratio >= 1.0:
            site = "AMC"
            decision_path.append("lvot_subsite=AMC_by_transition")
        else:
            site = "Summit"
            decision_path.append("lvot_subsite=Summit_fallback")
    else:
        region = "RVOT"
        confidence = 0.55 + min(0.15, max(0.0, BETENSKY_LVOT_THRESHOLD - v2_ratio))
        v1v3 = v1_v3_transition_index(sr, pvc)
        features["v1_v3_transition_index"] = v1v3
        v1_rs = rs_ratio(pvc[LEAD_V1])
        features["v1_rs_ratio"] = v1_rs

        if v1v3 is not None:
            decision_path.append(f"di_v1_v3_index={v1v3:.3f}")
            if v1v3 > DI_RVOT_V1V3_THRESHOLD:
                site = "posterior_septal"
                decision_path.append("di_vote=RVOT_septal")
            elif v1_rs >= 0.35:
                site = "free_wall"
                decision_path.append("v1_rs_vote=RVOT_free_wall")
            else:
                site = "anterior_septal"
                decision_path.append("rvot_subsite=anterior_septal")
        elif v1_rs >= 0.35:
            site = "free_wall"
            decision_path.append("fallback=RVOT_free_wall_by_v1_rs")
        else:
            site = "posterior_septal"
            decision_path.append("fallback=RVOT_posterior_septal")

    assert region_for_site(site) == region
    return LocalizationInfo.from_algorithm(
        region=region,
        site=site,
        algorithm=ALGORITHM_NAME,
        confidence=confidence,
        features=features,
        decision_path=decision_path,
        literature_references=LITERATURE_REFS,
        confidence_cap=PVC_ALGORITHM_CONFIDENCE_CAP,
    )
