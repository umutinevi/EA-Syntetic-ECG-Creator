"""Literature-based WPW accessory pathway localization (Arruda + Milstein)."""

from __future__ import annotations

from synthecg.localization.features import (
    LEAD_III,
    LEAD_V1,
    extract_delta_polarities,
    extract_qrs_segment_at_peak,
    rs_ratio,
)
from synthecg.localization.literature import (
    WPW_ALGORITHM_CONFIDENCE_CAP,
    WPW_ARRUDA_1998,
    WPW_MILSTEIN_2008,
)
from synthecg.localization.types import LocalizationInfo

ALGORITHM_NAME = "wpw_arruda_milstein_v2"
LITERATURE_REFS = [WPW_ARRUDA_1998, WPW_MILSTEIN_2008]


def infer_wpw_localization(signal, fs: float) -> LocalizationInfo | None:
    """
    Accessory pathway localization using published ECG rules.

    Primary rules (Arruda et al. 1998):
    - Negative delta in lead II -> coronary venous / CS region (CS OS territory)
    - Delta polarity in I, II, aVF, V1 plus R/S in III and V1 for annular region

    Secondary rules (Milstein / Chiang 2008):
    - Negative delta in II, III, or aVF -> right-sided (septal or lateral)
    - V1 morphology distinguishes septal/CS vs lateral/free wall
    """
    qrs = extract_qrs_segment_at_peak(signal, fs)
    decision_path: list[str] = []
    features: dict = {}

    pol = extract_delta_polarities(signal, fs)
    features["delta_polarity_20ms"] = pol
    decision_path.append(f"arruda_delta_polarity={pol}")

    rs_v1 = rs_ratio(qrs[LEAD_V1])
    rs_iii = rs_ratio(qrs[LEAD_III])
    features["rs_v1"] = rs_v1
    features["rs_iii"] = rs_iii
    decision_path.append(f"arruda_rs_v1={rs_v1:.3f} rs_iii={rs_iii:.3f}")

    region = "AP"
    confidence = 0.45

    # Arruda: truly negative delta in lead II -> coronary venous system (CS OS).
    if pol["II"] == "neg":
        site = "coronary_sinus_ostium"
        decision_path.append("arruda_rule=negative_delta_II_coronary_venous_CS_OS")
        confidence = 0.48

    # Milstein: inferior negative deltas -> right-sided; V1 negative + low R/S -> septal/CS.
    elif pol["III"] == "neg" or pol["aVF"] == "neg":
        if pol["V1"] == "neg" and rs_v1 < 0.5:
            site = "coronary_sinus_ostium"
            decision_path.append("milstein_rule=inferior_neg_v1_neg_low_rs_CS_OS")
        elif pol["V1"] == "pos" or rs_v1 >= 0.5:
            site = "right_lateral"
            decision_path.append("milstein_rule=inferior_neg_v1_pos_right_lateral")
        else:
            site = "posteroseptal"
            decision_path.append("milstein_rule=inferior_neg_posteroseptal")
        confidence = 0.46

    # Right-sided lateral: positive II, negative I, positive V1 or dominant R in V1.
    elif pol["I"] == "neg" and pol["II"] == "pos" and (pol["V1"] == "pos" or rs_v1 >= 0.5):
        site = "right_free_wall"
        decision_path.append("arruda_milstein_rule=right_lateral_free_wall_pattern")
        confidence = 0.44

    # Septal / CS OS pattern: negative I and V1 with low R/S (EP-confused with free wall).
    elif pol["I"] == "neg" and pol["V1"] == "neg" and rs_v1 < 0.5:
        site = "coronary_sinus_ostium"
        decision_path.append("combined_rule=I_neg_V1_neg_low_rs_CS_OS_not_free_wall")
        confidence = 0.47

    # Left-sided pathways.
    elif pol["I"] == "pos" and pol["II"] in {"pos", "iso"}:
        if pol["V1"] == "neg":
            site = "left_lateral"
            decision_path.append("arruda_rule=left_lateral")
        else:
            site = "left_septal"
            decision_path.append("arruda_rule=left_septal")
        confidence = 0.44

    elif rs_iii >= 0.5:
        site = "anteroseptal"
        decision_path.append("fallback=anteroseptal_by_rsIII")
        confidence = 0.42
    else:
        site = "posteroseptal"
        decision_path.append("fallback=posteroseptal_indeterminate")
        confidence = 0.40

    features["clinical_note"] = (
        "Algorithm prediction only — requires EP/ablation confirmation before clinical use."
    )

    return LocalizationInfo.from_algorithm(
        region=region,
        site=site,
        algorithm=ALGORITHM_NAME,
        confidence=confidence,
        features=features,
        decision_path=decision_path,
        literature_references=LITERATURE_REFS,
        confidence_cap=WPW_ALGORITHM_CONFIDENCE_CAP,
    )
