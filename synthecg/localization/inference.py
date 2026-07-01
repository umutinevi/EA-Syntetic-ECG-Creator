"""Dispatch localization inference from SCP codes and waveform morphology."""

from __future__ import annotations

import numpy as np

from synthecg.localization.pvc_otva import infer_pvc_localization
from synthecg.localization.types import LocalizationInfo
from synthecg.localization.wpw import infer_wpw_localization


def primary_scp_code(scp_codes: dict) -> str | None:
    if not scp_codes:
        return None
    return max(scp_codes.items(), key=lambda item: float(item[1]))[0]


def infer_localization_from_signal(
    signal: np.ndarray,
    fs: float,
    scp_codes: dict,
) -> LocalizationInfo | None:
    """Run morphology-based localization when SCP codes indicate PVC or WPW."""
    code = primary_scp_code(scp_codes)
    if code == "PVC":
        return infer_pvc_localization(signal, fs)
    if code == "WPW":
        return infer_wpw_localization(signal, fs)
    if code == "AFIB":
        return LocalizationInfo.not_applicable("Atrial fibrillation has no discrete ectopic origin")
    if code in {"PSVT", "AVNRT"}:
        return LocalizationInfo.from_algorithm(
            region=None,
            site="slow_fast_avnrt",
            algorithm="mechanism_proxy_v1",
            confidence=0.5,
            level="mechanism",
            features={"note": "PTB-XL PSVT used as AVNRT proxy; mechanism only"},
            decision_path=["psvt_proxy=avnrt_mechanism"],
        )
    return None


def infer_localization_from_record(record, scp_codes: dict) -> LocalizationInfo | None:
    signal = record.p_signal.T if record.p_signal.shape[1] == 12 else record.p_signal
    return infer_localization_from_signal(signal, float(record.fs), scp_codes)
