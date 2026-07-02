"""ECG morphology feature extraction for localization algorithms."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from synthecg.config import LEAD_NAMES

# Lead indices for standard 12-lead order.
LEAD_I = LEAD_NAMES.index("I")
LEAD_II = LEAD_NAMES.index("II")
LEAD_III = LEAD_NAMES.index("III")
LEAD_AVR = LEAD_NAMES.index("aVR")
LEAD_AVL = LEAD_NAMES.index("aVL")
LEAD_AVF = LEAD_NAMES.index("aVF")
LEAD_V1 = LEAD_NAMES.index("V1")
LEAD_V2 = LEAD_NAMES.index("V2")
LEAD_V3 = LEAD_NAMES.index("V3")
LEAD_V4 = LEAD_NAMES.index("V4")
LEAD_V5 = LEAD_NAMES.index("V5")
LEAD_V6 = LEAD_NAMES.index("V6")


def _lead_signal(signal: np.ndarray, lead_idx: int) -> np.ndarray:
    if signal.ndim == 2:
        if signal.shape[0] == 12:
            return signal[lead_idx]
        return signal[:, lead_idx]
    raise ValueError(f"Expected 2D signal, got shape {signal.shape}")


def detect_r_peaks(lead: np.ndarray, fs: float) -> np.ndarray:
    """Detect R peaks on a single lead using a simple energy envelope."""
    diff = np.diff(lead, prepend=lead[0])
    envelope = np.abs(diff)
    min_distance = max(1, int(0.25 * fs))
    height = np.percentile(envelope, 75)
    peaks, _ = find_peaks(envelope, distance=min_distance, height=height)
    return peaks


def _qrs_window(lead: np.ndarray, peak: int, fs: float) -> slice:
    pre = int(0.08 * fs)
    post = int(0.08 * fs)
    start = max(0, peak - pre)
    end = min(len(lead), peak + post)
    return slice(start, end)


def _max_r_min_s(segment: np.ndarray) -> tuple[float, float]:
    """Return (R, S) wave amplitudes as non-negative deflections from baseline.

    R is an upward deflection and S a downward deflection relative to the
    isoelectric line (0 on bandpass-filtered signals). A monophasic QRS has no
    opposing wave, so the missing wave's amplitude is 0 — never negative.
    Clamping matters: an unclamped ``-min`` goes negative for an all-positive
    window and corrupts every downstream ratio (e.g. the Yoshida V2S/V3R index),
    which can flip the predicted site of origin.
    """
    if segment.size == 0:
        return 0.0, 0.0
    r = float(max(0.0, np.max(segment)))
    s = float(max(0.0, -np.min(segment)))
    return r, s


def isolate_pvc_beat(
    signal: np.ndarray,
    fs: float,
    *,
    reference_lead: int = LEAD_II,
) -> tuple[int | None, int | None]:
    """Return indices of (sinus_beat_peak, pvc_beat_peak) or (None, None)."""
    lead = _lead_signal(signal, reference_lead)
    peaks = detect_r_peaks(lead, fs)
    # Drop peaks whose full QRS window falls off the record edges; an edge
    # artifact selected as the sinus reference beat yields a truncated window
    # and unreliable amplitudes.
    margin = int(0.08 * fs)
    interior = peaks[(peaks >= margin) & (peaks < len(lead) - margin)]
    if len(interior) >= 2:
        peaks = interior
    if len(peaks) < 2:
        return None, None

    energies = []
    for peak in peaks:
        window = _qrs_window(lead, peak, fs)
        energies.append(float(np.sum(lead[window] ** 2)))
    energies = np.asarray(energies)

    pvc_idx = int(np.argmax(energies))
    if len(peaks) == 2:
        sr_idx = 1 - pvc_idx
    else:
        median_energy = float(np.median(energies))
        candidates = [i for i, e in enumerate(energies) if i != pvc_idx and e <= median_energy * 1.2]
        sr_idx = candidates[0] if candidates else (pvc_idx - 1 if pvc_idx > 0 else pvc_idx + 1)

    if sr_idx == pvc_idx:
        return None, None
    return int(peaks[sr_idx]), int(peaks[pvc_idx])


def beat_segments(signal: np.ndarray, fs: float, sr_peak: int, ectopic_peak: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract 12-lead QRS windows for sinus and ectopic beats."""
    sr_slices = []
    ectopic_slices = []
    for lead_idx in range(12):
        lead = _lead_signal(signal, lead_idx)
        sr_slices.append(lead[_qrs_window(lead, sr_peak, fs)])
        ectopic_slices.append(lead[_qrs_window(lead, ectopic_peak, fs)])

    min_len = min(min(len(s) for s in sr_slices), min(len(s) for s in ectopic_slices))
    sr = np.stack([s[:min_len] for s in sr_slices])
    ectopic = np.stack([s[:min_len] for s in ectopic_slices])
    return sr, ectopic


def v2_transition_ratio(sr: np.ndarray, ectopic: np.ndarray) -> float | None:
    """Betensky V2 transition ratio comparing PVC to preceding sinus beat."""
    sr_v2 = sr[LEAD_V2]
    ect_v2 = ectopic[LEAD_V2]
    sr_r, sr_s = _max_r_min_s(sr_v2)
    ect_r, ect_s = _max_r_min_s(ect_v2)
    if sr_r + sr_s <= 1e-6 or ect_r + ect_s <= 1e-6:
        return None
    sr_ratio = sr_r / (sr_r + sr_s)
    ect_ratio = ect_r / (ect_r + ect_s)
    if sr_ratio <= 1e-6:
        return None
    return float(ect_ratio / sr_ratio)


def rs_ratio(segment: np.ndarray) -> float:
    r, s = _max_r_min_s(segment)
    if r + s <= 1e-6:
        return 0.0
    return float(r / (r + s))


def v2s_v3r_index(ectopic: np.ndarray) -> float | None:
    """Yoshida V2S/V3R index during the ectopic beat (< 1.5 suggests LVOT)."""
    _, s_v2 = _max_r_min_s(ectopic[LEAD_V2])
    r_v3, _ = _max_r_min_s(ectopic[LEAD_V3])
    if r_v3 <= 1e-6:
        return None
    return float(s_v2 / r_v3)


def v1_v3_transition_index(sr: np.ndarray, ectopic: np.ndarray) -> float | None:
    """
    Di et al. V1-V3 transition index.

    > -1.60 predicts RVOT origin (vs LVOT) when precordial transition is in V3.
    """
    terms_s = []
    terms_r = []
    for lead_idx in (LEAD_V1, LEAD_V2):
        sr_r, sr_s = _max_r_min_s(sr[lead_idx])
        ect_r, ect_s = _max_r_min_s(ectopic[lead_idx])
        if sr_s <= 1e-6 or sr_r <= 1e-6:
            return None
        terms_s.append(ect_s / sr_s)
    for lead_idx in (LEAD_V1, LEAD_V2, LEAD_V3):
        sr_r, sr_s = _max_r_min_s(sr[lead_idx])
        ect_r, ect_s = _max_r_min_s(ectopic[lead_idx])
        if sr_r <= 1e-6:
            return None
        terms_r.append(ect_r / sr_r)
    return float(sum(terms_s) - sum(terms_r))


def precordial_transition_lead(ectopic: np.ndarray) -> int | None:
    """First precordial lead where R >= S during the ectopic beat (V1=0 .. V6=5)."""
    for offset, lead_idx in enumerate((LEAD_V1, LEAD_V2, LEAD_V3, LEAD_V4, LEAD_V5, LEAD_V6)):
        r, s = _max_r_min_s(ectopic[lead_idx])
        if r >= s:
            return offset + 1
    return None


def v1_r_wave_amplitude(ectopic: np.ndarray) -> float:
    """Peak positive deflection in V1 during ectopic beat (LCC often shows small r)."""
    return float(np.max(ectopic[LEAD_V1]))


def delta_polarity(lead_segment: np.ndarray, fs: float, ms: int = 20) -> str:
    """Classify early QRS/delta polarity as positive, negative, or isoelectric."""
    n = max(1, int(ms / 1000.0 * fs))
    early = lead_segment[:n]
    amp = float(np.max(early) - np.min(early))
    peak = float(np.max(early))
    trough = float(np.min(early))
    if amp < 0.02:
        return "iso"
    if peak >= abs(trough):
        return "pos"
    return "neg"


def extract_delta_polarities(signal: np.ndarray, fs: float, ms: int = 20) -> dict[str, str]:
    """Arruda-style delta polarity in I, II, III, aVF, V1 at the QRS onset."""
    lead_ii = _lead_signal(signal, LEAD_II)
    peaks = detect_r_peaks(lead_ii, fs)
    if len(peaks) == 0:
        peak = len(lead_ii) // 2
    else:
        peak = int(peaks[0])

    onset = max(0, peak - int(0.04 * fs))
    n = max(1, int(ms / 1000.0 * fs))
    labels = {
        "I": LEAD_I,
        "II": LEAD_II,
        "III": LEAD_III,
        "aVF": LEAD_AVF,
        "V1": LEAD_V1,
    }
    pol: dict[str, str] = {}
    for name, lead_idx in labels.items():
        lead = _lead_signal(signal, lead_idx)
        end = min(len(lead), onset + n)
        pol[name] = delta_polarity(lead[onset:end], fs, ms=ms)
    return pol


def extract_qrs_segment_at_peak(signal: np.ndarray, fs: float) -> np.ndarray:
    """12-lead QRS segment aligned to the primary R peak (lead II)."""
    lead_ii = _lead_signal(signal, LEAD_II)
    peaks = detect_r_peaks(lead_ii, fs)
    peak = int(peaks[0]) if len(peaks) else len(lead_ii) // 2
    segments = []
    for lead_idx in range(12):
        lead = _lead_signal(signal, lead_idx)
        segments.append(lead[_qrs_window(lead, peak, fs)])
    min_len = min(len(s) for s in segments)
    return np.stack([s[:min_len] for s in segments])
