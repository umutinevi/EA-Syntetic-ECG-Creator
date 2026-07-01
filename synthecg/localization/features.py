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
    r = float(np.max(segment))
    s = float(-np.min(segment))
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


def average_beat_qrs(signal: np.ndarray, fs: float) -> np.ndarray:
    """Build an average 12-lead QRS using the first detected beat."""
    lead_ii = _lead_signal(signal, LEAD_II)
    peaks = detect_r_peaks(lead_ii, fs)
    if len(peaks) == 0:
        mid = len(lead_ii) // 2
        peaks = np.array([mid])
    peak = int(peaks[0])
    segments = []
    for lead_idx in range(12):
        lead = _lead_signal(signal, lead_idx)
        segments.append(lead[_qrs_window(lead, peak, fs)])
    min_len = min(len(s) for s in segments)
    return np.stack([s[:min_len] for s in segments])
