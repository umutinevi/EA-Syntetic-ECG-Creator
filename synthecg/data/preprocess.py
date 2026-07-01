"""Signal preprocessing utilities."""

from copy import copy

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter_signal(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter along the time axis."""
    nyquist = fs / 2.0
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.999)
    if low >= high:
        raise ValueError(f"Invalid bandpass range: {low_hz}-{high_hz} Hz at fs={fs}")

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered.astype(np.float32)


def preprocess_record(
    record,
    *,
    bandpass: bool = False,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
):
    """Return a copy of the record with optional bandpass filtering applied."""
    if not bandpass:
        return record

    processed = copy(record)
    processed.p_signal = bandpass_filter_signal(record.p_signal, record.fs, low_hz, high_hz)
    return processed
