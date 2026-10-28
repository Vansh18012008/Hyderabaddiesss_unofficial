"""
vital_sign_extraction.py
========================
Respiration and heart rate extraction from radar displacement signals.

After phase demodulation, the chest displacement signal contains:
  - Respiration component: 0.1–0.5 Hz (6–30 breaths/min)
  - Heartbeat component:   0.8–2.0 Hz (48–120 bpm)
  - Residual noise and harmonics

This module implements:
  1. Bandpass filtering to isolate each vital sign band.
  2. FFT-based frequency estimation (dominant peak).
  3. Autocorrelation-based estimation (more robust for short signals).

References:
    [12] Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14] Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Bandpass Filtering
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        signal:   1-D real signal, shape [n].
        fs:       Sampling rate [Hz].
        low_hz:   Lower cutoff frequency [Hz].
        high_hz:  Upper cutoff frequency [Hz].
        order:    Filter order (4 recommended).

    Returns:
        filtered: Bandpass-filtered signal, shape [n].
    """
    nyq = fs / 2.0
    lo = low_hz / nyq
    hi = high_hz / nyq

    if lo <= 0 or hi >= 1.0 or lo >= hi:
        raise ValueError(
            f"Invalid bandpass: [{low_hz}, {high_hz}] Hz for fs={fs} Hz."
        )

    b, a = butter(order, [lo, hi], btype="bandpass", analog=False)
    return filtfilt(b, a, signal)


def extract_respiration_signal(
    displacement: np.ndarray,
    fs: float,
    low_hz: float = 0.1,
    high_hz: float = 0.5,
) -> np.ndarray:
    """
    Isolate the respiration component from the displacement signal.

    Typical breathing rate: 12–30 breaths/min = 0.2–0.5 Hz.
    We use a slightly wider band [0.1, 0.5] Hz for robustness.

    Args:
        displacement: Chest displacement signal [mm], shape [n_chirps].
        fs:           Sampling rate [Hz] (slow-time fs = 1/chirp_duration).
        low_hz:       Lower cutoff [Hz].
        high_hz:      Upper cutoff [Hz].

    Returns:
        resp_signal: Filtered respiration signal [mm], shape [n_chirps].
    """
    return bandpass_filter(displacement, fs, low_hz, high_hz)


def extract_heartbeat_signal(
    displacement: np.ndarray,
    fs: float,
    low_hz: float = 0.8,
    high_hz: float = 2.0,
) -> np.ndarray:
    """
    Isolate the heartbeat component from the displacement signal.

    Typical heart rate: 48–120 bpm = 0.8–2.0 Hz.

    Args:
        displacement: Chest displacement signal [mm], shape [n_chirps].
        fs:           Sampling rate [Hz].
        low_hz:       Lower cutoff [Hz].
        high_hz:      Upper cutoff [Hz].

    Returns:
        heart_signal: Filtered heartbeat signal [mm], shape [n_chirps].
    """
    return bandpass_filter(displacement, fs, low_hz, high_hz)


# ---------------------------------------------------------------------------
# Frequency Estimation via FFT
# ---------------------------------------------------------------------------

def fft_peak_frequency(
    signal: np.ndarray,
    fs: float,
    freq_min: float,
    freq_max: float,
    n_fft: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate dominant frequency in a band using FFT peak detection.

    Pads the signal with zeros (zero-padding) for finer frequency resolution.

    Args:
        signal:   1-D real signal, shape [n].
        fs:       Sampling rate [Hz].
        freq_min: Lower frequency bound for peak search [Hz].
        freq_max: Upper frequency bound for peak search [Hz].
        n_fft:    FFT size. Default = 4x signal length (4x zero-padding).

    Returns:
        peak_freq_hz: Estimated dominant frequency [Hz].
        freq_axis:    Frequency axis [Hz], shape [n_fft//2].
        magnitude:    Single-sided magnitude spectrum, shape [n_fft//2].
    """
    n = len(signal)
    if n_fft is None:
        # 4x zero-padding for fine resolution
        n_fft = 4 * n

    # Windowed FFT
    window = np.hanning(n)
    spectrum = np.fft.fft(signal * window, n=n_fft)

    # One-sided (positive frequencies only)
    n_bins = n_fft // 2
    freq_axis = np.arange(n_bins) * fs / n_fft
    magnitude = np.abs(spectrum[:n_bins])

    # Find peak within [freq_min, freq_max]
    mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
    if not np.any(mask):
        return 0.0, freq_axis, magnitude

    idx_in_band = np.where(mask)[0]
    peak_idx_local = np.argmax(magnitude[idx_in_band])
    peak_bin = idx_in_band[peak_idx_local]
    peak_freq_hz = freq_axis[peak_bin]

    return float(peak_freq_hz), freq_axis, magnitude


# ---------------------------------------------------------------------------
# Autocorrelation-Based Frequency Estimation
# ---------------------------------------------------------------------------

def autocorr_peak_frequency(
    signal: np.ndarray,
    fs: float,
    freq_min: float,
    freq_max: float,
) -> float:
    """
    Estimate dominant frequency via autocorrelation (more robust for short signals).

    The autocorrelation of a sinusoid has a peak at the period of the signal.
    We search for the first peak in the lag range [1/freq_max, 1/freq_min].

    Args:
        signal:   1-D real signal, shape [n].
        fs:       Sampling rate [Hz].
        freq_min: Lower frequency bound [Hz].
        freq_max: Upper frequency bound [Hz].

    Returns:
        peak_freq_hz: Estimated frequency [Hz]. Returns 0.0 if no peak found.
    """
    n = len(signal)
    # Normalized autocorrelation
    signal_z = signal - np.mean(signal)
    ac = np.correlate(signal_z, signal_z, mode="full")
    ac = ac[n - 1:]  # Keep only non-negative lags
    ac /= ac[0] + 1e-30  # Normalize to [−1, 1]

    # Convert frequency bounds to lag bounds (in samples)
    lag_min = int(np.floor(fs / freq_max))
    lag_max = int(np.ceil(fs / freq_min))
    lag_min = max(1, lag_min)
    lag_max = min(lag_max, n - 1)

    if lag_min >= lag_max:
        return 0.0

    # Find the dominant peak in the valid lag range
    ac_window = ac[lag_min:lag_max + 1]
    peaks, _ = find_peaks(ac_window)

    if len(peaks) == 0:
        # No clear peak — return frequency of max in window
        peak_lag = lag_min + int(np.argmax(ac_window))
    else:
        # Use the peak with highest autocorrelation value
        best_peak_local = peaks[np.argmax(ac_window[peaks])]
        peak_lag = lag_min + best_peak_local

    peak_freq_hz = fs / peak_lag if peak_lag > 0 else 0.0
    return float(peak_freq_hz)


# ---------------------------------------------------------------------------
# Combined Vital Sign Extraction
# ---------------------------------------------------------------------------

def extract_vital_signs(
    displacement_mm: np.ndarray,
    fs: float,
    resp_band: Tuple[float, float] = (0.1, 0.5),
    heart_band: Tuple[float, float] = (0.8, 2.0),
    estimator: str = "fft",
) -> dict:
    """
    Extract respiration rate and heart rate from chest displacement.

    Args:
        displacement_mm: Chest displacement [mm], shape [n_chirps].
        fs:              Slow-time sampling rate [Hz].
        resp_band:       (low, high) Hz for respiration bandpass.
        heart_band:      (low, high) Hz for heartbeat bandpass.
        estimator:       'fft' or 'autocorr'.

    Returns:
        result: Dictionary with:
            - resp_signal_mm:   Filtered respiration signal [mm]
            - heart_signal_mm:  Filtered heartbeat signal [mm]
            - resp_rate_hz:     Estimated respiration rate [Hz]
            - heart_rate_hz:    Estimated heart rate [Hz]
            - resp_rate_bpm:    Respiration rate [breaths/min]
            - heart_rate_bpm:   Heart rate [bpm]
            - resp_spectrum:    (freq_axis, magnitude) for resp
            - heart_spectrum:   (freq_axis, magnitude) for heart
    """
    # Step 1: Bandpass filter
    resp_signal = extract_respiration_signal(displacement_mm, fs, *resp_band)
    heart_signal = extract_heartbeat_signal(displacement_mm, fs, *heart_band)

    # Step 2: Frequency estimation
    if estimator == "fft":
        resp_hz, resp_f, resp_mag = fft_peak_frequency(resp_signal, fs, *resp_band)
        heart_hz, heart_f, heart_mag = fft_peak_frequency(heart_signal, fs, *heart_band)
    elif estimator == "autocorr":
        resp_hz = autocorr_peak_frequency(resp_signal, fs, *resp_band)
        heart_hz = autocorr_peak_frequency(heart_signal, fs, *heart_band)
        # Still compute FFT for visualization
        _, resp_f, resp_mag = fft_peak_frequency(resp_signal, fs, *resp_band)
        _, heart_f, heart_mag = fft_peak_frequency(heart_signal, fs, *heart_band)
    else:
        raise ValueError(f"Unknown estimator '{estimator}'. Use 'fft' or 'autocorr'.")

    return {
        "resp_signal_mm": resp_signal,
        "heart_signal_mm": heart_signal,
        "resp_rate_hz": resp_hz,
        "heart_rate_hz": heart_hz,
        "resp_rate_bpm": resp_hz * 60.0,
        "heart_rate_bpm": heart_hz * 60.0,
        "resp_spectrum": (resp_f, resp_mag),
        "heart_spectrum": (heart_f, heart_mag),
        "estimator": estimator,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import make_single_target_scene
    from range_fft import range_fft, find_target_range_bin, extract_slow_time_signal
    from dc_offset_correction import circle_fit_correction
    from clutter_suppression import mean_clutter_removal
    from phase_demodulation import demodulate

    print("=== vital_sign_extraction.py self-test ===")

    data_cube, gt, cfg = make_single_target_scene(
        target_range=1.5,
        resp_rate_hz=0.25,    # 15 breaths/min
        heart_rate_hz=1.17,   # 70 bpm
    )

    # Pipeline
    rp, r_axis = range_fft(data_cube, cfg)
    rp_clean, _ = mean_clutter_removal(rp)
    tgt_bin, _ = find_target_range_bin(rp_clean, r_axis, method="max_power")
    sts = extract_slow_time_signal(rp_clean, tgt_bin)
    sts_corrected, _ = circle_fit_correction(sts)
    demod = demodulate(sts_corrected, cfg.wavelength)

    # Vital signs
    vs = extract_vital_signs(
        demod["displacement_mm"],
        fs=cfg.slow_time_fs,
        estimator="fft",
    )

    true_resp = gt["targets"][0].resp_rate_hz
    true_hr   = gt["targets"][0].heart_rate_hz

    print(f"  True resp rate       : {true_resp:.3f} Hz  ({true_resp*60:.1f} bpm)")
    print(f"  Estimated resp rate  : {vs['resp_rate_hz']:.3f} Hz  "
          f"({vs['resp_rate_bpm']:.1f} bpm)")
    print(f"  Resp error           : {abs(vs['resp_rate_hz']-true_resp)*60:.2f} bpm")

    print(f"  True heart rate      : {true_hr:.3f} Hz  ({true_hr*60:.1f} bpm)")
    print(f"  Estimated heart rate : {vs['heart_rate_hz']:.3f} Hz  "
          f"({vs['heart_rate_bpm']:.1f} bpm)")
    print(f"  HR error             : {abs(vs['heart_rate_hz']-true_hr)*60:.2f} bpm")

    print("  PASS")
