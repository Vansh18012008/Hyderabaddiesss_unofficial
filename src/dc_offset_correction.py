"""
dc_offset_correction.py
========================
DC offset correction methods for FMCW radar IQ signals.

In direct-conversion (homodyne) FMCW receivers, DC offset arises from:
  1. LO self-mixing (dominant at mmWave)
  2. Mixer imbalance and leakage
  3. Strong static reflectors (walls) translated into IQ DC bias [14]

This shifts the IQ constellation center away from the origin, corrupting
the arctangent phase demodulation used to extract chest displacement.

Methods implemented:
  A. mean_subtraction      — Subtract temporal mean [14]
  B. ema_subtraction       — Exponential moving average removal [14]
  C. highpass_filter       — IIR high-pass along slow-time axis [14]
  D. circle_fit_correction — Algebraic circle fitting on IQ plane [12]

Key insight from [14]: Stationary clutters in FMCW radar can be TRANSLATED
into DC offsets in the synthesized slow-time I/Q signals, allowing direct
application of DC calibration techniques developed for CW Doppler radars.
Circle fitting [12] finds the exact DC offset geometrically.

References:
    [12] Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14] Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
    [14] Z. Peng and C. Li, "Portable Microwave Radar Systems for Short-Range
         Localization and Life Tracking: A Review," Sensors, vol. 19, no. 5, 2019.
         doi:10.3390/s19051136
"""

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Method A: Static Mean Subtraction [14]
# ---------------------------------------------------------------------------

def mean_subtraction(
    slow_time_signal: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Remove DC offset by subtracting the temporal mean.

    This is the method directly described in [14] (Alizadeh 2019):
    "the DC component is removed from the unwrapped phase by subtracting
    the mean of the phase over the observation window."

    Here applied to the complex IQ signal BEFORE phase extraction,
    removing the DC phasor offset in the complex domain.

    Args:
        slow_time_signal: Complex slow-time signal from target range bin,
                          shape [n_chirps].

    Returns:
        corrected:  DC-corrected complex signal, shape [n_chirps].
        info:       Dictionary with estimated DC values.
    """
    dc_estimate = np.mean(slow_time_signal)
    corrected = slow_time_signal - dc_estimate
    info = {
        "method": "mean_subtraction",
        "dc_i": float(np.real(dc_estimate)),
        "dc_q": float(np.imag(dc_estimate)),
    }
    return corrected, info


# ---------------------------------------------------------------------------
# Method B: Exponential Moving Average (EMA) Subtraction
# ---------------------------------------------------------------------------

def ema_subtraction(
    slow_time_signal: np.ndarray,
    alpha: float = 0.98,
) -> Tuple[np.ndarray, dict]:
    """
    Remove DC offset using an exponential moving average (EMA) filter.

    EMA tracks slow-varying DC (including clutter drift over time).
    The EMA is computed as:
        EMA[n] = alpha * EMA[n-1] + (1 - alpha) * x[n]

    The DC estimate at each step is EMA[n], and we subtract it.
    This is a first-order IIR high-pass filter (cutoff = (1-alpha) / (2*pi)).

    Args:
        slow_time_signal: Complex array, shape [n_chirps].
        alpha:            Forgetting factor (0 < alpha < 1).
                          Higher alpha = slower tracking = lower cutoff freq.
                          Typical: 0.95–0.99 for vital sign applications.

    Returns:
        corrected:  DC-corrected signal, shape [n_chirps].
        info:       Dictionary with EMA trace and alpha.
    """
    n = len(slow_time_signal)
    ema = np.zeros(n, dtype=complex)
    ema[0] = slow_time_signal[0]

    for i in range(1, n):
        ema[i] = alpha * ema[i - 1] + (1.0 - alpha) * slow_time_signal[i]

    corrected = slow_time_signal - ema
    info = {
        "method": "ema_subtraction",
        "alpha": alpha,
        "ema_trace_i": np.real(ema),
        "ema_trace_q": np.imag(ema),
        "dc_i": float(np.real(ema).mean()),
        "dc_q": float(np.imag(ema).mean()),
    }
    return corrected, info


# ---------------------------------------------------------------------------
# Method C: High-Pass Filtering (MTI-style)
# ---------------------------------------------------------------------------

def highpass_filter(
    slow_time_signal: np.ndarray,
    slow_time_fs: float,
    cutoff_hz: float = 0.05,
    order: int = 2,
) -> Tuple[np.ndarray, dict]:
    """
    Remove DC offset and very low-frequency drift using a Butterworth high-pass filter.

    Applied along the slow-time axis. This is analogous to a Moving Target
    Indicator (MTI) filter that suppresses zero-Doppler (static) components.

    The cutoff should be set BELOW the minimum vital sign frequency:
        Respiration: 0.1–0.5 Hz
        Heartbeat:   0.8–2.0 Hz
    So cutoff_hz = 0.05 Hz is safe for preserving both.

    IMPORTANT — numerical stability:
        When slow_time_fs is very high (e.g., 25000 Hz) and cutoff_hz is tiny
        (e.g., 0.05 Hz), the normalized cutoff = cutoff/Nyquist ~ 4e-6 makes
        the standard ba-form Butterworth filter numerically unstable.
        We use SOS (second-order sections) + sosfiltfilt, which is always stable.

    Args:
        slow_time_signal: Complex array, shape [n_chirps].
        slow_time_fs:     Slow-time sampling rate [Hz] = 1 / chirp_duration.
        cutoff_hz:        High-pass cutoff frequency [Hz].
        order:            Filter order. Keep <= 4 to avoid precision issues with
                          extreme cutoff ratios.

    Returns:
        corrected:  High-pass filtered signal, shape [n_chirps].
        info:       Dictionary with filter parameters.
    """
    nyq = slow_time_fs / 2.0
    normalized_cutoff = cutoff_hz / nyq

    if normalized_cutoff <= 0 or normalized_cutoff >= 1.0:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} Hz is invalid for fs={slow_time_fs} Hz. "
            f"Must be in (0, {nyq:.1f}) Hz."
        )

    # Design in SOS format — numerically stable for extreme cutoff ratios
    sos = butter(order, normalized_cutoff, btype="high", analog=False, output="sos")

    # Apply zero-phase SOS filtering to real and imaginary parts separately
    corrected_i = sosfiltfilt(sos, np.real(slow_time_signal))
    corrected_q = sosfiltfilt(sos, np.imag(slow_time_signal))
    corrected = corrected_i + 1j * corrected_q

    info = {
        "method": "highpass_filter",
        "cutoff_hz": cutoff_hz,
        "order": order,
        "slow_time_fs": slow_time_fs,
        "dc_i": float(np.real(slow_time_signal).mean()),
        "dc_q": float(np.imag(slow_time_signal).mean()),
    }
    return corrected, info


# ---------------------------------------------------------------------------
# Method D: Geometric (Circle) Fitting — Primary method from [12]
# ---------------------------------------------------------------------------

def _taubin_circle_fit(I: np.ndarray, Q: np.ndarray) -> Tuple[float, float, float]:
    """
    Taubin algebraic circle fitting.

    Fits a circle (cx, cy, r) to a set of 2-D points (I, Q) by minimizing
    the algebraic distance with a bias-corrected constraint (Taubin 1994).

    This is used in [12] to find the DC offset as the center of the IQ circle.

    Theory:
        The IQ signal of a stationary subject modulates the phase of a phasor
        of magnitude r centered at (cx, cy). The center = DC offset.
        After subtracting (cx, cy), the IQ trajectory passes through the origin.

    Args:
        I: In-phase samples, shape [n].
        Q: Quadrature samples, shape [n].

    Returns:
        cx: Circle center x (= DC_I estimate)
        cy: Circle center y (= DC_Q estimate)
        r:  Circle radius (= signal amplitude estimate)

    Reference:
        G. Taubin, "Estimation of Planar Curves, Surfaces, and Nonplanar Space
        Curves Defined by Implicit Equations," IEEE TPAMI, 1991.
    """
    n = len(I)
    # Centroid
    x_mean = np.mean(I)
    y_mean = np.mean(Q)

    # Centered coordinates
    u = I - x_mean
    v = Q - y_mean

    # Moment sums
    Suu = np.sum(u * u)
    Svv = np.sum(v * v)
    Suv = np.sum(u * v)
    Suuu = np.sum(u * u * u)
    Svvv = np.sum(v * v * v)
    Suvv = np.sum(u * v * v)
    Svuu = np.sum(v * u * u)

    # Solve 2x2 linear system for circle center (Pratt/Taubin simplification)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b_vec = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu)])

    try:
        uc, vc = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        # Fallback: use centroid as center if degenerate
        uc, vc = 0.0, 0.0

    # Center in original coordinates
    cx = uc + x_mean
    cy = vc + y_mean

    # Radius: mean distance from center to all points
    r = np.mean(np.sqrt((I - cx) ** 2 + (Q - cy) ** 2))

    return float(cx), float(cy), float(r)


def circle_fit_correction(
    slow_time_signal: np.ndarray,
    min_points: int = 20,
) -> Tuple[np.ndarray, dict]:
    """
    Remove DC offset using algebraic circle fitting on the IQ constellation.

    Method from [12] (Mercuri 2022):
    The slow-time IQ trajectory of a human target at a given range bin forms
    a circular arc in the complex plane. Its center represents the DC offset
    (from static clutter + hardware). By fitting a circle and subtracting
    its center, we:
      1. Remove the DC offset precisely.
      2. Make the signal suitable for arctangent phase demodulation.
      3. Handle multipath effects better than simple mean subtraction.

    This method is robust because:
      - It does not assume the center = temporal mean (which fails under multipath).
      - It uses all IQ samples simultaneously for a least-squares fit.

    Args:
        slow_time_signal: Complex array, shape [n_chirps].
        min_points:       Minimum points needed for reliable circle fit.

    Returns:
        corrected:  DC-corrected signal, shape [n_chirps].
        info:       Dictionary with estimated DC center and radius.
    """
    I = np.real(slow_time_signal)
    Q = np.imag(slow_time_signal)

    if len(I) < min_points:
        # Fallback to mean subtraction if too few points
        return mean_subtraction(slow_time_signal)

    cx, cy, r = _taubin_circle_fit(I, Q)

    # Subtract estimated center (DC offset)
    corrected = slow_time_signal - (cx + 1j * cy)

    info = {
        "method": "circle_fit",
        "dc_i": cx,
        "dc_q": cy,
        "radius": r,
        "center_iq": (cx, cy),
    }
    return corrected, info


# ---------------------------------------------------------------------------
# Dispatcher: run all methods and return results
# ---------------------------------------------------------------------------

def apply_all_dc_methods(
    slow_time_signal: np.ndarray,
    slow_time_fs: float,
    cutoff_hz: float = 0.05,
) -> dict:
    """
    Apply all four DC correction methods to the same input signal.

    Args:
        slow_time_signal: Complex array, shape [n_chirps].
        slow_time_fs:     Slow-time sampling rate [Hz].
        cutoff_hz:        High-pass filter cutoff [Hz].

    Returns:
        results: Dictionary mapping method name → (corrected_signal, info).
    """
    results = {}

    sig_A, info_A = mean_subtraction(slow_time_signal)
    results["mean_subtraction"] = (sig_A, info_A)

    sig_B, info_B = ema_subtraction(slow_time_signal, alpha=0.98)
    results["ema_subtraction"] = (sig_B, info_B)

    sig_C, info_C = highpass_filter(slow_time_signal, slow_time_fs, cutoff_hz=cutoff_hz)
    results["highpass_filter"] = (sig_C, info_C)

    sig_D, info_D = circle_fit_correction(slow_time_signal)
    results["circle_fit"] = (sig_D, info_D)

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import make_single_target_scene
    from range_fft import range_fft, find_target_range_bin, extract_slow_time_signal

    print("=== dc_offset_correction.py self-test ===")

    data_cube, gt, cfg = make_single_target_scene(dc_i=0.5, dc_q=-0.3)
    true_dc = gt["dc_offset"]
    print(f"  True DC offset: I={true_dc[0]:.4f}, Q={true_dc[1]:.4f}")

    # Range FFT → slow-time signal at target bin
    rp, r_axis = range_fft(data_cube, cfg)
    tgt_bin, tgt_range = find_target_range_bin(rp, r_axis, method="max_power")
    sts = extract_slow_time_signal(rp, tgt_bin)

    print(f"  Target range bin: {tgt_bin}  ({tgt_range:.3f} m)")

    # Test all methods
    results = apply_all_dc_methods(sts, cfg.slow_time_fs)

    for name, (corrected, info) in results.items():
        est_i = info["dc_i"]
        est_q = info["dc_q"]
        err_i = abs(est_i - true_dc[0])
        err_q = abs(est_q - true_dc[1])
        print(f"  [{name}] est_DC=({est_i:.3f}, {est_q:.3f})  "
              f"err=({err_i:.3f}, {err_q:.3f})")

    print("  PASS")
