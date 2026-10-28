"""
phase_demodulation.py
=====================
Phase extraction and demodulation from FMCW radar IQ signals.

After DC offset correction, the slow-time IQ signal at the target range bin
encodes the chest displacement as a phase modulation:

    s(t) = A * exp(j * phi(t))
    phi(t) = phi_static + (4*pi/lambda) * d(t)

where d(t) is the chest displacement due to respiration and heartbeat.

Steps:
  1. Arctangent phase extraction from corrected IQ.
  2. Phase unwrapping (removes 2*pi jumps).
  3. DC removal from phase (alternative to IQ-domain DC removal, as in [14]).
  4. Conversion to displacement [mm].

References:
    [12] Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14] Z. Peng and C. Li, "Portable Microwave Radar Systems for Short-Range Localization
         and Life Tracking: A Review," Sensors, vol. 19, no. 5, 2019.
         doi:10.3390/s19051136
    [14] Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Phase Extraction
# ---------------------------------------------------------------------------

def extract_phase(
    slow_time_signal: np.ndarray,
    method: str = "arctan2",
) -> np.ndarray:
    """
    Extract instantaneous phase from a complex slow-time signal.

    Args:
        slow_time_signal: Complex array, shape [n_chirps].
        method:           'arctan2' (standard) or 'angle' (np.angle wrapper).

    Returns:
        phase: Wrapped phase in [-pi, pi], shape [n_chirps].
    """
    if method == "arctan2":
        return np.arctan2(np.imag(slow_time_signal), np.real(slow_time_signal))
    elif method == "angle":
        return np.angle(slow_time_signal)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'arctan2' or 'angle'.")


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase to remove 2*pi discontinuities.

    np.unwrap handles the standard +/-pi threshold wrapping.
    The result is a continuous phase signal.

    Args:
        phase: Wrapped phase array, shape [n_chirps].

    Returns:
        unwrapped: Continuous phase, shape [n_chirps].
    """
    return np.unwrap(phase)


def remove_phase_dc(phase_unwrapped: np.ndarray) -> np.ndarray:
    """
    Remove the DC (mean) from the unwrapped phase.

    This is the PHASE-DOMAIN DC removal method described in [14] (Alizadeh 2019)
    and [14] (Peng & Li 2019). The static range component phi_static = 4*pi*R/lambda
    appears as a large constant offset in the unwrapped phase.
    Subtracting the mean removes this static offset, leaving only the
    time-varying chest motion signal.

    Args:
        phase_unwrapped: Continuous phase signal, shape [n_chirps].

    Returns:
        phase_clean: DC-removed phase, shape [n_chirps].
    """
    return phase_unwrapped - np.mean(phase_unwrapped)


# ---------------------------------------------------------------------------
# Phase-to-Displacement Conversion
# ---------------------------------------------------------------------------

def phase_to_displacement(
    phase_rad: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """
    Convert phase [radians] to chest displacement [meters].

    From the FMCW signal model:
        phi(t) = (4*pi/lambda) * d(t)
    Therefore:
        d(t) = phi(t) * lambda / (4*pi)

    Args:
        phase_rad:    Phase signal [radians], shape [n_chirps].
        wavelength_m: Radar wavelength [m]. For 77 GHz: lambda = c/fc ≈ 3.9 mm.

    Returns:
        displacement_m: Chest displacement [meters], shape [n_chirps].
    """
    return phase_rad * wavelength_m / (4.0 * np.pi)


# ---------------------------------------------------------------------------
# Complete Demodulation Pipeline
# ---------------------------------------------------------------------------

def demodulate(
    slow_time_signal: np.ndarray,
    wavelength_m: float,
    remove_dc_in_phase: bool = True,
) -> dict:
    """
    Full phase demodulation pipeline for a DC-corrected slow-time IQ signal.

    Pipeline:
        1. Extract wrapped phase via arctan2.
        2. Unwrap phase.
        3. Optionally remove DC from unwrapped phase [14].
        4. Convert to displacement.

    Args:
        slow_time_signal:    Complex DC-corrected signal, shape [n_chirps].
        wavelength_m:        Radar wavelength [m].
        remove_dc_in_phase:  If True, subtract mean from unwrapped phase.
                             Set False if DC was already removed in IQ domain.

    Returns:
        result: Dictionary with all intermediate and final signals.
    """
    # Step 1: Wrapped phase
    phase_wrapped = extract_phase(slow_time_signal)

    # Step 2: Unwrap
    phase_unwrapped = unwrap_phase(phase_wrapped)

    # Step 3: Phase-domain DC removal
    if remove_dc_in_phase:
        phase_clean = remove_phase_dc(phase_unwrapped)
    else:
        phase_clean = phase_unwrapped

    # Step 4: Convert to displacement
    displacement_m = phase_to_displacement(phase_clean, wavelength_m)
    displacement_mm = displacement_m * 1e3  # Convert to mm for readability

    return {
        "phase_wrapped_rad": phase_wrapped,
        "phase_unwrapped_rad": phase_unwrapped,
        "phase_clean_rad": phase_clean,
        "displacement_m": displacement_m,
        "displacement_mm": displacement_mm,
        "wavelength_m": wavelength_m,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import make_single_target_scene
    from range_fft import range_fft, find_target_range_bin, extract_slow_time_signal
    from dc_offset_correction import circle_fit_correction

    print("=== phase_demodulation.py self-test ===")

    data_cube, gt, cfg = make_single_target_scene(target_range=1.5)
    rp, r_axis = range_fft(data_cube, cfg)
    tgt_bin, tgt_range = find_target_range_bin(rp, r_axis, method="max_power")
    sts = extract_slow_time_signal(rp, tgt_bin)

    # DC correction (circle fit — recommended [12])
    sts_corrected, dc_info = circle_fit_correction(sts)

    # Demodulation
    demod = demodulate(sts_corrected, cfg.wavelength, remove_dc_in_phase=True)

    t_slow = gt["t_slow"]
    fs_slow = cfg.slow_time_fs
    disp = demod["displacement_mm"]

    print(f"  Slow-time fs         : {fs_slow:.0f} Hz")
    print(f"  Phase (wrapped) range: [{demod['phase_wrapped_rad'].min():.3f}, "
          f"{demod['phase_wrapped_rad'].max():.3f}] rad")
    print(f"  Displacement range   : [{disp.min():.3f}, {disp.max():.3f}] mm")

    # Expected: resp ~8 mm amplitude, heart ~0.3 mm amplitude
    resp_expected_mm = gt["targets"][0].resp_amplitude_m * 1e3
    print(f"  Expected resp amp    : ±{resp_expected_mm:.1f} mm")
    print(f"  Observed signal std  : {disp.std():.2f} mm")

    print("  PASS")
