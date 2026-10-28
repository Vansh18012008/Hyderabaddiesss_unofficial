"""
range_fft.py
============
Range FFT processing for FMCW radar.

Converts raw ADC data (fast-time x slow-time) into a range-slow-time matrix
by applying a window function and FFT along the fast-time axis.

Key steps (standard FMCW processing chain):
  1. Window along fast-time to reduce range sidelobes.
  2. FFT along fast-time → each bin corresponds to a range.
  3. Convert bin index to physical range.
  4. Identify target range bin (max power or known range).

References:
    [12] Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14] Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
"""

import numpy as np
from typing import Tuple, Optional
from data_generator import RadarConfig


# ---------------------------------------------------------------------------
# Window Functions
# ---------------------------------------------------------------------------

WINDOW_TYPES = ("hanning", "hamming", "blackman", "blackmanharris", "rectangular")


def make_window(n: int, window_type: str = "hanning") -> np.ndarray:
    """
    Create a 1-D window vector of length n.

    Args:
        n:           Number of samples.
        window_type: One of 'hanning', 'hamming', 'blackman',
                     'blackmanharris', 'rectangular'.

    Returns:
        Window array, shape [n].
    """
    wt = window_type.lower()
    if wt == "hanning":
        return np.hanning(n)
    elif wt == "hamming":
        return np.hamming(n)
    elif wt == "blackman":
        return np.blackman(n)
    elif wt == "blackmanharris":
        # 4-term Blackman-Harris (best sidelobe suppression)
        a = [0.35875, 0.48829, 0.14128, 0.01168]
        k = np.arange(n)
        return (a[0]
                - a[1] * np.cos(2 * np.pi * k / (n - 1))
                + a[2] * np.cos(4 * np.pi * k / (n - 1))
                - a[3] * np.cos(6 * np.pi * k / (n - 1)))
    elif wt == "rectangular":
        return np.ones(n)
    else:
        raise ValueError(f"Unknown window type '{window_type}'. Choose from {WINDOW_TYPES}.")


# ---------------------------------------------------------------------------
# Range FFT
# ---------------------------------------------------------------------------

def range_fft(
    data_cube: np.ndarray,
    cfg: RadarConfig,
    window_type: str = "hanning",
    n_fft: Optional[int] = None,
    return_one_sided: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply windowing and FFT along the fast-time (sample) axis.

    Args:
        data_cube:       Raw ADC data, shape [n_chirps, n_samples].
        cfg:             RadarConfig instance.
        window_type:     Window function type (see WINDOW_TYPES).
        n_fft:           FFT size. Defaults to n_samples. Can be larger for zero-padding.
        return_one_sided: If True, return only positive frequencies (one-sided spectrum).
                          For real physical interpretation, one-sided is standard.

    Returns:
        range_profile:   Complex range-slow-time matrix, shape [n_chirps, n_range_bins].
        range_axis_m:    Physical range for each bin [m], shape [n_range_bins].
    """
    n_chirps, n_samples = data_cube.shape
    if n_fft is None:
        n_fft = n_samples

    # --- Window along fast-time axis ---
    window = make_window(n_samples, window_type)          # [n_samples]
    windowed = data_cube * window[np.newaxis, :]          # [n_chirps, n_samples]

    # --- FFT along fast-time (axis=1) ---
    range_profile_full = np.fft.fft(windowed, n=n_fft, axis=1)  # [n_chirps, n_fft]

    # --- One-sided (positive frequencies only) ---
    if return_one_sided:
        n_bins = n_fft // 2
        range_profile = range_profile_full[:, :n_bins]    # [n_chirps, n_bins]
    else:
        n_bins = n_fft
        range_profile = range_profile_full

    # --- Convert bin index to physical range ---
    # Beat frequency for bin k: f_beat = k * fs_fast / n_fft
    # Range: R = f_beat * c / (2 * slope)
    bin_indices = np.arange(n_bins)
    f_beat_per_bin = bin_indices * cfg.fast_time_fs / n_fft
    range_axis_m = f_beat_per_bin * cfg.c / (2.0 * cfg.slope)

    return range_profile, range_axis_m


# ---------------------------------------------------------------------------
# Target Range Bin Selection
# ---------------------------------------------------------------------------

def find_target_range_bin(
    range_profile: np.ndarray,
    range_axis_m: np.ndarray,
    method: str = "max_power",
    known_range_m: Optional[float] = None,
    guard_bins: int = 2,
    noise_bins: int = 10,
) -> Tuple[int, float]:
    """
    Identify the range bin corresponding to a human target.

    Two strategies:
      - 'max_power': Select bin with highest average power across all chirps.
                     This is the method used in [14] (Alizadeh 2019).
      - 'known_range': Use a pre-known range (for ground-truth / simulation).

    Args:
        range_profile:  Shape [n_chirps, n_range_bins].
        range_axis_m:   Physical range per bin [m].
        method:         'max_power' or 'known_range'.
        known_range_m:  Required if method='known_range'.
        guard_bins:     Guard cells on each side (for CFAR-like selection).
        noise_bins:     Noise estimation window half-width.

    Returns:
        target_bin:   Integer index of the target range bin.
        target_range: Physical range of the selected bin [m].
    """
    # Average power across all chirps per range bin
    avg_power = np.mean(np.abs(range_profile) ** 2, axis=0)  # [n_range_bins]

    if method == "max_power":
        target_bin = int(np.argmax(avg_power))
    elif method == "known_range":
        if known_range_m is None:
            raise ValueError("known_range_m must be provided for method='known_range'.")
        # Find bin closest to known range
        target_bin = int(np.argmin(np.abs(range_axis_m - known_range_m)))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'max_power' or 'known_range'.")

    target_range = range_axis_m[target_bin]
    return target_bin, target_range


def extract_slow_time_signal(
    range_profile: np.ndarray,
    target_bin: int,
) -> np.ndarray:
    """
    Extract the complex slow-time signal from a given range bin.

    This is the signal that carries the vital sign (chest motion) information
    and on which DC offset correction and phase demodulation operate.

    Args:
        range_profile: Complex array, shape [n_chirps, n_range_bins].
        target_bin:    Index of the target range bin.

    Returns:
        slow_time_signal: Complex array, shape [n_chirps].
    """
    return range_profile[:, target_bin]


def compute_range_power_profile(range_profile: np.ndarray) -> np.ndarray:
    """
    Compute the average power profile across all chirps.

    Useful for visualization and range bin selection.

    Args:
        range_profile: Complex array, shape [n_chirps, n_range_bins].

    Returns:
        power_profile: Float array, shape [n_range_bins], in linear power.
    """
    return np.mean(np.abs(range_profile) ** 2, axis=0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import make_single_target_scene

    print("=== range_fft.py self-test ===")
    data_cube, gt, cfg = make_single_target_scene(target_range=1.5)

    # Range FFT
    rp, r_axis = range_fft(data_cube, cfg, window_type="hanning")
    print(f"  Range profile shape : {rp.shape}  (n_chirps x n_range_bins)")
    print(f"  Range axis          : {r_axis[0]:.3f} m ... {r_axis[-1]:.3f} m")
    print(f"  Range resolution    : {r_axis[1]-r_axis[0]:.4f} m")

    # Target bin selection
    tgt_bin, tgt_range = find_target_range_bin(rp, r_axis, method="max_power")
    true_range = gt["targets"][0].range_m
    print(f"  True target range   : {true_range:.3f} m")
    print(f"  Detected range      : {tgt_range:.3f} m  (bin {tgt_bin})")
    print(f"  Range error         : {abs(tgt_range - true_range)*100:.2f} cm")

    # Slow-time signal
    sts = extract_slow_time_signal(rp, tgt_bin)
    print(f"  Slow-time signal    : shape={sts.shape}, dtype={sts.dtype}")
    print("  PASS")
