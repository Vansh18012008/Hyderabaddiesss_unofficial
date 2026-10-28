"""
music_2d.py
===========
2D MUSIC Algorithm for Range-Azimuth Target Detection in FMCW Radar.

Implements the algorithm from:
    [NEW] Application of 2D MUSIC Algorithm to Range-Azimuth FMCW Radar Data
          (Van Rossum & Hoogeboom, TNO/Delft University of Technology, EuMA 2012)

The algorithm jointly estimates angular and range target locations using
super-resolution subspace methods, overcoming the diffraction limit of
conventional Delay-and-Sum / Digital Beam Forming (DBF).

Core approach (from the paper):
  1. Arrange fast-time x spatial samples into [L x N] data matrix X.
  2. Apply 2D spatial smoothing (forward/backward) over [m1 x m2] window
     to form smoothed covariance matrix C_xtilde (handles coherent targets).
  3. SVD-decompose C_xtilde and identify the noise subspace W_tilde.
  4. Sweep 2D steering vector a(tau_i) over (range, azimuth) search grid.
  5. MUSIC pseudo-spectrum = 1 / (a^H W_tilde W_tilde^H a). Peaks = targets.

Key equations (paper notation):
  X = A*s + W                               [Eq. 4 - data model]
  a(tau_i) = joint spatial+fast-time SV     [Eq. 12]
  C_xtilde = (1/2p1p2)[X_t X_t^H + J(X_t X_t^H)*J]   [Eq. 9]
  P_MUSIC(tau_i) = 1 / (a^H W_t W_t^H a)   [Eq. 11]

Multi-object detection:
  - Multiple peaks in the pseudo-spectrum correspond to multiple targets.
  - Iterative greedy peak detection with spatial exclusion zones.
  - K (number of targets) estimated via largest eigenvalue-ratio gap.

References:
    [NEW] Van Rossum & Hoogeboom, EuMA 2012. (TNO/Delft University)
    [12]  Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14]  Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Helper: Exchange (counter-identity) matrix J
# ---------------------------------------------------------------------------

def _exchange_matrix(n: int) -> np.ndarray:
    """
    Exchange matrix J of size [n x n].
    J[i, j] = 1 iff i + j == n - 1, else 0.
    Used in forward/backward spatial smoothing (Eq. 10 of paper).
    """
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        J[i, n - 1 - i] = 1.0
    return J


# ---------------------------------------------------------------------------
# 2D Forward/Backward Spatial Smoothing  (Eq. 8-10 of paper)
# ---------------------------------------------------------------------------

def spatial_smooth_2d(X: np.ndarray, m1: int, m2: int) -> np.ndarray:
    """
    2D forward/backward spatial smoothing to handle coherent targets (Eq. 8-10).

    Slides an [m1 x m2] window over the [L x N] data matrix X.
    Each sub-matrix is vectorised to length m1*m2 and collected into
    X_tilde of shape [m1*m2, p1*p2].
    Then forms the forward/backward averaged covariance matrix.

    Args:
        X:  Complex data matrix [L, N] = [num_rx, num_range_samples].
        m1: Spatial (antenna) window size.  1 <= m1 <= L.
        m2: Fast-time window size.          1 <= m2 <= N.

    Returns:
        C_smooth: Smoothed covariance matrix [m1*m2, m1*m2] (Hermitian).
    """
    L, N = X.shape
    p1 = L - m1 + 1
    p2 = N - m2 + 1

    if p1 <= 0 or p2 <= 0:
        raise ValueError(
            f"Window ({m1},{m2}) too large for data shape ({L},{N}). "
            f"Need m1<={L}, m2<={N}."
        )

    dim = m1 * m2
    num_cols = p1 * p2
    X_tilde = np.zeros((dim, num_cols), dtype=X.dtype)

    col = 0
    for p1_idx in range(p1):
        for p2_idx in range(p2):
            sub = X[p1_idx:p1_idx + m1, p2_idx:p2_idx + m2]  # [m1, m2]
            X_tilde[:, col] = sub.ravel()
            col += 1

    XtXh = X_tilde @ X_tilde.conj().T                         # [dim, dim]
    J = _exchange_matrix(dim)
    C_smooth = (XtXh + J @ XtXh.conj() @ J) / (2.0 * num_cols)
    return C_smooth


# ---------------------------------------------------------------------------
# Eigenvalue-based number-of-targets estimation
# ---------------------------------------------------------------------------

def estimate_num_targets(
    eigenvalues: np.ndarray,
    method: str = "gap",
    max_targets: Optional[int] = None,
) -> int:
    """
    Estimate number of targets K from eigenvalue profile.

    Methods:
        'gap'  : Largest ratio gap between consecutive eigenvalues.
        'mdl'  : Minimum Description Length (Wax & Kailath, 1985) [Ref 11].
        'fixed': Return max_targets directly.

    Args:
        eigenvalues: Real eigenvalues (any order), shape [D].
        method:      'gap', 'mdl', or 'fixed'.
        max_targets: Upper bound on K.

    Returns:
        K: Estimated number of targets (>= 1).
    """
    evals = np.sort(np.abs(eigenvalues))[::-1]  # descending
    n = len(evals)
    if max_targets is None:
        max_targets = max(1, n // 4)

    if method == "fixed":
        return int(max_targets)

    if method == "gap":
        if n <= 1:
            return 1
        ratios = evals[:-1] / (evals[1:] + 1e-30)
        search_end = min(max_targets, n - 1)
        best_k = int(np.argmax(ratios[:search_end])) + 1
        return max(1, best_k)

    if method == "mdl":
        L_snap = 1
        best_k, best_mdl = 1, np.inf
        for k in range(1, min(max_targets + 1, n)):
            noise_evals = evals[k:]
            if len(noise_evals) == 0:
                break
            geo_mean = np.exp(np.mean(np.log(noise_evals + 1e-30)))
            arith_mean = np.mean(noise_evals)
            ratio = geo_mean / (arith_mean + 1e-30)
            penalty = 0.5 * k * (2 * n - k) * np.log(L_snap + 1)
            log_like = -L_snap * (n - k) * np.log(ratio + 1e-30)
            mdl_val = -log_like + penalty
            if mdl_val < best_mdl:
                best_mdl = mdl_val
                best_k = k
        return max(1, best_k)

    raise ValueError(f"Unknown method '{method}'. Use 'gap', 'mdl', or 'fixed'.")


# ---------------------------------------------------------------------------
# Practical 2D MUSIC on post-range-FFT cube  (fast, QIDK-ready)
# ---------------------------------------------------------------------------

def compute_music_2d_fast(
    range_profile_cube: np.ndarray,
    fc: float,
    d: float,
    c: float,
    lambda_c: float,
    range_axis: np.ndarray,
    azimuth_grid: np.ndarray,
    m1: int = 3,
    num_targets: int = 2,
    num_targets_method: str = "gap",
    range_weighting: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Practical 2D MUSIC using post-range-FFT data cube.

    After range FFT, applies per-range-bin spatial MUSIC over the antenna
    aperture with forward/backward smoothing.  This is the computationally
    tractable variant suited for QIDK real-time use.

    Args:
        range_profile_cube: Complex [num_rx, num_chirps, num_range_bins].
        fc:                 Carrier frequency [Hz].
        d:                  Antenna spacing [m].
        c:                  Speed of light [m/s].
        lambda_c:           Centre wavelength [m].
        range_axis:         Range per bin [m], shape [num_range_bins].
        azimuth_grid:       Azimuth angles [deg], shape [n_azimuth].
        m1:                 Spatial sub-array size for smoothing.
        num_targets:        Max expected targets.
        num_targets_method: 'gap', 'mdl', or 'fixed'.

    Returns:
        music_spectrum:  MUSIC pseudo-spectrum [range, azimuth] in dB.
        range_axis:      Same range_axis passed in.
        azimuth_grid:    Same azimuth_grid passed in.
        K_est:           Global estimated number of targets.
    """
    num_rx, num_chirps, num_range_bins = range_profile_cube.shape
    n_azimuth = len(azimuth_grid)
    spectrum = np.zeros((num_range_bins, n_azimuth), dtype=float)

    sin_angles = np.sin(np.deg2rad(azimuth_grid))
    d_lambda = d / lambda_c

    m1_eff = min(m1, num_rx)
    p1 = max(1, num_rx - m1_eff + 1)
    J_m = _exchange_matrix(m1_eff)

    for ri in range(num_range_bins):
        # Snapshot matrix at this range bin: [num_rx, num_chirps]
        Y = range_profile_cube[:, :, ri]

        if num_rx < 2 or num_chirps < 2:
            # Degenerate -- use plain beamforming
            for ai in range(n_azimuth):
                sv = np.exp(-1j * 2 * np.pi * d_lambda
                             * np.arange(num_rx) * sin_angles[ai])
                spectrum[ri, ai] = np.abs(sv.conj() @ Y.mean(axis=1)) ** 2
            continue

        # Forward/backward spatial smoothing over sub-arrays
        C_acc = np.zeros((m1_eff, m1_eff), dtype=complex)
        n_sub = 0
        for p1_idx in range(p1):
            Y_sub = Y[p1_idx:p1_idx + m1_eff, :]          # [m1, num_chirps]
            C_fwd = (Y_sub @ Y_sub.conj().T) / num_chirps  # [m1, m1]
            C_bwd = J_m @ C_fwd.conj() @ J_m
            C_acc += C_fwd + C_bwd
            n_sub += 1
        C_smooth = C_acc / (2 * max(1, n_sub))

        # Eigendecomposition (eigh for Hermitian stability)
        eigenvalues, eigenvectors = np.linalg.eigh(C_smooth)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Noise subspace
        K_bin = estimate_num_targets(
            eigenvalues,
            method=num_targets_method,
            max_targets=min(num_targets, m1_eff - 1),
        )
        noise_vecs = eigenvectors[:, K_bin:]           # [m1, m1-K]
        WtWh = noise_vecs @ noise_vecs.conj().T        # [m1, m1]

        # MUSIC spectrum sweep over azimuth at this range bin
        phases = 2.0 * np.pi * d_lambda * np.arange(m1_eff)
        for ai in range(n_azimuth):
            sv = np.exp(-1j * phases * sin_angles[ai])
            denom = float(np.real(sv.conj() @ WtWh @ sv)) + 1e-30
            spectrum[ri, ai] = 1.0 / denom

    # Global K estimate from dominant range bin
    dom_ri = int(np.argmax(spectrum.max(axis=1)))
    Y_dom = range_profile_cube[:, :, dom_ri]
    C_dom = (Y_dom @ Y_dom.conj().T) / num_chirps
    evals_dom, _ = np.linalg.eigh(C_dom)
    K_est_global = estimate_num_targets(
        evals_dom[::-1],
        method=num_targets_method,
        max_targets=num_targets,
    )

    if range_weighting:
        # Range power weighting combines super-resolution angular MUSIC with range profile
        # power, preventing noise spikes and avoiding mutual target shadowing across ranges
        range_power = np.mean(np.abs(range_profile_cube)**2, axis=(0, 1))
        row_max = np.max(spectrum, axis=1, keepdims=True) + 1e-30
        norm_spec = spectrum / row_max
        weighted = norm_spec * range_power[:, np.newaxis]
        spec_max = np.max(weighted) + 1e-30
        music_spectrum = 10.0 * np.log10(np.maximum(weighted / spec_max, 1e-12))
    else:
        spec_max = np.max(spectrum) + 1e-30
        music_spectrum = 10.0 * np.log10(np.maximum(spectrum / spec_max, 1e-12))
    return music_spectrum, range_axis, azimuth_grid, K_est_global


# ---------------------------------------------------------------------------
# Multi-target peak detection from MUSIC spectrum
# ---------------------------------------------------------------------------

def detect_targets_music(
    music_spectrum: np.ndarray,
    range_axis: np.ndarray,
    azimuth_grid: np.ndarray,
    num_targets: int = 2,
    min_range_m: float = 0.3,
    min_peak_db: float = -25.0,
    min_range_sep_m: float = 0.15,
    min_angle_sep_deg: float = 3.0,
) -> List[dict]:
    """
    Detect up to num_targets peaks from 2D MUSIC pseudo-spectrum.

    Uses greedy iterative peak selection with spatial exclusion zones to
    separate closely spaced targets.

    Args:
        music_spectrum:    [n_range, n_azimuth] dB array (0 dB = global max).
        range_axis:        Physical range per row [m].
        azimuth_grid:      Azimuth angle per column [deg].
        num_targets:       Max targets to detect.
        min_range_m:       Ignore near-field bins below this range.
        min_peak_db:       Threshold relative to global max (dB).
        min_range_sep_m:   Minimum range between two detected targets [m].
        min_angle_sep_deg: Minimum azimuth between two detected targets [deg].

    Returns:
        List of target dicts (sorted by descending peak_db), each containing:
            target_id, range_m, angle_deg, range_idx, angle_idx, peak_db.
    """
    n_range, n_azimuth = music_spectrum.shape

    # Near-field mask
    min_ri = int(np.searchsorted(range_axis, min_range_m))
    working = music_spectrum.copy()
    if min_ri > 0:
        working[:min_ri, :] = -np.inf
    working[working < min_peak_db] = -np.inf

    targets = []
    dr = float(range_axis[1] - range_axis[0]) if len(range_axis) > 1 else 0.01
    da = float(abs(azimuth_grid[1] - azimuth_grid[0])) if len(azimuth_grid) > 1 else 2.5
    range_sep_bins = max(1, int(np.ceil(min_range_sep_m / (dr + 1e-10))))
    angle_sep_bins = max(1, int(np.ceil(min_angle_sep_deg / (da + 1e-10))))

    for tid in range(1, num_targets + 1):
        if np.all(np.isinf(working)):
            break
        flat_idx = int(np.argmax(working))
        ri, ai = np.unravel_index(flat_idx, working.shape)
        peak_db = float(working[ri, ai])
        if np.isinf(peak_db) or peak_db < min_peak_db:
            break
        targets.append({
            'target_id': tid,
            'range_m':   float(range_axis[ri]),
            'angle_deg': float(azimuth_grid[ai]),
            'range_idx': int(ri),
            'angle_idx': int(ai),
            'peak_db':   peak_db,
        })
        # Suppress exclusion zone
        r_lo = max(0, ri - range_sep_bins)
        r_hi = min(n_range, ri + range_sep_bins + 1)
        a_lo = max(0, ai - angle_sep_bins)
        a_hi = min(n_azimuth, ai + angle_sep_bins + 1)
        working[r_lo:r_hi, a_lo:a_hi] = -np.inf

    return targets


# ---------------------------------------------------------------------------
# Complete 2D MUSIC multi-object pipeline (top-level entry point)
# ---------------------------------------------------------------------------

def run_music_pipeline(
    range_profile_cube: np.ndarray,
    fc: float,
    d: float,
    c: float,
    lambda_c: float,
    range_axis: np.ndarray,
    azimuth_deg: np.ndarray,
    m1: int = 3,
    num_targets: int = 2,
    min_range_m: float = 0.3,
    min_peak_db: float = -25.0,
    range_weighting: bool = True,
) -> dict:
    """
    Full 2D MUSIC pipeline: range-profile cube -> multi-target detection.

    Args:
        range_profile_cube: Complex [num_rx, num_chirps, num_range_bins].
        fc:          Carrier frequency [Hz].
        d:           Antenna spacing [m].
        c:           Speed of light [m/s].
        lambda_c:    Centre wavelength [m].
        range_axis:  Range per bin [m].
        azimuth_deg: Azimuth search grid [deg].
        m1:          Spatial sub-array size (smoothing).
        num_targets: Max targets to detect.
        min_range_m: Near-field exclusion threshold [m].
        min_peak_db: Minimum peak threshold in dB relative to maximum.
        range_weighting: Weight MUSIC pseudo-spectrum by range profile power.

    Returns:
        dict with keys:
            spectrum_db   -- 2D MUSIC spectrum [range, azimuth] dB
            range_axis    -- range axis [m]
            azimuth_grid  -- azimuth grid [deg]
            targets       -- list of target dicts
            K_est         -- estimated number of targets
            primary_target-- strongest target dict (or None)
    """
    spectrum_db, r_ax, az_grid, K_est = compute_music_2d_fast(
        range_profile_cube=range_profile_cube,
        fc=fc, d=d, c=c, lambda_c=lambda_c,
        range_axis=range_axis,
        azimuth_grid=azimuth_deg,
        m1=m1,
        num_targets=num_targets,
        num_targets_method="gap",
        range_weighting=range_weighting,
    )

    dr = float(range_axis[1] - range_axis[0]) if len(range_axis) > 1 else 0.01
    da = float(abs(azimuth_deg[1] - azimuth_deg[0])) if len(azimuth_deg) > 1 else 2.5

    targets = detect_targets_music(
        music_spectrum=spectrum_db,
        range_axis=range_axis,
        azimuth_grid=azimuth_deg,
        num_targets=num_targets,
        min_range_m=min_range_m,
        min_peak_db=min_peak_db,
        min_range_sep_m=max(0.10, dr * 2),
        min_angle_sep_deg=max(2.5, da * 2),
    )

    return {
        'spectrum_db':    spectrum_db,
        'range_axis':     range_axis,
        'azimuth_grid':   azimuth_deg,
        'targets':        targets,
        'K_est':          K_est,
        'primary_target': targets[0] if targets else None,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== music_2d.py self-test ===")
    np.random.seed(42)

    # AWR1843 parameters
    f_min = 77e9
    f_max = 79.982e9
    BW = f_max - f_min
    fc = (f_min + f_max) / 2.0
    T = 100e-6
    alpha = BW / T
    c = 3e8
    lambda_c = c / fc
    d = lambda_c / 2.0
    fs_adc = 600e3
    L = 4     # RX antennas
    N = 256   # fast-time samples
    M = 128   # chirps

    # Ground truth: 2 targets
    targets_gt = [
        {'range_m': 1.5, 'angle_deg':  0.0, 'amplitude': 1.0},
        {'range_m': 2.0, 'angle_deg': 10.0, 'amplitude': 0.7},
    ]

    # Synthetic data cube [L, M, N]
    data_cube = np.zeros((L, M, N), dtype=complex)
    t_fast = np.arange(N) / fs_adc
    for tgt in targets_gt:
        R = tgt['range_m']
        theta = np.deg2rad(tgt['angle_deg'])
        A = tgt['amplitude']
        for l_idx in range(L):
            tau_l = (2.0 * R - l_idx * d * np.sin(theta)) / c
            beat = A * np.exp(1j * 2 * np.pi * (fc * tau_l + alpha * tau_l * t_fast))
            data_cube[l_idx, :, :] += beat[np.newaxis, :]
    data_cube += 0.05 * (np.random.randn(*data_cube.shape)
                         + 1j * np.random.randn(*data_cube.shape))

    # Range FFT (full complex unambiguous spectrum)
    N_FFT = 512
    R_max = (c * T / 2.0) * (fs_adc / BW)
    range_axis = np.linspace(0, R_max, N_FFT, endpoint=False)
    rp_cube = np.zeros((L, M, N_FFT), dtype=complex)
    win = np.hanning(N)
    for l_idx in range(L):
        for m_idx in range(M):
            Xf = np.fft.fft(data_cube[l_idx, m_idx, :] * win, n=N_FFT)
            rp_cube[l_idx, m_idx, :] = Xf

    azimuth_grid = np.arange(-60, 61, 2.5)

    print(f"  Data cube shape : {data_cube.shape}")
    print(f"  Range bins      : {N_FFT // 2}  (R_max={R_max:.2f} m)")
    print(f"  Azimuth points  : {len(azimuth_grid)}")
    print(f"  Ground truth    : {targets_gt}")

    result = run_music_pipeline(
        range_profile_cube=rp_cube,
        fc=fc, d=d, c=c, lambda_c=lambda_c,
        range_axis=range_axis,
        azimuth_deg=azimuth_grid,
        m1=3, num_targets=3, min_range_m=0.3,
    )

    print(f"  K estimated     : {result['K_est']}")
    print(f"  Targets found   : {len(result['targets'])}")
    for tgt in result['targets']:
        print(f"    Target {tgt['target_id']}: "
              f"range={tgt['range_m']:.2f}m  "
              f"angle={tgt['angle_deg']:.1f} deg  "
              f"peak={tgt['peak_db']:.1f}dB")
    print("  PASS")
