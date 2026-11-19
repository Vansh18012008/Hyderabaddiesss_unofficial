"""Joint range/azimuth MUSIC on complex FMCW ADC data (rx, chirp, sample).

Uses a separable far-field, narrowband receive-array model, sliding spatial
AND fast-time subarrays, and optional forward/backward covariance averaging.
The angular term is 2*pi*p*sin(theta)/lambda for a fixed transmitter: only
the target range has the round-trip factor two. This corrects the old model
which also doubled the receiver spacing. Historical maps must be regenerated.

The slope is bandwidth_hz / chirp_duration_s. These two values must describe
the same sweep interval; range calibration cannot be inferred from file size.
angular_sign defines the array's coordinate convention, not a fitted correction.
"""

import numpy as np


def _positive(**values):
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def steering_matrix(ranges_m, angles_deg, n_rx, n_fast, carrier_hz,
                    bandwidth_hz, chirp_duration_s, sample_rate_hz,
                    antenna_spacing_m, c=3e8, *, angular_sign=1):
    """Rows: unit-magnitude vectors in angle-major/range-minor order.

    Inside each row: receiver-major/fast-sample-minor, matching the covariance.
    Carrier is the effective sampled RF frequency. Target-constant phase is
    omitted because it cancels in the MUSIC projection.
    """
    _positive(carrier_hz=carrier_hz, bandwidth_hz=bandwidth_hz,
              chirp_duration_s=chirp_duration_s, sample_rate_hz=sample_rate_hz,
              antenna_spacing_m=antenna_spacing_m, c=c)
    if angular_sign not in (-1, 1):
        raise ValueError("angular_sign must be +1 or -1")
    if n_rx < 1 or n_fast < 1:
        raise ValueError("steering dimensions must be positive")
    ranges = np.asarray(ranges_m, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    if ranges.ndim != 1 or angles.ndim != 1 or not ranges.size or not angles.size:
        raise ValueError("range and angle axes must be nonempty vectors")
    if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(angles)):
        raise ValueError("scan axes must be finite")
    if np.any(ranges < 0) or np.any(np.abs(angles) > 90):
        raise ValueError("ranges must be nonnegative and angles within +/-90 degrees")
    slope = bandwidth_hz / chirp_duration_s
    beat = 2 * slope * ranges / c
    if np.any(beat >= sample_rate_hz):
        raise ValueError("scan exceeds complex-ADC unambiguous beat-frequency interval")
    fast = np.exp(2j * np.pi * beat[:, None] * np.arange(n_fast) / sample_rate_hz)
    spatial = np.exp(angular_sign * 2j * np.pi * carrier_hz / c
                     * np.sin(np.deg2rad(angles))[:, None]
                     * np.arange(n_rx) * antenna_spacing_m)
    return (spatial[:, None, :, None] * fast[None, :, None, :]).reshape(
        angles.size * ranges.size, n_rx * n_fast)


def _steering_vector(range_m, angle_deg, n_rx, n_fast, carrier_hz,
                     bandwidth_hz, chirp_duration_s, sample_rate_hz,
                     antenna_spacing_m, c=3e8, *, angular_sign=1):
    return steering_matrix([range_m], [angle_deg], n_rx, n_fast, carrier_hz,
                           bandwidth_hz, chirp_duration_s, sample_rate_hz,
                           antenna_spacing_m, c, angular_sign=angular_sign)[0]


def smoothed_covariance(data, spatial_window=4, fast_window=16, chirp_step=4,
                        fast_step=4, forward_backward=True):
    """Average overlapping spatial/fast-time snapshots from selected chirps.

    A shorter spatial window slides over EVERY valid receiver start. The old
    implementation only used the first spatial window. Overlapping snapshots
    are correlated; their count is not an independent sample count for MDL.
    """
    data = np.asarray(data)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("data must be complex with shape (rx, chirp, sample)")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains nonfinite values")
    n_rx, n_chirps, n_samples = data.shape
    if not (1 <= spatial_window <= n_rx and 1 <= fast_window <= n_samples):
        raise ValueError("smoothing windows must fit the data dimensions")
    if chirp_step < 1 or fast_step < 1 or not n_chirps:
        raise ValueError("steps and chirp count must be positive")
    snapshots = []
    for chirp in range(0, n_chirps, chirp_step):
        for rx in range(n_rx - spatial_window + 1):
            for start in range(0, n_samples - fast_window + 1, fast_step):
                snapshots.append(data[rx:rx + spatial_window, chirp,
                                      start:start + fast_window].reshape(-1))
    x = np.asarray(snapshots, dtype=np.complex128)
    covariance = x.T @ x.conj() / len(x)
    if forward_backward:
        covariance = (covariance + covariance[::-1, ::-1].conj()) / 2
    return (covariance + covariance.conj().T) / 2


def spectrum_from_steering(covariance, steering, grid_shape, n_targets,
                           chunk_size=4096):
    """MUSIC using ||E_n^H a||^2 = ||a||^2 - ||E_s^H a||^2.

    This is the same noise-subspace MUSIC spectrum, evaluated efficiently when
    the chosen signal rank is small. The floor bounds ideal/noiseless peaks.
    """
    covariance = np.asarray(covariance)
    dimension = steering.shape[1]
    if covariance.shape != (dimension, dimension):
        raise ValueError("covariance and steering dimensions do not agree")
    if not 1 <= n_targets < dimension:
        raise ValueError("n_targets must be between 1 and covariance dimension - 1")
    if not np.all(np.isfinite(covariance)) or not np.any(np.abs(covariance) > 0):
        raise ValueError("covariance must be finite and nonzero")
    if np.prod(grid_shape) != len(steering):
        raise ValueError("grid shape does not match steering matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    signal_basis = eigenvectors[:, -n_targets:]
    values = np.empty(len(steering), dtype=float)
    for start in range(0, len(steering), chunk_size):
        a = steering[start:start + chunk_size]
        norm = np.sum(np.abs(a) ** 2, axis=1)
        projection = a @ signal_basis.conj()
        denominator = norm - np.sum(np.abs(projection) ** 2, axis=1)
        values[start:start + chunk_size] = 1 / np.maximum(denominator, norm * 1e-12)
    return values.reshape(grid_shape), eigenvalues[::-1]


def music_spectrum(covariance, ranges_m, angles_deg, n_targets, carrier_hz,
                   bandwidth_hz, chirp_duration_s, sample_rate_hz,
                   antenna_spacing_m, n_rx, fast_window, c=3e8, *, angular_sign=1):
    steering = steering_matrix(ranges_m, angles_deg, n_rx, fast_window, carrier_hz,
                               bandwidth_hz, chirp_duration_s, sample_rate_hz,
                               antenna_spacing_m, c, angular_sign=angular_sign)
    return spectrum_from_steering(covariance, steering,
                                  (len(angles_deg), len(ranges_m)), n_targets)[0]


def estimate_range_azimuth(data, ranges_m, angles_deg, n_targets, carrier_hz,
                           bandwidth_hz, chirp_duration_s, sample_rate_hz,
                           antenna_spacing_m, spatial_window=4, fast_window=16,
                           chirp_step=4, fast_step=4, *, angular_sign=1) -> dict:
    """Return the map and strongest peak; use find_ra_peaks for multiple peaks."""
    covariance = smoothed_covariance(data, spatial_window, fast_window,
                                     chirp_step, fast_step)
    spectrum = music_spectrum(covariance, ranges_m, angles_deg, n_targets,
                              carrier_hz, bandwidth_hz, chirp_duration_s,
                              sample_rate_hz, antenna_spacing_m, spatial_window,
                              fast_window, angular_sign=angular_sign)
    a, r = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return dict(spectrum=spectrum, covariance=covariance,
                range_m=float(ranges_m[r]), angle_deg=float(angles_deg[a]),
                range_index=int(r), angle_index=int(a),
                eigenvalues=np.linalg.eigvalsh(covariance)[::-1])


def find_ra_peaks(spectrum, ranges_m, angles_deg, max_peaks=6,
                   relative_threshold_db=-8, min_range_separation_m=0.25,
                   min_angle_separation_deg=8, min_contrast_db=3):
    """Heuristic local maxima/NMS: candidate reflectors, not object labels.

    Suppression requires closeness in BOTH range and angle. Boundary maxima are
    excluded. Require contrast above the map median to avoid reporting tiny
    ripples in a flat spectrum. Neither threshold is calibrated CFAR.
    """
    from scipy.ndimage import maximum_filter
    p = np.asarray(spectrum)
    if p.shape != (len(angles_deg), len(ranges_m)) or np.any(p < 0):
        raise ValueError("spectrum must be nonnegative on the given grid")
    if not np.all(np.isfinite(p)) or p.max() <= 0:
        return []
    db = 10 * np.log10(np.maximum(p / p.max(), 1e-30))
    mask = (p == maximum_filter(p, size=3)) & (db >= relative_threshold_db)
    mask &= db >= np.median(db) + min_contrast_db
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    indices = sorted(np.argwhere(mask), key=lambda ar: p[tuple(ar)], reverse=True)
    peaks = []
    for a, r in indices:
        distance, angle = float(ranges_m[r]), float(angles_deg[a])
        if any(abs(distance - q['range_m']) < min_range_separation_m and
               abs(angle - q['angle_deg']) < min_angle_separation_deg for q in peaks):
            continue
        peaks.append(dict(range_m=distance, angle_deg=angle,
                          relative_db=float(db[a, r]), range_index=int(r),
                          angle_index=int(a)))
        if len(peaks) >= max_peaks:
            break
    return peaks
