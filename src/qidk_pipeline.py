"""
qidk_pipeline.py
================
FMCW Radar Vital-Signs Pipeline for QIDK (Qualcomm Intelligent Devices Kit).

This module integrates all three research-paper methods:
    [12] Mercuri 2022    -- Circle-fit DC correction, SVD clutter removal
    [14] Alizadeh 2019   -- Mean subtraction, phase demodulation, vital sign FFT
    [NEW] Van Rossum 2012 -- 2D MUSIC range-azimuth multi-object detection

Pipeline modes:
    1. CBF (Conventional Beamforming)  -- fast, single-object, existing method.
    2. MUSIC-2D                        -- super-resolution, multi-object, new method.

QIDK Output:
    Results are displayed on-screen using ASCII terminal art and optionally
    saved to a JSON results file. No GUI dependencies -- terminal output only,
    suitable for QIDK embedded deployment.

Usage:
    # Single file, CBF method:
    python qidk_pipeline.py --file ./Single_Subject/D1m_0d_HR_80_pm_5_BR_15.0.bin --method cbf

    # Single file, 2D MUSIC:
    python qidk_pipeline.py --file ./Single_Subject/D1m_0d_HR_80_pm_5_BR_15.0.bin --method music

    # Both methods, compare:
    python qidk_pipeline.py --file ./Single_Subject/D1m_0d_HR_80_pm_5_BR_15.0.bin --method both

    # Batch folder processing:
    python qidk_pipeline.py --folder ./Single_Subject/ --method music --max_files 20
"""

import argparse
import os
import sys
import time
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Module imports from this package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dc_offset_correction import circle_fit_correction, mean_subtraction
from clutter_suppression import svd_clutter_removal, mean_clutter_removal
from phase_demodulation import demodulate
from vital_sign_extraction import extract_vital_signs
from music_2d import run_music_pipeline, detect_targets_music
from heterogeneous import HeterogeneousPipeline


# ===========================================================================
# Hardware Constants -- AWR1843 / DCA1000  (do NOT change)
# ===========================================================================
F_MIN      = 77e9
F_MAX      = 79.982e9
C          = 3e8
BW         = F_MAX - F_MIN
TM         = 100e-6           # chirp duration
F_ADC      = 600e3            # ADC sampling rate
LAMBDA_C   = C / ((F_MIN + F_MAX) / 2.0)
LAMBDA_MAX = C / F_MIN
D_ANT      = LAMBDA_MAX / 2   # antenna spacing (half-wavelength at f_min)
R_MAX      = (C * TM / 2.0) * (F_ADC / BW)
R_RES      = C / (2.0 * BW)

N_CHIRPS    = 512   # ADC samples per chirp
M_CHIRPS    = 512   # chirps per frame
NUM_RX      = 4     # receive antennas
N_FFT_RANGE = 512   # range FFT size

# Slow-time fs: calibrated from dataset (512 chirps / 25.6 s)
FS_VITAL   = M_CHIRPS / 25.6   # = 20.0 Hz

# Azimuth search grid
AZIMUTH_GRID = np.arange(90, -90 - 2.5, -2.5)  # +90 to -90 in 2.5-degree steps
RANGE_AXIS   = np.linspace(0, R_MAX, N_FFT_RANGE // 2)

# Window for range FFT
WIN_RANGE = np.hanning(N_CHIRPS)


# ===========================================================================
# DCA1000 binary reader
# ===========================================================================

def read_dca1000(path: str) -> np.ndarray:
    """
    Read raw 16-bit IQ binary from DCA1000 LVDS capture.

    Returns:
        data: complex64 array, shape [NUM_RX, numChirps * N_CHIRPS].
    """
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.int16)

    I = raw[0::4] + 1j * raw[2::4]
    Q = raw[1::4] + 1j * raw[3::4]
    LVDS = np.empty(I.size + Q.size, dtype=np.complex64)
    LVDS[0::2] = I
    LVDS[1::2] = Q

    num_chirps = raw.shape[0] // (2 * N_CHIRPS * NUM_RX)
    LVDS = LVDS.reshape((N_CHIRPS * NUM_RX, num_chirps), order='F').T

    out = np.zeros((NUM_RX, num_chirps * N_CHIRPS), dtype=np.complex64)
    for rx in range(NUM_RX):
        for chp in range(num_chirps):
            out[rx, chp * N_CHIRPS:(chp + 1) * N_CHIRPS] = \
                LVDS[chp, rx * N_CHIRPS:(rx + 1) * N_CHIRPS]
    return out


# ===========================================================================
# Beamforming (CBF -- Conventional Beam Forming)
# ===========================================================================

def cbf_weights(rx_positions_wl: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Conventional (Delay-and-Sum) beamformer weights.

    Args:
        rx_positions_wl: RX element positions in wavelength units [NUM_RX].
        angles_deg:      Azimuth angles to steer [num_angles].

    Returns:
        W: Steering matrix [NUM_RX, num_angles], normalised by NUM_RX.
    """
    ang_rad = np.deg2rad(angles_deg)
    W = np.exp(-1j * 2 * np.pi
               * rx_positions_wl[:, None]
               * np.sin(ang_rad)[None, :])
    return W / NUM_RX


def apply_cbf(if_data: np.ndarray) -> np.ndarray:
    """
    Apply CBF across RX aperture.

    Args:
        if_data: [NUM_RX, M_CHIRPS, N_CHIRPS] complex.

    Returns:
        beamed: [num_angles, M_CHIRPS, N_CHIRPS] complex.
    """
    rx_pos_wl = np.arange(NUM_RX) * (D_ANT / LAMBDA_C)
    W = cbf_weights(rx_pos_wl, AZIMUTH_GRID)          # [NUM_RX, num_angles]
    return np.einsum('ra,rcs->acs', W, if_data)        # [num_angles, M, N]


# ===========================================================================
# Range-FFT helper
# ===========================================================================

def range_fft_cube(beamed: np.ndarray) -> np.ndarray:
    """
    Apply range FFT across all angle-chirp combinations.

    OPTIMISED: Single batched np.fft.fft call on the entire
    [num_angles, M_CHIRPS, N_CHIRPS] cube instead of a per-angle loop.
    This lets NumPy/BLAS schedule the FFTs efficiently across all CPU cores.

    Args:
        beamed: [num_angles, M_CHIRPS, N_CHIRPS] complex.

    Returns:
        rfft_cube: [num_angles, M_CHIRPS, N_FFT_RANGE//2] complex.
    """
    # Apply Hanning window and batch-FFT in one call  (axis=-1 = fast-time)
    windowed = beamed * WIN_RANGE[np.newaxis, np.newaxis, :]   # broadcast
    Xf = np.fft.fft(windowed, n=N_FFT_RANGE, axis=-1)          # [A, M, N_FFT]
    return Xf[:, :, :N_FFT_RANGE // 2].astype(np.complex64)     # one-sided


# ===========================================================================
# AutoCorrelation heart-rate estimator  [12]
# ===========================================================================

def autocorr_rate(signal: np.ndarray, fs: float) -> float | None:
    """Estimate rate [bpm] via ACF first peak. Returns None if no peak found."""
    x = signal - np.mean(signal)
    r = np.correlate(x, x, mode='full')
    r = r[r.size // 2:]
    r /= (r[0] + 1e-30)
    lags_s = np.arange(len(r)) / fs
    peaks, _ = find_peaks(r, distance=int(fs * 0.4))
    if len(peaks) == 0:
        return None
    T = lags_s[peaks][0]
    return (1.0 / T) * 60.0 if T > 0 else None


# ===========================================================================
# Vital-sign extraction on a slow-time IQ signal
# ===========================================================================

def extract_vitals_from_iq(slow_iq: np.ndarray) -> dict:
    """
    DC correction -> phase demodulation -> vital sign FFT.

    Args:
        slow_iq: Complex slow-time signal [M_CHIRPS].

    Returns:
        dict: hr_fft, br_fft, hr_ac, br_ac, disp_mm, resp_signal, heart_signal.
    """
    # Circle-fit DC correction [12]
    slow_iq_corr, _ = circle_fit_correction(slow_iq)

    # Phase demodulation [14]
    demod = demodulate(slow_iq_corr, wavelength_m=LAMBDA_C, remove_dc_in_phase=True)
    disp_mm = demod['displacement_mm']

    # Vital sign extraction [14]
    vs = extract_vital_signs(
        disp_mm, fs=FS_VITAL,
        resp_band=(0.1, 0.5),
        heart_band=(0.8, 2.5),
        estimator='fft',
    )

    # AutoCorr estimators [12]
    hr_ac = autocorr_rate(vs['heart_signal_mm'], FS_VITAL)
    br_ac = autocorr_rate(vs['resp_signal_mm'], FS_VITAL)

    return {
        'hr_fft':       vs['heart_rate_bpm'],
        'br_fft':       vs['resp_rate_bpm'],
        'hr_ac':        hr_ac,
        'br_ac':        br_ac,
        'disp_mm':      disp_mm,
        'resp_signal':  vs['resp_signal_mm'],
        'heart_signal': vs['heart_signal_mm'],
    }


# ===========================================================================
# CBF Single-Object Pipeline  [12][14]
# ===========================================================================

def run_cbf_pipeline(bin_path: str) -> dict:
    """
    Conventional Beamforming pipeline -- single dominant target.

    Steps:
        1. Read & reshape DCA1000 binary.
        2. Stage-1a: Mean subtraction per angle [14].
        3. Range FFT.
        4. SVD rank-1 clutter removal [12].
        5. Peak detection -> target (range, angle).
        6. Stage-1b: Circle-fit DC at target bin [12].
        7. Phase demodulation [14].
        8. Vital sign extraction (FFT + AutoCorr).

    Returns:
        dict: hr_fft, br_fft, hr_ac, br_ac, range_m, angle_deg, timing, ...
    """
    timing = {}
    num_angles = len(AZIMUTH_GRID)

    # Step 1: Read
    t0 = time.perf_counter()
    raw = read_dca1000(bin_path)
    if_data = raw.reshape(NUM_RX, M_CHIRPS, N_CHIRPS).astype(np.complex64)
    timing['read_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 2: CBF beamforming
    t0 = time.perf_counter()
    beamed = apply_cbf(if_data)                              # [num_angles, M, N]
    timing['cbf_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 3: Mean subtraction per angle (Stage-1a)
    t0 = time.perf_counter()
    # OPTIMISED: vectorised mean subtraction across the whole cube at once
    # (was a Python for-loop over 73 angles)
    beamed -= np.mean(beamed, axis=1, keepdims=True)   # [A, M, N] – in-place
    beamed = beamed.astype(np.complex64)
    timing['dc1a_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 4: Range FFT (OPTIMISED: single batched FFT call)
    t0 = time.perf_counter()
    rfft_cube = range_fft_cube(beamed)                    # [num_angles, M, N_FFT//2]
    timing['rfft_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 5: SVD rank-1 clutter removal — PARALLELISED across azimuth angles
    # Uses Power Iteration (fast path) inside svd_clutter_removal.
    # ThreadPoolExecutor maps each of the 73 angle slices to a worker thread,
    # saturating all ARM cores on the QIDK simultaneously.
    t0 = time.perf_counter()
    rfft_mean = np.zeros((num_angles, N_FFT_RANGE // 2))

    def _clean_one_angle(nn):
        cleaned, _ = svd_clutter_removal(rfft_cube[nn], rank=1, auto_rank=False)
        return nn, cleaned.astype(np.complex64), np.mean(np.abs(cleaned), axis=0)

    n_workers = min(num_angles, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_clean_one_angle, nn) for nn in range(num_angles)]
        for fut in as_completed(futures):
            nn, cleaned, mean_profile = fut.result()
            rfft_cube[nn]  = cleaned
            rfft_mean[nn]  = mean_profile
    timing['svd_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 6: Detect dominant target bin
    t0 = time.perf_counter()
    # Skip near-field (<0.3 m)
    min_bin = max(1, int(0.3 / R_MAX * (N_FFT_RANGE // 2)))
    masked = rfft_mean.copy()
    masked[:, :min_bin] = 0
    flat_idx = int(np.argmax(masked))
    peak_ai = flat_idx // (N_FFT_RANGE // 2)
    peak_ri = flat_idx % (N_FFT_RANGE // 2)
    peak_range_m = float(RANGE_AXIS[peak_ri])
    peak_angle_deg = float(AZIMUTH_GRID[peak_ai])
    timing['detect_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 7: Vital signs at detected bin
    t0 = time.perf_counter()
    slow_iq = rfft_cube[peak_ai, :, peak_ri]
    vitals = extract_vitals_from_iq(slow_iq)
    timing['vitals_ms'] = (time.perf_counter() - t0) * 1e3

    timing['total_ms'] = sum(timing.values())

    return {
        'method':        'CBF',
        'hr_fft':        vitals['hr_fft'],
        'br_fft':        vitals['br_fft'],
        'hr_ac':         vitals['hr_ac'],
        'br_ac':         vitals['br_ac'],
        'range_m':       peak_range_m,
        'angle_deg':     peak_angle_deg,
        'ra_heatmap':    rfft_mean,
        'slow_iq':       slow_iq,
        'disp_mm':       vitals['disp_mm'],
        'resp_signal':   vitals['resp_signal'],
        'heart_signal':  vitals['heart_signal'],
        'num_targets':   1,
        'all_targets':   [{'target_id': 1, 'range_m': peak_range_m,
                           'angle_deg': peak_angle_deg, 'peak_db': 0.0}],
        'timing':        timing,
    }


# ===========================================================================
# 2D MUSIC Multi-Object Pipeline  [NEW][12][14]
# ===========================================================================

def run_music_2d_pipeline(
    bin_path: str,
    num_targets: int = 3,
    m1: int = 3,
) -> dict:
    """
    2D MUSIC super-resolution pipeline -- multiple objects.

    Steps:
        1. Read & reshape DCA1000 binary.
        2. Stage-1a: Mean subtraction per antenna [14].
        3. Range FFT across all RX channels.
        4. SVD rank-1 clutter removal per RX channel [12].
        5. 2D MUSIC spectrum computation [NEW].
        6. Multi-target peak detection from MUSIC spectrum [NEW].
        7. For each detected target: Stage-1b DC correction + vital signs [12][14].

    Returns:
        dict: targets list with (hr, br, range, angle) per object;
              primary_target for the strongest object.
    """
    timing = {}

    # Step 1: Read
    t0 = time.perf_counter()
    raw = read_dca1000(bin_path)
    if_data = raw.reshape(NUM_RX, M_CHIRPS, N_CHIRPS).astype(np.complex64)
    timing['read_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 2: Mean subtraction per RX channel (Stage-1a, applied per-antenna)
    t0 = time.perf_counter()
    for rx in range(NUM_RX):
        m = if_data[rx]
        if_data[rx] = (m - np.mean(m, axis=0, keepdims=True)).astype(np.complex64)
    timing['dc1a_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 3: Range FFT per RX channel
    t0 = time.perf_counter()
    rp_cube = np.zeros((NUM_RX, M_CHIRPS, N_FFT_RANGE // 2), dtype=np.complex64)
    for rx in range(NUM_RX):
        Xf = np.fft.fft(
            if_data[rx] * WIN_RANGE[np.newaxis, :],
            n=N_FFT_RANGE, axis=1)
        rp_cube[rx] = Xf[:, :N_FFT_RANGE // 2].astype(np.complex64)
    timing['rfft_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 4: SVD clutter removal per RX channel [12]
    t0 = time.perf_counter()
    for rx in range(NUM_RX):
        cleaned, _ = svd_clutter_removal(rp_cube[rx], rank=1, auto_rank=False)
        rp_cube[rx] = cleaned.astype(np.complex64)
    timing['svd_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 5: 2D MUSIC spectrum [NEW]
    t0 = time.perf_counter()
    music_result = run_music_pipeline(
        range_profile_cube=rp_cube,
        fc=(F_MIN + F_MAX) / 2.0,
        d=D_ANT,
        c=C,
        lambda_c=LAMBDA_C,
        range_axis=RANGE_AXIS,
        azimuth_deg=AZIMUTH_GRID,
        m1=m1,
        num_targets=num_targets,
        min_range_m=0.3,
    )
    timing['music_ms'] = (time.perf_counter() - t0) * 1e3

    detected_targets = music_result['targets']
    K_est = music_result['K_est']

    # Step 6: CBF-based range-azimuth heatmap (for reference)
    beamed = apply_cbf(rp_cube[:, :, :].mean(axis=2, keepdims=True).
                       repeat(N_FFT_RANGE // 2, axis=2))
    # Just use the MUSIC spectrum as our localisation map
    ra_heatmap = 10 ** (music_result['spectrum_db'] / 10.0)  # linear

    # Step 7: Vital signs for each detected target
    t0 = time.perf_counter()
    target_results = []
    for tgt in detected_targets:
        ri = tgt['range_idx']
        ai = tgt['angle_idx']

        # Find nearest azimuth angle in CBF grid for extracting slow-time IQ
        # (MUSIC gives us precise angle; we use nearest CBF angle for extraction)
        cbf_angle_idx = int(np.argmin(np.abs(AZIMUTH_GRID - tgt['angle_deg'])))
        slow_iq = rp_cube[0, :, ri]  # Use RX-0 slow-time at detected range bin
        # For multi-antenna, combine using CBF weights at detected angle
        sv = np.exp(-1j * 2 * np.pi * (D_ANT / LAMBDA_C)
                    * np.arange(NUM_RX)
                    * np.sin(np.deg2rad(tgt['angle_deg'])))
        # Beamformed slow-time IQ at target (range_bin, angle)
        slow_iq_bf = np.zeros(M_CHIRPS, dtype=complex)
        for rx in range(NUM_RX):
            slow_iq_bf += sv[rx].conj() * rp_cube[rx, :, ri]
        slow_iq_bf /= NUM_RX

        vitals = extract_vitals_from_iq(slow_iq_bf)

        target_results.append({
            'target_id':  tgt['target_id'],
            'range_m':    tgt['range_m'],
            'angle_deg':  tgt['angle_deg'],
            'peak_db':    tgt['peak_db'],
            'hr_fft':     vitals['hr_fft'],
            'br_fft':     vitals['br_fft'],
            'hr_ac':      vitals['hr_ac'],
            'br_ac':      vitals['br_ac'],
            'disp_mm':    vitals['disp_mm'],
        })

    timing['vitals_ms'] = (time.perf_counter() - t0) * 1e3
    timing['total_ms'] = sum(timing.values())

    primary = target_results[0] if target_results else None

    return {
        'method':        '2D-MUSIC',
        'hr_fft':        primary['hr_fft'] if primary else 0.0,
        'br_fft':        primary['br_fft'] if primary else 0.0,
        'hr_ac':         primary['hr_ac'] if primary else None,
        'br_ac':         primary['br_ac'] if primary else None,
        'range_m':       primary['range_m'] if primary else 0.0,
        'angle_deg':     primary['angle_deg'] if primary else 0.0,
        'ra_heatmap':    ra_heatmap,
        'music_spectrum': music_result['spectrum_db'],
        'K_est':         K_est,
        'num_targets':   len(target_results),
        'all_targets':   target_results,
        'primary_target': primary,
        'timing':        timing,
    }


# ===========================================================================
# Heterogeneous Tri-Engine Pipeline (CPU + GPU + NPU) [NEW]
# ===========================================================================

_tri_engine_singleton = None

def get_tri_engine_pipeline() -> HeterogeneousPipeline:
    global _tri_engine_singleton
    if _tri_engine_singleton is None:
        _tri_engine_singleton = HeterogeneousPipeline()
    return _tri_engine_singleton

def run_tri_engine_pipeline(bin_path: str) -> dict:
    """
    Heterogeneous execution across CPU (I/O, Demod, Vitals),
    GPU (CBF, Range-FFT, Batched SVD), and NPU (Target Detection).
    """
    pipe = get_tri_engine_pipeline()
    return pipe.process_single_frame(bin_path)


# ===========================================================================
# QIDK Terminal Display  -- show HR & BR on screen
# ===========================================================================

def display_qidk_result(result: dict, filename: str = "", true_hr: int = None,
                        true_br: float = None):
    """
    Display HR and BR estimates on the QIDK terminal screen.

    Uses ASCII art blocks and colour ANSI codes for a readable embedded
    dashboard. This is the primary QIDK "on-screen" output.

    Args:
        result:    Pipeline result dict.
        filename:  Source filename for header display.
        true_hr:   Ground-truth HR (bpm), optional.
        true_br:   Ground-truth BR (br/min), optional.
    """
    # ANSI colour codes
    GREEN  = '\033[92m'
    CYAN   = '\033[96m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'
    MAGENTA = '\033[95m'
    WHITE  = '\033[97m'

    method = result.get('method', 'UNKNOWN')
    hr_fft = result.get('hr_fft', 0.0)
    br_fft = result.get('br_fft', 0.0)
    hr_ac  = result.get('hr_ac', None)
    br_ac  = result.get('br_ac', None)
    rng_m  = result.get('range_m', 0.0)
    ang_d  = result.get('angle_deg', 0.0)
    n_tgt  = result.get('num_targets', 1)
    timing = result.get('timing', {})

    sep = "=" * 68

    print(f"\n{BOLD}{WHITE}{sep}{RESET}")
    print(f"{BOLD}{CYAN}  FMCW RADAR -- QIDK VITAL SIGNS DISPLAY{RESET}")
    print(f"{BOLD}{WHITE}{sep}{RESET}")
    if filename:
        fname = Path(filename).name[:55]
        print(f"  {YELLOW}File   : {WHITE}{fname}{RESET}")
    print(f"  {YELLOW}Method : {WHITE}{method}{RESET}")
    print(f"  {YELLOW}Targets: {WHITE}{n_tgt}{RESET}")
    print(f"{BOLD}{WHITE}{'─' * 68}{RESET}")

    # Primary target vital signs (large display)
    hr_col = GREEN if true_hr is None or abs(hr_fft - true_hr) <= 10 else RED
    br_col = GREEN if true_br is None or abs(br_fft - true_br) <= 3 else RED

    print(f"\n  {BOLD}{YELLOW}-- PRIMARY TARGET --{RESET}")
    print(f"  {BOLD}Location : {WHITE}{rng_m:.2f} m  |  {ang_d:.1f} deg{RESET}")
    print()
    print(f"  {BOLD}{'HEART RATE':^30}  {'BREATHING RATE':^30}{RESET}")
    print(f"  {BOLD}{hr_col}{'FFT: ' + f'{hr_fft:.1f} bpm':^30}{RESET}  "
          f"{BOLD}{br_col}{'FFT: ' + f'{br_fft:.1f} br/min':^30}{RESET}")

    if hr_ac is not None:
        hr_ac_col = GREEN if true_hr is None or abs(hr_ac - true_hr) <= 10 else RED
        print(f"  {hr_ac_col}{'AC:  ' + f'{hr_ac:.1f} bpm':^32}{RESET}  ", end='')
    else:
        print(f"  {'AC:  N/A':^32}  ", end='')
    if br_ac is not None:
        br_ac_col = GREEN if true_br is None or abs(br_ac - true_br) <= 3 else RED
        print(f"{br_ac_col}{'AC:  ' + f'{br_ac:.1f} br/min':^32}{RESET}")
    else:
        print(f"{'AC:  N/A':^32}")

    if true_hr is not None:
        hr_err = abs(hr_fft - true_hr)
        hr_err_col = GREEN if hr_err <= 5 else (YELLOW if hr_err <= 10 else RED)
        print(f"  {hr_err_col}{'True HR: ' + str(true_hr) + ' bpm  |  Err: ' + f'{hr_err:.1f} bpm':^64}{RESET}")
    if true_br is not None and true_br > 0:
        br_err = abs(br_fft - true_br)
        br_err_col = GREEN if br_err <= 2 else (YELLOW if br_err <= 4 else RED)
        print(f"  {br_err_col}{'True BR: ' + f'{true_br:.1f} br/min  |  Err: ' + f'{br_err:.1f} br/min':^64}{RESET}")

    # Multi-target breakdown (MUSIC only)
    all_tgts = result.get('all_targets', [])
    if method == '2D-MUSIC' and len(all_tgts) > 1:
        print(f"\n{BOLD}{MAGENTA}  -- ALL DETECTED OBJECTS ({len(all_tgts)}) --{RESET}")
        hdr = f"  {'ID':>3}  {'Range':>7}  {'Angle':>7}  {'HR (FFT)':>10}  {'BR (FFT)':>12}  {'HR (AC)':>10}"
        print(f"{BOLD}{WHITE}{hdr}{RESET}")
        print(f"  {'─' * 62}")
        for tgt in all_tgts:
            hr_str = f"{tgt['hr_fft']:.1f} bpm"
            br_str = f"{tgt['br_fft']:.1f} br/min"
            ac_str = f"{tgt['hr_ac']:.1f}" if tgt['hr_ac'] else "N/A"
            print(f"  {tgt['target_id']:>3}  "
                  f"{tgt['range_m']:>6.2f}m  "
                  f"{tgt['angle_deg']:>6.1f}°  "
                  f"{hr_str:>10}  "
                  f"{br_str:>12}  "
                  f"{ac_str:>10}")

    # Timing
    print(f"\n{BOLD}{WHITE}{'─' * 68}{RESET}")
    print(f"  {YELLOW}Timing breakdown:{RESET}")
    for k, v in timing.items():
        if k != 'total_ms':
            print(f"    {k:<20} {v:>8.1f} ms")
    total = timing.get('total_ms', 0.0)
    print(f"  {BOLD}{WHITE}  {'TOTAL':.<20} {total:>8.1f} ms  ({1000.0/total:.1f} fps){RESET}")
    print(f"{BOLD}{WHITE}{sep}{RESET}\n")


# ===========================================================================
# Filename metadata parser
# ===========================================================================

_FNAME_RE = re.compile(
    r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin'
)


def parse_filename_meta(path: str) -> dict:
    """Extract ground-truth metadata from standardised filename."""
    m = _FNAME_RE.match(Path(path).name)
    if not m:
        return {'dist_m': None, 'angle_deg': None,
                'hr_true': None, 'br_true': None, 'has_br': False}
    br_val = float(m.group('br'))
    return {
        'dist_m':    float(m.group('dist')),
        'angle_deg': int(m.group('angle')),
        'hr_true':   int(m.group('hr')),
        'br_true':   br_val,
        'has_br':    br_val > 0,
    }


# ===========================================================================
# Single-file runner
# ===========================================================================

def process_single_file(
    bin_path: str,
    method: str = 'both',
    num_targets: int = 3,
    m1: int = 3,
    quiet: bool = False,
    save_json: str = None,
) -> dict:
    """
    Run the selected pipeline on a single .bin file and display results.

    Args:
        bin_path:    Path to .bin DCA1000 capture.
        method:      'cbf', 'music', or 'both'.
        num_targets: Max targets for MUSIC.
        m1:          MUSIC spatial smoothing window.
        quiet:       If True, suppress on-screen display.
        save_json:   Optional path to save result JSON.

    Returns:
        dict with all results.
    """
    meta = parse_filename_meta(bin_path)
    results = {}

    if method in ('tri', 'all'):
        print(f"  Running Tri-Engine (CPU+GPU+NPU) pipeline on {Path(bin_path).name} ...")
        tri_res = run_tri_engine_pipeline(bin_path)
        results['tri'] = tri_res
        if not quiet:
            display_qidk_result(
                tri_res, bin_path,
                true_hr=meta['hr_true'],
                true_br=meta['br_true'] if meta['has_br'] else None,
            )

    if method in ('cbf', 'both', 'all'):
        print(f"  Running CBF pipeline on {Path(bin_path).name} ...")
        cbf_res = run_cbf_pipeline(bin_path)
        results['cbf'] = cbf_res
        if not quiet:
            display_qidk_result(
                cbf_res, bin_path,
                true_hr=meta['hr_true'],
                true_br=meta['br_true'] if meta['has_br'] else None,
            )

    if method in ('music', 'both', 'all'):
        print(f"  Running 2D-MUSIC pipeline on {Path(bin_path).name} ...")
        music_res = run_music_2d_pipeline(bin_path, num_targets=num_targets, m1=m1)
        results['music'] = music_res
        if not quiet:
            display_qidk_result(
                music_res, bin_path,
                true_hr=meta['hr_true'],
                true_br=meta['br_true'] if meta['has_br'] else None,
            )

    results['meta'] = meta

    if save_json:
        # Serialise (remove numpy arrays)
        def _serialise(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        def _clean(d):
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items()
                        if not isinstance(v, np.ndarray)}
            if isinstance(d, list):
                return [_clean(i) for i in d]
            return _serialise(d)

        with open(save_json, 'w') as f:
            json.dump(_clean(results), f, indent=2)
        print(f"  Results saved -> {save_json}")

    return results


# ===========================================================================
# Batch folder runner
# ===========================================================================

def process_folder(
    folder_path: str,
    method: str = 'music',
    num_targets: int = 3,
    m1: int = 3,
    max_files: int = None,
    save_csv: str = None,
) -> list:
    """
    Process all .bin files in a folder and print aggregate statistics.

    Args:
        folder_path: Path to folder containing .bin files.
        method:      'cbf', 'music', or 'both'.
        num_targets: Max targets for MUSIC.
        m1:          MUSIC spatial smoothing window.
        max_files:   Process at most this many files.
        save_csv:    Optional CSV path for aggregate results.

    Returns:
        List of result dicts.
    """
    folder = Path(folder_path)
    bins = sorted(folder.rglob('*.bin'))
    valid = [p for p in bins if _FNAME_RE.match(p.name)]

    if max_files is not None:
        valid = valid[:max_files]

    print(f"\n{'=' * 68}")
    print(f"  FMCW QIDK Batch Processing")
    print(f"  Folder : {folder}")
    print(f"  Files  : {len(valid)}   Method: {method.upper()}")
    print(f"{'=' * 68}\n")

    all_results = []
    errors = []

    for bin_path in valid:
        meta = parse_filename_meta(str(bin_path))
        try:
            res = process_single_file(
                str(bin_path), method=method,
                num_targets=num_targets, m1=m1, quiet=False,
            )
            entry = {
                'filename':    bin_path.name,
                'dist_m':      meta['dist_m'],
                'angle_deg':   meta['angle_deg'],
                'hr_true':     meta['hr_true'],
                'br_true':     meta['br_true'],
                'has_br':      meta['has_br'],
            }
            # Add CBF results
            if 'cbf' in res:
                entry.update({
                    'cbf_hr_fft': res['cbf']['hr_fft'],
                    'cbf_br_fft': res['cbf']['br_fft'],
                    'cbf_hr_ac':  res['cbf']['hr_ac'],
                    'cbf_range':  res['cbf']['range_m'],
                    'cbf_angle':  res['cbf']['angle_deg'],
                    'cbf_time_ms': res['cbf']['timing']['total_ms'],
                })
            # Add TRI results
            if 'tri' in res:
                entry.update({
                    'tri_hr_fft': res['tri']['hr_fft'],
                    'tri_br_fft': res['tri']['br_fft'],
                    'tri_hr_ac':  res['tri']['hr_ac'],
                    'tri_range':  res['tri']['range_m'],
                    'tri_angle':  res['tri']['angle_deg'],
                    'tri_time_ms': res['tri']['timing']['total_ms'],
                })
            # Add MUSIC results
            if 'music' in res:
                entry.update({
                    'music_hr_fft':  res['music']['hr_fft'],
                    'music_br_fft':  res['music']['br_fft'],
                    'music_hr_ac':   res['music']['hr_ac'],
                    'music_range':   res['music']['range_m'],
                    'music_angle':   res['music']['angle_deg'],
                    'music_n_targets': res['music']['num_targets'],
                    'music_time_ms': res['music']['timing']['total_ms'],
                })
            all_results.append(entry)
        except Exception as exc:
            errors.append({'file': bin_path.name, 'error': str(exc)})
            print(f"  ERROR on {bin_path.name}: {exc}")

    # Aggregate statistics
    if all_results:
        _print_aggregate_stats(all_results, method)

    if save_csv and all_results:
        import csv
        keys = all_results[0].keys()
        with open(save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  CSV saved -> {save_csv}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    {e['file']}: {e['error']}")

    return all_results


def _print_aggregate_stats(results: list, method: str):
    """Print aggregate HR/BR accuracy statistics."""
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    BOLD = '\033[1m'; RESET = '\033[0m'; WHITE = '\033[97m'; CYAN = '\033[96m'

    def _mae(key_est, key_true):
        vals = [(r[key_est], r[key_true]) for r in results
                if r.get(key_est) and r.get(key_true)]
        if not vals:
            return None, 0
        errs = [abs(e - t) for e, t in vals]
        return np.mean(errs), len(errs)

    print(f"\n{'=' * 68}")
    print(f"{BOLD}{WHITE}  AGGREGATE STATISTICS{RESET}")
    print(f"{'─' * 68}")

    if method in ('tri', 'all'):
        hr_mae, n_hr = _mae('tri_hr_fft', 'hr_true')
        if hr_mae is not None:
            col = GREEN if hr_mae <= 10 else (YELLOW if hr_mae <= 15 else RED)
            print(f"  TRI   HR MAE : {col}{hr_mae:.2f} bpm{RESET}  (n={n_hr})")
        br_vals = [(r['tri_br_fft'], r['br_true']) for r in results
                   if r.get('has_br') and r.get('tri_br_fft')]
        if br_vals:
            br_mae = np.mean([abs(e - t) for e, t in br_vals])
            col = GREEN if br_mae <= 3 else (YELLOW if br_mae <= 5 else RED)
            print(f"  TRI   BR MAE : {col}{br_mae:.2f} br/min{RESET}  (n={len(br_vals)})")
        tri_times = [r['tri_time_ms'] for r in results if r.get('tri_time_ms') is not None]
        if tri_times:
            print(f"  TRI   Avg Lat: {CYAN}{np.mean(tri_times):.1f} ms{RESET}  ({1000.0/np.mean(tri_times):.1f} fps)")

    if method in ('cbf', 'both', 'all'):
        hr_mae, n_hr = _mae('cbf_hr_fft', 'hr_true')
        if hr_mae is not None:
            col = GREEN if hr_mae <= 10 else (YELLOW if hr_mae <= 15 else RED)
            print(f"  CBF   HR MAE : {col}{hr_mae:.2f} bpm{RESET}  (n={n_hr})")

    if method in ('music', 'both', 'all'):
        hr_mae, n_hr = _mae('music_hr_fft', 'hr_true')
        if hr_mae is not None:
            col = GREEN if hr_mae <= 10 else (YELLOW if hr_mae <= 15 else RED)
            print(f"  MUSIC HR MAE : {col}{hr_mae:.2f} bpm{RESET}  (n={n_hr})")
        br_vals = [(r['music_br_fft'], r['br_true']) for r in results
                   if r.get('has_br') and r.get('music_br_fft')]
        if br_vals:
            br_mae = np.mean([abs(e - t) for e, t in br_vals])
            col = GREEN if br_mae <= 3 else (YELLOW if br_mae <= 5 else RED)
            print(f"  MUSIC BR MAE : {col}{br_mae:.2f} br/min{RESET}  (n={len(br_vals)})")

    print(f"{'=' * 68}\n")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FMCW QIDK Vital Signs Pipeline -- CBF + 2D MUSIC"
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--file',   type=str, help="Single .bin file to process.")
    mode.add_argument('--folder', type=str, help="Folder of .bin files (batch).")

    p.add_argument('--method', choices=['cbf', 'music', 'both', 'tri', 'all'], default='tri',
                   help="Detection method: cbf, music, both, tri, all (default: tri)")
    p.add_argument('--num_targets', type=int, default=3,
                   help="Max targets for 2D MUSIC (default: 3)")
    p.add_argument('--m1', type=int, default=3,
                   help="MUSIC spatial smoothing window size (default: 3)")
    p.add_argument('--max_files', type=int, default=None,
                   help="Max files for batch mode.")
    p.add_argument('--save_json', type=str, default=None,
                   help="Save single-file result to JSON.")
    p.add_argument('--save_csv', type=str, default=None,
                   help="Save batch results to CSV.")
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.file:
        process_single_file(
            bin_path=args.file,
            method=args.method,
            num_targets=args.num_targets,
            m1=args.m1,
            save_json=args.save_json,
        )
    elif args.folder:
        process_folder(
            folder_path=args.folder,
            method=args.method,
            num_targets=args.num_targets,
            m1=args.m1,
            max_files=args.max_files,
            save_csv=args.save_csv,
        )
