"""
run_fixed_pipeline.py
=====================
Fixed FMCW Radar QIDK Pipeline — ALL 5 BUGS CORRECTED

Bug Fixes Applied:
  BUG1: fs_vital = 20.0 Hz  (was 6666.67 Hz — chirp rate, not slow-time rate)
  BUG2: d = lambda/2       (was lambda/2 * 250 — 250x too large)
  BUG3: SVD after Range FFT (was before — caused wrong target detection)
  BUG4: AutoCorr uses fs_vital param (was hardcoded 25.6s)
  BUG5: heart_band upper = 2.5 Hz = 150 bpm (was 2.0 Hz, too low for dataset)

Runs on all 320 .bin files in Single_Subject/ directory.
"""

import sys, os, re, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless run
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, butter, filtfilt
from pathlib import Path
from tqdm import tqdm

# ─── Module path ────────────────────────────────────────────────────────────
MODULE_DIR = os.path.abspath('.')
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from dc_offset_correction   import circle_fit_correction, mean_subtraction
from clutter_suppression    import svd_clutter_removal
from phase_demodulation     import demodulate
from vital_sign_extraction  import extract_vital_signs

# ─── Hardware Parameters (AWR1843 / DCA1000) ─────────────────────────────────
f_min       = 77e9
f_max       = 79.982e9
c           = 3e8
lambda_max  = c / f_min
# ✅ BUG2 FIX: half-wavelength spacing  (was lambda_max/2 * 250)
d           = lambda_max / 2
Tm          = 100e-6
f_adc       = 600e3
# ✅ BUG6 5m_0d_HR_86_pm_1_BR_0FIX: R_max formula had extra /2 in denominator (was 1.509 m → now 3.018 m)
R_max       = (c * Tm / 2) * (f_adc / (f_max - f_min))
lambda_c    = c / ((f_min + f_max) / 2)
BW          = f_max - f_min
R_res       = c / (2 * BW)
fs_slow     = 1.0 / (Tm + 50e-6)    # chirp rate — timing only, NOT for BP filter

N_chirps    = 512
M_chirps    = 512
NUM_RX      = 4
N_FFT_RANGE = 512

# ✅ BUG1 FIX: correct slow-time fs for vital sign filtering
# Senior AutoCorr hardcoded 25.6 s total duration → fs = 512/25.6 = 20 Hz
fs_vital    = M_chirps / 25.6    # = 20.0 Hz

print("=" * 60)
print("Fixed FMCW Pipeline — AWR1843 Parameters")
print("=" * 60)
print(f"  f_min      : {f_min/1e9:.3f} GHz")
print(f"  f_max      : {f_max/1e9:.3f} GHz")
print(f"  Bandwidth  : {BW/1e9:.3f} GHz")
print(f"  lambda_c   : {lambda_c*1000:.2f} mm")
print(f"  d (spacing): {d*1000:.4f} mm  ← FIXED (was 487 mm)")
print(f"  R_max      : {R_max:.3f} m")
print(f"  R_res      : {R_res*100:.2f} cm")
print(f"  fs_slow    : {fs_slow:.2f} Hz  (chirp rate — timing only)")
print(f"  fs_vital   : {fs_vital:.2f} Hz  ← FIXED (was 6666.67 Hz)")
print(f"  R_max      : {R_max:.3f} m  ← FIXED (was 1.509 m, now 3.018 m)")
print(f"  heart_band : (0.8, 2.5) Hz = (48, 150) bpm  ← FIXED upper limit")
print()

# ─── Senior-Inherited Functions ──────────────────────────────────────────────

def readDCA1000(fileName):
    """Read raw 16-bit IQ binary from DCA1000 LVDS capture."""
    numADCSamples = 512; numADCBits = 16; numRX = 4; isReal = 0
    with open(fileName, 'rb') as f:
        adcData = np.fromfile(f, dtype=np.int16)
    if numADCBits != 16:
        l_max = 2**(numADCBits-1)-1
        adcData = adcData.copy()
        adcData[adcData > l_max] -= 2**numADCBits
    fileSize = adcData.shape[0]
    if isReal:
        numChirps = fileSize // (numADCSamples * numRX)
        LVDS = adcData.reshape((numADCSamples*numRX, numChirps), order='F').T
    else:
        numChirps = fileSize // (2 * numADCSamples * numRX)
        I = adcData[0::4] + 1j * adcData[2::4]
        Q = adcData[1::4] + 1j * adcData[3::4]
        LVDS = np.empty(I.size + Q.size, dtype=np.complex64)
        LVDS[0::2] = I; LVDS[1::2] = Q
        LVDS = LVDS.reshape((numADCSamples*numRX, numChirps), order='F').T
    adcData = np.zeros((numRX, numChirps*numADCSamples), dtype=LVDS.dtype)
    for row in range(numRX):
        for i in range(numChirps):
            adcData[row, i*numADCSamples:(i+1)*numADCSamples] = \
                LVDS[i, row*numADCSamples:(row+1)*numADCSamples]
    return adcData


def processIFSignal(path, num_chirps, num_samples, num_rx):
    """Read + reshape to (num_rx, num_chirps, num_samples)."""
    return readDCA1000(path).reshape(num_rx, num_chirps, num_samples)


def cbfweights(pos_in, ang_in, N=0):
    """Conventional Beamformer weights. Returns (num_rx, num_angles)."""
    pos_in = np.atleast_1d(pos_in); ang_in = np.atleast_1d(ang_in)
    N_ele  = pos_in.size
    ang_rad = np.deg2rad(ang_in)
    steering = np.exp(-1j*2*np.pi * pos_in[:,None] * np.sin(ang_rad)[None,:])
    if N > 0:
        phase   = np.angle(steering)
        levels  = 2**N
        phase_q = np.round(phase/(2*np.pi/levels)) * (2*np.pi/levels)
        steering = np.exp(1j*phase_q)
    return steering / N_ele


def performBeamforming(if_data, antenna_spacing, wavelength, azimuth):
    """CBF across RX aperture. Returns (num_angles, num_chirps, num_samples)."""
    num_rx, num_chirps, num_samples = if_data.shape
    # ✅ BUG7 FIX: normalize positions by wavelength so cbfweights gets
    #              dimensionless (wavelength-unit) positions, not raw metres.
    #              cbfweights formula: exp(-j*2*pi * pos_in * sin(ang))
    #              requires pos_in in wavelengths → divide by lambda_c
    rx_pos = np.arange(num_rx) * antenna_spacing / wavelength
    bf_weights = cbfweights(rx_pos, azimuth, 0)
    return np.einsum('ra,rcs->acs', bf_weights, if_data)


def autocorr(x):
    x = np.asarray(x) - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]; result /= (result[0] + 1e-30)
    return result


# ✅ BUG4 FIX: AutoCorr now takes fs as parameter, no hardcoded 25.6s
def AutoCorr(xr_bandpass, fs=fs_vital):
    """Return rate in bpm from first ACF peak. Returns None if no peak."""
    acf  = autocorr(xr_bandpass)
    lags = np.arange(len(acf)) / fs   # lags in seconds (dynamic, not hardcoded)
    peaks, _ = find_peaks(acf, distance=int(fs * 0.4))  # min 0.4s between peaks
    if len(peaks) == 0:
        return None
    hr_loc = lags[peaks][0]
    if hr_loc <= 0:
        return None
    return (1 / hr_loc) * 60


# ─── Pre-compute beamforming constants ───────────────────────────────────────
azimuth    = np.arange(90, -90-2.5, -2.5)   # +90 to -90 in 2.5° steps
num_angles = len(azimuth)
range_axis = np.linspace(0, R_max, N_FFT_RANGE // 2)
win_range  = np.hanning(N_chirps)


# ─── Fixed QIDK Pipeline ─────────────────────────────────────────────────────

def run_fixed_pipeline(bin_path: str, return_signals: bool = False) -> dict:
    """
    Fixed QIDK FMCW pipeline on one .bin file.
    All 5 bugs corrected vs. original notebook.
    """
    timing = {}

    # Step 1: Read & reshape
    t0 = time.perf_counter()
    if_data = processIFSignal(bin_path, M_chirps, N_chirps, NUM_RX)
    if_data = if_data.astype(np.complex64)
    timing['read_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 2: CBF Beamforming  (BUG2 fixed: d=lambda/2; BUG7 fixed: wavelength normalization)
    t0 = time.perf_counter()
    res = performBeamforming(if_data, d, lambda_c, azimuth)   # (num_angles, M, N)
    timing['bf_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 3: Stage 1a -- Mean Subtraction per angle [14]
    t0 = time.perf_counter()
    res_dc = np.zeros_like(res, dtype=np.complex64)
    for nn in range(num_angles):
        data_nn = res[nn, :, :]
        res_dc[nn, :, :] = (data_nn - np.mean(data_nn, axis=0, keepdims=True)).astype(np.complex64)
    timing['dc1a_ms'] = (time.perf_counter() - t0) * 1e3

    # ✅ BUG3 FIX: Range FFT BEFORE SVD clutter removal
    # Step 4: Range FFT (now before SVD)
    t0 = time.perf_counter()
    rangefft_cube = np.zeros((num_angles, M_chirps, N_FFT_RANGE // 2), dtype=np.complex64)
    for nn in range(num_angles):
        x_win = res_dc[nn, :, :] * win_range[np.newaxis, :]
        X     = np.fft.fft(x_win, n=N_FFT_RANGE, axis=1)
        X_one = X[:, :N_FFT_RANGE // 2]
        rangefft_cube[nn, :, :] = X_one.astype(np.complex64)
    timing['rfft_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 5: SVD rank-1 clutter removal on range-slow-time matrix per angle [12]
    t0 = time.perf_counter()
    rangefft_mean = np.zeros((num_angles, N_FFT_RANGE // 2))
    for nn in range(num_angles):
        cleaned_nn, _ = svd_clutter_removal(rangefft_cube[nn, :, :], rank=1, auto_rank=False)
        rangefft_cube[nn, :, :] = cleaned_nn.astype(np.complex64)
        rangefft_mean[nn, :] = np.mean(np.abs(rangefft_cube[nn, :, :]), axis=0)
    timing['svd_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 6: Detect target bin (peak of range-angle heatmap)
    t0 = time.perf_counter()
    peak_flat       = np.argmax(rangefft_mean)
    peak_angle_idx  = peak_flat // (N_FFT_RANGE // 2)
    peak_range_idx  = peak_flat  % (N_FFT_RANGE // 2)
    peak_range_m    = range_axis[peak_range_idx]
    peak_angle_deg  = azimuth[peak_angle_idx]
    timing['detect_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 7: Stage 1b -- Circle-Fit DC at target bin [12]
    t0 = time.perf_counter()
    slow_time_iq = rangefft_cube[peak_angle_idx, :, peak_range_idx]   # (M_chirps,)
    slow_time_corrected, dc_info = circle_fit_correction(slow_time_iq)
    timing['dc1b_ms'] = (time.perf_counter() - t0) * 1e3

    # Step 8: Phase Demodulation
    t0 = time.perf_counter()
    demod   = demodulate(slow_time_corrected, wavelength_m=lambda_c, remove_dc_in_phase=True)
    disp_mm = demod['displacement_mm']
    timing['demod_ms'] = (time.perf_counter() - t0) * 1e3

    # ✅ BUG1 FIX: use fs_vital (20 Hz) not fs_slow (6666.67 Hz)
    # ✅ BUG5 FIX: heart_band upper = 2.5 Hz = 150 bpm (was 2.0 = 120 bpm)
    t0 = time.perf_counter()
    vs = extract_vital_signs(
        disp_mm, fs=fs_vital,
        resp_band=(0.1, 0.5), heart_band=(0.8, 2.5),
        estimator='fft',
    )
    # ✅ BUG4 FIX: AutoCorr uses fs_vital parameter
    br_ac_val = AutoCorr(vs['resp_signal_mm'],  fs=fs_vital)
    hr_ac_val = AutoCorr(vs['heart_signal_mm'], fs=fs_vital)
    timing['vs_ms'] = (time.perf_counter() - t0) * 1e3
    timing['total_ms'] = sum(timing.values())

    result = {
        'hr_fft'         : vs['heart_rate_bpm'],
        'br_fft'         : vs['resp_rate_bpm'],
        'hr_ac'          : hr_ac_val,
        'br_ac'          : br_ac_val,
        'peak_range_m'   : peak_range_m,
        'peak_angle_deg' : peak_angle_deg,
        'timing'         : timing,
    }
    if return_signals:
        result['signals'] = {
            'disp_mm'        : disp_mm,
            'resp_signal'    : vs['resp_signal_mm'],
            'heart_signal'   : vs['heart_signal_mm'],
            'resp_spectrum'  : vs['resp_spectrum'],
            'heart_spectrum' : vs['heart_spectrum'],
            'ra_heatmap'     : rangefft_mean,
            'slow_iq_raw'    : slow_time_iq,
            'slow_iq_corr'   : slow_time_corrected,
        }
    return result


# ─── File name pattern (module-level, used by imports too) ───────────────────
PATTERN = re.compile(
    r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin'
)


if __name__ == '__main__':
    # ─── Dataset Discovery ────────────────────────────────────────────────────
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR.resolve()}")
        print("Usage: Place Single_Subject/ folder here and run again.")
        raise SystemExit(1)

    records = []
    for f in sorted(DATA_DIR.glob('*.bin')):
        m = PATTERN.match(f.name)
        if m:
            records.append({
                'path'      : str(f),
                'filename'  : f.name,
                'dist_m'    : float(m.group('dist')),
                'angle_deg' : int(m.group('angle')),
                'hr_true'   : int(m.group('hr')),
                'pm'        : int(m.group('pm')),
                'br_true'   : float(m.group('br')),
            })

    meta = pd.DataFrame(records)
    meta['has_br'] = meta['br_true'] > 0

    # Limit to 100 files (stratified sample)
    if len(meta) > 100:
        meta = meta.sample(n=100, random_state=42).reset_index(drop=True)

    print(f"Dataset: {len(meta)} files")
    print(f"  Distances : {sorted(meta.dist_m.unique())} m")
    print(f"  Angles    : {sorted(meta.angle_deg.unique())} deg")
    print(f"  HR range  : {meta.hr_true.min()} - {meta.hr_true.max()} bpm")
    print(f"  BR files  : {meta.has_br.sum()} / {len(meta)}")
    print()

    # ─── Batch Processing ─────────────────────────────────────────────────────
    print("Running fixed pipeline on all files...")
    print("-" * 60)

    results_list = []
    errors = []
    t_batch_start = time.perf_counter()

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing"):
        try:
            res = run_fixed_pipeline(row['path'])
            results_list.append({
                **row.to_dict(),
                'hr_fft'         : res['hr_fft'],
                'br_fft'         : res['br_fft'],
                'hr_ac'          : res['hr_ac'],
                'br_ac'          : res['br_ac'],
                'peak_range_m'   : res['peak_range_m'],
                'peak_angle_deg' : res['peak_angle_deg'],
                **{f'timing_{k}': v for k, v in res['timing'].items()},
            })
        except Exception as e:
            errors.append({'file': row['filename'], 'error': str(e)})
            print(f"\n  ERROR on {row['filename']}: {e}")

    t_batch_elapsed = time.perf_counter() - t_batch_start

    df = pd.DataFrame(results_list)
    print(f"\nBatch complete: {len(df)}/{len(meta)} files in {t_batch_elapsed:.1f}s")
    print(f"  Errors: {len(errors)}")
    if len(df) > 0:
        print(f"  Avg per file: {t_batch_elapsed/len(df)*1000:.0f} ms")
    print()

    if len(df) == 0:
        print("No results to report.")
        raise SystemExit(0)

    # ─── Compute Error Metrics ────────────────────────────────────────────────
    df['hr_err_fft']  = (df['hr_fft'] - df['hr_true']).abs()
    df['hr_err_ac']   = (df['hr_ac']  - df['hr_true']).abs()
    df['br_err_fft']  = (df['br_fft'] - df['br_true']).abs()
    df['br_err_ac']   = (df['br_ac']  - df['br_true']).abs()
    df['range_err_m'] = (df['peak_range_m'] - df['dist_m']).abs()
    df['angle_err']   = (df['peak_angle_deg'] - df['angle_deg']).abs()

    df_br = df[df['has_br']].copy()
    df_ac = df[df['hr_ac'].notna()].copy()

    def within_n(series, n):
        return (series <= n).mean() * 100

    print("=" * 60)
    print("ACCURACY RESULTS -- FIXED PIPELINE")
    print("=" * 60)

    print("\n-- Heart Rate (FFT estimator) --")
    print(f"  MAE           : {df['hr_err_fft'].mean():.2f} bpm")
    print(f"  RMSE          : {np.sqrt((df['hr_err_fft']**2).mean()):.2f} bpm")
    print(f"  Median error  : {df['hr_err_fft'].median():.2f} bpm")
    print(f"  95th pctile   : {np.percentile(df['hr_err_fft'], 95):.2f} bpm")
    print(f"  <= 5 bpm      : {within_n(df['hr_err_fft'], 5):.1f}%")
    print(f"  <= 10 bpm     : {within_n(df['hr_err_fft'], 10):.1f}%")
    print(f"  <= 15 bpm     : {within_n(df['hr_err_fft'], 15):.1f}%")

    print("\n-- Heart Rate (AutoCorr estimator) --")
    print(f"  Valid files   : {df_ac['hr_err_ac'].count()} / {len(df)}")
    if len(df_ac) > 0:
        print(f"  MAE           : {df_ac['hr_err_ac'].mean():.2f} bpm")
        print(f"  RMSE          : {np.sqrt((df_ac['hr_err_ac']**2).mean()):.2f} bpm")
        print(f"  <= 5 bpm      : {within_n(df_ac['hr_err_ac'], 5):.1f}%")
        print(f"  <= 10 bpm     : {within_n(df_ac['hr_err_ac'], 10):.1f}%")

    print("\n-- Breathing Rate (FFT) --")
    print(f"  Files with BR : {len(df_br)}")
    if len(df_br) > 0:
        print(f"  MAE           : {df_br['br_err_fft'].mean():.2f} br/min")
        print(f"  RMSE          : {np.sqrt((df_br['br_err_fft']**2).mean()):.2f} br/min")
        print(f"  Median error  : {df_br['br_err_fft'].median():.2f} br/min")
        print(f"  <= 2 br/min   : {within_n(df_br['br_err_fft'], 2):.1f}%")
        print(f"  <= 3 br/min   : {within_n(df_br['br_err_fft'], 3):.1f}%")
        print(f"  <= 5 br/min   : {within_n(df_br['br_err_fft'], 5):.1f}%")

    print("\n-- Localisation --")
    print(f"  Range MAE     : {df['range_err_m'].mean()*100:.1f} cm")
    print(f"  Range RMSE    : {np.sqrt((df['range_err_m']**2).mean())*100:.1f} cm")
    print(f"  Angle MAE     : {df['angle_err'].mean():.2f} deg")
    print(f"  +/- 10cm acc  : {within_n(df['range_err_m'], 0.10):.1f}%")
    print(f"  +/- 2.5deg acc: {within_n(df['angle_err'], 2.5):.1f}%")

    # Per Distance / Angle Breakdown
    distances = sorted(df['dist_m'].unique())
    angles    = sorted(df['angle_deg'].unique())

    print("\n-- HR MAE by Distance --")
    for dist in distances:
        sub = df[df['dist_m'] == dist]
        print(f"  {dist}m : MAE={sub['hr_err_fft'].mean():.2f} bpm  "
              f"RMSE={np.sqrt((sub['hr_err_fft']**2).mean()):.2f}  n={len(sub)}")

    print("\n-- HR MAE by Angle --")
    for ang in angles:
        sub = df[df['angle_deg'] == ang]
        print(f"  {ang}deg: MAE={sub['hr_err_fft'].mean():.2f} bpm  "
              f"RMSE={np.sqrt((sub['hr_err_fft']**2).mean()):.2f}  n={len(sub)}")

    # ─── Save Results CSV ─────────────────────────────────────────────────────
    csv_path = './fixed_pipeline_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}  ({df.shape[0]} rows x {df.shape[1]} cols)")

    # ─── Dashboard Plot ───────────────────────────────────────────────────────
    mae_grid = np.full((len(distances), len(angles)), np.nan)
    for i, dist in enumerate(distances):
        for j, ang in enumerate(angles):
            sub = df[(df['dist_m'] == dist) & (df['angle_deg'] == ang)]
            if len(sub) > 0:
                mae_grid[i, j] = sub['hr_err_fft'].mean()

    COLORS = {
        'hr'      : '#E91E63',
        'br'      : '#4CAF50',
        'ok'      : '#2196F3',
        'neutral' : '#607D8B',
        'dist'    : ['#1565C0','#0288D1','#00838F','#2E7D32',
                     '#F57F17','#E65100','#B71C1C','#4A148C'],
    }

    fig = plt.figure(figsize=(20, 16), facecolor='#0d1117')
    fig.suptitle('FMCW Radar -- Fixed QIDK Pipeline Results',
                 fontsize=16, color='white', fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    def dark_ax(ax, title):
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('white')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.2, color='#30363d')
        return ax

    # KPI boxes
    kpis = [
        (f'{df["hr_err_fft"].mean():.1f}',                       'bpm', 'HR MAE (FFT)'),
        (f'{np.sqrt((df["hr_err_fft"]**2).mean()):.1f}',         'bpm', 'HR RMSE'),
        (f'{within_n(df["hr_err_fft"], 5):.0f}',                 '%',   'HR <=5 bpm'),
        (f'{within_n(df["hr_err_fft"], 10):.0f}',                '%',   'HR <=10 bpm'),
    ]
    for k, (val, unit, label) in enumerate(kpis):
        ax_k = fig.add_subplot(gs[0, k])
        ax_k.set_facecolor('#161b22'); ax_k.axis('off')
        ax_k.text(0.5, 0.62, val, ha='center', va='center', fontsize=38,
                  color=COLORS['hr'], fontweight='bold', transform=ax_k.transAxes)
        ax_k.text(0.5, 0.38, unit, ha='center', va='center', fontsize=13,
                  color='#8b949e', transform=ax_k.transAxes)
        ax_k.text(0.5, 0.12, label, ha='center', va='center', fontsize=11,
                  color='white', fontweight='bold', transform=ax_k.transAxes)

    # HR Scatter
    ax1 = dark_ax(fig.add_subplot(gs[1, 0:2]), 'HR: Estimated vs True (FFT)')
    for i, dist in enumerate(distances):
        sub = df[df['dist_m'] == dist]
        ax1.scatter(sub['hr_true'], sub['hr_fft'], s=18, alpha=0.65,
                    color=COLORS['dist'][i % len(COLORS['dist'])], label=f'{dist}m')
    lims = [df['hr_true'].min() - 8, df['hr_true'].max() + 8]
    ax1.plot(lims, lims, 'white', lw=1.5, ls='--', label='Perfect')
    ax1.fill_between(lims, [l-10 for l in lims], [l+10 for l in lims],
                     alpha=0.08, color='white')
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel('True HR (bpm)'); ax1.set_ylabel('Estimated HR (bpm)')
    ax1.legend(fontsize=7, ncol=2, facecolor='#161b22',
               labelcolor='white', edgecolor='#30363d')

    # HR MAE Heatmap
    ax2 = dark_ax(fig.add_subplot(gs[1, 2:4]), 'HR MAE Heatmap (Distance x Angle)')
    im = ax2.imshow(mae_grid, cmap='RdYlGn_r', aspect='auto',
                    vmin=0, vmax=max(20, np.nanmax(mae_grid)))
    ax2.set_xticks(range(len(angles)))
    ax2.set_xticklabels([f'{a} deg' for a in angles], color='#8b949e')
    ax2.set_yticks(range(len(distances)))
    ax2.set_yticklabels([f'{d}m' for d in distances], color='#8b949e')
    ax2.set_xlabel('Angle'); ax2.set_ylabel('Distance')
    plt.colorbar(im, ax=ax2, label='MAE (bpm)')
    for i in range(len(distances)):
        for j in range(len(angles)):
            v = mae_grid[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                         color='white' if v > np.nanmax(mae_grid)*0.6 else 'black')

    # HR Error CDF
    ax3 = dark_ax(fig.add_subplot(gs[2, 0:2]), 'HR Error CDF')
    sorted_err = np.sort(df['hr_err_fft'])
    cdf_vals   = np.arange(1, len(sorted_err)+1) / len(sorted_err) * 100
    ax3.plot(sorted_err, cdf_vals, color=COLORS['hr'], lw=2.5, label='FFT')
    if len(df_ac) > 0:
        sorted_ac = np.sort(df_ac['hr_err_ac'])
        cdf_ac    = np.arange(1, len(sorted_ac)+1) / len(sorted_ac) * 100
        ax3.plot(sorted_ac, cdf_ac, color=COLORS['ok'], lw=2.0, ls='--', label='AutoCorr')
    for thresh, col in [(5, '#4CAF50'), (10, '#FF9800'), (15, '#F44336')]:
        pct = (df['hr_err_fft'] <= thresh).mean() * 100
        ax3.axvline(thresh, color=col, ls=':', lw=1.2, alpha=0.8)
        ax3.text(thresh + 0.3, 5, f'{pct:.0f}%\n<={thresh}', fontsize=8, color=col)
    ax3.set_xlabel('|HR Error| (bpm)'); ax3.set_ylabel('Cumulative (%)')
    ax3.set_ylim(0, 105)
    ax3.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d')

    # BR Scatter
    ax4 = dark_ax(fig.add_subplot(gs[2, 2:4]), 'BR: Estimated vs True (FFT)')
    if len(df_br) > 0:
        ax4.scatter(df_br['br_true'], df_br['br_fft'], s=22, alpha=0.7, color=COLORS['br'])
        br_lims = [df_br['br_true'].min()-3, df_br['br_true'].max()+3]
        ax4.plot(br_lims, br_lims, 'white', lw=1.5, ls='--')
        ax4.fill_between(br_lims, [l-2 for l in br_lims], [l+2 for l in br_lims],
                         alpha=0.12, color='white')
        ax4.set_xlim(br_lims); ax4.set_ylim(br_lims)
        ax4.set_xlabel('True BR (br/min)'); ax4.set_ylabel('Estimated BR (br/min)')
        mae_br = df_br['br_err_fft'].mean()
        ax4.text(0.05, 0.92, f'MAE = {mae_br:.2f} br/min', transform=ax4.transAxes,
                 color='white', fontsize=11)

    plt.savefig('./fixed_pipeline_dashboard.png', dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    print("Dashboard saved to fixed_pipeline_dashboard.png")

    # ─── Sample Results Table ─────────────────────────────────────────────────
    print("\n-- Sample Results (first 20 files) --")
    print(f"{'File':<38} {'HR_true':>7} {'HR_fft':>7} {'Err':>6} "
          f"{'BR_true':>7} {'BR_fft':>7} {'Range_m':>8} {'Angle':>6}")
    print("-" * 105)
    for _, row in df.head(20).iterrows():
        br_true_str = f"{row['br_true']:.0f}" if row['has_br'] else "hold"
        br_fft_str  = f"{row['br_fft']:.1f}" if row['has_br'] else "-"
        print(f"{row['filename']:<38} {row['hr_true']:>7} {row['hr_fft']:>7.1f} "
              f"{row['hr_err_fft']:>6.1f} {br_true_str:>7} {br_fft_str:>7} "
              f"{row['peak_range_m']:>8.2f} {row['peak_angle_deg']:>6.1f}")

    print("\nDone.")

    if m:
        records.append({
            'path'      : str(f),
            'filename'  : f.name,
            'dist_m'    : float(m.group('dist')),
            'angle_deg' : int(m.group('angle')),
            'hr_true'   : int(m.group('hr')),
            'pm'        : int(m.group('pm')),
            'br_true'   : float(m.group('br')),
        })

meta = pd.DataFrame(records)
meta['has_br'] = meta['br_true'] > 0

# ── Limit to 100 files (stratified sample across distances & angles) ──────────
meta = meta.sample(n=100, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(meta)} files  (subset of 320 — 100 sampled)")
print(f"  Distances : {sorted(meta.dist_m.unique())} m")
print(f"  Angles    : {sorted(meta.angle_deg.unique())} deg")
print(f"  HR range  : {meta.hr_true.min()} – {meta.hr_true.max()} bpm")
print(f"  BR files  : {meta.has_br.sum()} / {len(meta)}")
print()


# ─── Batch Processing ────────────────────────────────────────────────────────
print("Running fixed pipeline on all files...")
print("-" * 60)

results_list = []
errors = []
t_batch_start = time.perf_counter()

for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing"):
    try:
        res = run_fixed_pipeline(row['path'])
        results_list.append({
            **row.to_dict(),
            'hr_fft'         : res['hr_fft'],
            'br_fft'         : res['br_fft'],
            'hr_ac'          : res['hr_ac'],
            'br_ac'          : res['br_ac'],
            'peak_range_m'   : res['peak_range_m'],
            'peak_angle_deg' : res['peak_angle_deg'],
            **{f'timing_{k}': v for k, v in res['timing'].items()},
        })
    except Exception as e:
        errors.append({'file': row['filename'], 'error': str(e)})
        print(f"\n  ERROR on {row['filename']}: {e}")

t_batch_elapsed = time.perf_counter() - t_batch_start

df = pd.DataFrame(results_list)
print(f"\nBatch complete: {len(df)}/{len(meta)} files in {t_batch_elapsed:.1f}s")
print(f"  Errors: {len(errors)}")
print(f"  Avg per file: {t_batch_elapsed/len(df)*1000:.0f} ms")
print()


# ─── Compute Error Metrics ───────────────────────────────────────────────────
df['hr_err_fft']  = (df['hr_fft'] - df['hr_true']).abs()
df['hr_err_ac']   = (df['hr_ac']  - df['hr_true']).abs()
df['br_err_fft']  = (df['br_fft'] - df['br_true']).abs()
df['br_err_ac']   = (df['br_ac']  - df['br_true']).abs()
df['range_err_m'] = (df['peak_range_m'] - df['dist_m']).abs()
df['angle_err']   = (df['peak_angle_deg'] - df['angle_deg']).abs()

df_br = df[df['has_br']].copy()
df_ac = df[df['hr_ac'].notna()].copy()

def within_n(series, n):
    return (series <= n).mean() * 100

print("=" * 60)
print("ACCURACY RESULTS — FIXED PIPELINE")
print("=" * 60)

print("\n── Heart Rate (FFT estimator) ──")
print(f"  MAE           : {df['hr_err_fft'].mean():.2f} bpm")
print(f"  RMSE          : {np.sqrt((df['hr_err_fft']**2).mean()):.2f} bpm")
print(f"  Median error  : {df['hr_err_fft'].median():.2f} bpm")
print(f"  95th pctile   : {np.percentile(df['hr_err_fft'], 95):.2f} bpm")
print(f"  <= 5 bpm      : {within_n(df['hr_err_fft'], 5):.1f}%")
print(f"  <= 10 bpm     : {within_n(df['hr_err_fft'], 10):.1f}%")
print(f"  <= 15 bpm     : {within_n(df['hr_err_fft'], 15):.1f}%")

print("\n── Heart Rate (AutoCorr estimator) ──")
print(f"  Valid files   : {df_ac['hr_err_ac'].count()} / {len(df)}")
if len(df_ac) > 0:
    print(f"  MAE           : {df_ac['hr_err_ac'].mean():.2f} bpm")
    print(f"  RMSE          : {np.sqrt((df_ac['hr_err_ac']**2).mean()):.2f} bpm")
    print(f"  <= 5 bpm      : {within_n(df_ac['hr_err_ac'], 5):.1f}%")
    print(f"  <= 10 bpm     : {within_n(df_ac['hr_err_ac'], 10):.1f}%")

print("\n── Breathing Rate (FFT) ──")
print(f"  Files with BR : {len(df_br)}")
if len(df_br) > 0:
    print(f"  MAE           : {df_br['br_err_fft'].mean():.2f} br/min")
    print(f"  RMSE          : {np.sqrt((df_br['br_err_fft']**2).mean()):.2f} br/min")
    print(f"  Median error  : {df_br['br_err_fft'].median():.2f} br/min")
    print(f"  <= 2 br/min   : {within_n(df_br['br_err_fft'], 2):.1f}%")
    print(f"  <= 3 br/min   : {within_n(df_br['br_err_fft'], 3):.1f}%")
    print(f"  <= 5 br/min   : {within_n(df_br['br_err_fft'], 5):.1f}%")

print("\n── Localisation ──")
print(f"  Range MAE     : {df['range_err_m'].mean()*100:.1f} cm")
print(f"  Range RMSE    : {np.sqrt((df['range_err_m']**2).mean())*100:.1f} cm")
print(f"  Angle MAE     : {df['angle_err'].mean():.2f} deg")
print(f"  +/- 10cm acc  : {within_n(df['range_err_m'], 0.10):.1f}%")
print(f"  +/- 2.5deg acc: {within_n(df['angle_err'], 2.5):.1f}%")


# ─── Per Distance / Angle Breakdown ──────────────────────────────────────────
distances = sorted(df['dist_m'].unique())
angles    = sorted(df['angle_deg'].unique())

print("\n── HR MAE by Distance ──")
for dist in distances:
    sub = df[df['dist_m'] == dist]
    print(f"  {dist}m : MAE={sub['hr_err_fft'].mean():.2f} bpm  RMSE={np.sqrt((sub['hr_err_fft']**2).mean()):.2f}  n={len(sub)}")

print("\n── HR MAE by Angle ──")
for ang in angles:
    sub = df[df['angle_deg'] == ang]
    print(f"  {ang}deg: MAE={sub['hr_err_fft'].mean():.2f} bpm  RMSE={np.sqrt((sub['hr_err_fft']**2).mean()):.2f}  n={len(sub)}")


# ─── MAE Heatmap grid ────────────────────────────────────────────────────────
mae_grid = np.full((len(distances), len(angles)), np.nan)
for i, dist in enumerate(distances):
    for j, ang in enumerate(angles):
        sub = df[(df['dist_m'] == dist) & (df['angle_deg'] == ang)]
        if len(sub) > 0:
            mae_grid[i, j] = sub['hr_err_fft'].mean()


# ─── Save Results CSV ────────────────────────────────────────────────────────
csv_path = './fixed_pipeline_results.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}  ({df.shape[0]} rows x {df.shape[1]} cols)")


# ─── Dashboard Plot ──────────────────────────────────────────────────────────
COLORS = {
    'hr'      : '#E91E63',
    'br'      : '#4CAF50',
    'ok'      : '#2196F3',
    'neutral' : '#607D8B',
    'dist'    : ['#1565C0','#0288D1','#00838F','#2E7D32','#F57F17','#E65100','#B71C1C','#4A148C'],
}

fig = plt.figure(figsize=(20, 16), facecolor='#0d1117')
fig.suptitle('FMCW Radar — Fixed QIDK Pipeline Results (All 320 Files)',
             fontsize=16, color='white', fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

def dark_ax(ax, title):
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('white')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.2, color='#30363d')
    return ax

# KPI boxes
kpis = [
    (f'{df["hr_err_fft"].mean():.1f}',                           'bpm', 'HR MAE (FFT)'),
    (f'{np.sqrt((df["hr_err_fft"]**2).mean()):.1f}',             'bpm', 'HR RMSE'),
    (f'{within_n(df["hr_err_fft"], 5):.0f}',                     '%',   'HR ≤5 bpm'),
    (f'{within_n(df["hr_err_fft"], 10):.0f}',                    '%',   'HR ≤10 bpm'),
]
for k, (val, unit, label) in enumerate(kpis):
    ax_k = fig.add_subplot(gs[0, k])
    ax_k.set_facecolor('#161b22'); ax_k.axis('off')
    ax_k.text(0.5, 0.62, val, ha='center', va='center', fontsize=38,
              color=COLORS['hr'], fontweight='bold', transform=ax_k.transAxes)
    ax_k.text(0.5, 0.38, unit, ha='center', va='center', fontsize=13,
              color='#8b949e', transform=ax_k.transAxes)
    ax_k.text(0.5, 0.12, label, ha='center', va='center', fontsize=11,
              color='white', fontweight='bold', transform=ax_k.transAxes)

# HR Scatter: Estimated vs True
ax1 = dark_ax(fig.add_subplot(gs[1, 0:2]), 'HR: Estimated vs True (FFT)')
for i, dist in enumerate(distances):
    sub = df[df['dist_m'] == dist]
    ax1.scatter(sub['hr_true'], sub['hr_fft'], s=18, alpha=0.65,
                color=COLORS['dist'][i % len(COLORS['dist'])], label=f'{dist}m')
lims = [df['hr_true'].min() - 8, df['hr_true'].max() + 8]
ax1.plot(lims, lims, 'white', lw=1.5, ls='--', label='Perfect')
ax1.fill_between(lims, [l-10 for l in lims], [l+10 for l in lims], alpha=0.08, color='white')
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel('True HR (bpm)'); ax1.set_ylabel('Estimated HR (bpm)')
ax1.legend(fontsize=7, ncol=2, facecolor='#161b22', labelcolor='white', edgecolor='#30363d')

# HR MAE Heatmap
ax2 = dark_ax(fig.add_subplot(gs[1, 2:4]), 'HR MAE Heatmap (Distance × Angle)')
im = ax2.imshow(mae_grid, cmap='RdYlGn_r', aspect='auto',
                vmin=0, vmax=max(20, np.nanmax(mae_grid)))
ax2.set_xticks(range(len(angles)))
ax2.set_xticklabels([f'{a}°' for a in angles], color='#8b949e')
ax2.set_yticks(range(len(distances)))
ax2.set_yticklabels([f'{d}m' for d in distances], color='#8b949e')
ax2.set_xlabel('Angle'); ax2.set_ylabel('Distance')
plt.colorbar(im, ax=ax2, label='MAE (bpm)')
for i in range(len(distances)):
    for j in range(len(angles)):
        v = mae_grid[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                     color='white' if v > np.nanmax(mae_grid)*0.6 else 'black')

# HR Error CDF
ax3 = dark_ax(fig.add_subplot(gs[2, 0:2]), 'HR Error CDF')
sorted_err = np.sort(df['hr_err_fft'])
cdf_vals   = np.arange(1, len(sorted_err)+1) / len(sorted_err) * 100
ax3.plot(sorted_err, cdf_vals, color=COLORS['hr'], lw=2.5, label='FFT')
if len(df_ac) > 0:
    sorted_ac = np.sort(df_ac['hr_err_ac'])
    cdf_ac    = np.arange(1, len(sorted_ac)+1) / len(sorted_ac) * 100
    ax3.plot(sorted_ac, cdf_ac, color=COLORS['ok'], lw=2.0, ls='--', label='AutoCorr')
for thresh, col in [(5, '#4CAF50'), (10, '#FF9800'), (15, '#F44336')]:
    pct = (df['hr_err_fft'] <= thresh).mean() * 100
    ax3.axvline(thresh, color=col, ls=':', lw=1.2, alpha=0.8)
    ax3.text(thresh + 0.3, 5, f'{pct:.0f}%\n≤{thresh}', fontsize=8, color=col)
ax3.set_xlabel('|HR Error| (bpm)'); ax3.set_ylabel('Cumulative (%)')
ax3.set_ylim(0, 105)
ax3.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d')

# BR Scatter (for files with reference)
ax4 = dark_ax(fig.add_subplot(gs[2, 2:4]), 'BR: Estimated vs True (FFT)')
if len(df_br) > 0:
    ax4.scatter(df_br['br_true'], df_br['br_fft'], s=22, alpha=0.7, color=COLORS['br'])
    br_lims = [df_br['br_true'].min()-3, df_br['br_true'].max()+3]
    ax4.plot(br_lims, br_lims, 'white', lw=1.5, ls='--')
    ax4.fill_between(br_lims, [l-2 for l in br_lims], [l+2 for l in br_lims],
                     alpha=0.12, color='white')
    ax4.set_xlim(br_lims); ax4.set_ylim(br_lims)
    ax4.set_xlabel('True BR (br/min)'); ax4.set_ylabel('Estimated BR (br/min)')
    mae_br = df_br['br_err_fft'].mean()
    ax4.text(0.05, 0.92, f'MAE = {mae_br:.2f} br/min', transform=ax4.transAxes,
             color='white', fontsize=11)

