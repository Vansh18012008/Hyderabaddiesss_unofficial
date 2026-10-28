#!/usr/bin/env python3
"""
run_folder.py — Full FMCW QIDK Pipeline for an Entire Folder of .bin Files
IC~WiBES Research Group

Usage:
    python3 run_folder.py ./Single_Subject/D1m_0d/     # one location folder
    python3 run_folder.py ./Single_Subject/             # all locations

Outputs:
  results/          — CSV with all metrics
  results/figures/  — all per-file + aggregate PNGs
  output_report.ipynb  — final notebook with all graphs (no code cells)
"""

import sys, os, re, time, warnings, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('.'))

from dc_offset_correction import circle_fit_correction
from phase_demodulation import demodulate
from vital_sign_extraction import extract_vital_signs

# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE CONSTANTS (AWR1843 / DCA1000)  — DO NOT CHANGE
# ══════════════════════════════════════════════════════════════════════════════
f_min     = 77e9
f_max     = 79.982e9
c         = 3e8
BW        = f_max - f_min
Tm        = 100e-6
f_adc     = 2e6                              # IF bandwidth → R_max ≈ 5 m
R_max     = (c * Tm / 2) * (f_adc / (2 * BW))
lambda_c  = c / ((f_min + f_max) / 2)
lambda_max = c / f_min
d         = lambda_max / 2 * 250             # antenna spacing
fs_slow   = 20.0                             # frame repetition rate [Hz]

N_chirps    = 512
M_chirps    = 512
NUM_RX      = 4
N_FFT_RANGE = 512
MIN_RANGE_M = 0.5                            # skip near-field artifact bins
SKIP_S      = 3                              # seconds of startup transient to skip

azimuth   = np.arange(90, -90 - 2.5, -2.5)
num_angles = len(azimuth)
range_axis = np.linspace(0, R_max, N_FFT_RANGE // 2)
win_range  = np.hanning(N_chirps)

plt.rcParams.update({
    'figure.dpi': 110, 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'legend.fontsize': 9, 'lines.linewidth': 1.6,
    'axes.grid': True, 'grid.alpha': 0.3,
})

COLORS = {
    'raw'     : '#607D8B', 'hr'       : '#E91E63', 'br'  : '#4CAF50',
    'svd'     : '#2E7D32', 'resp'     : '#4CAF50', 'heart': '#E91E63',
    'circle'  : '#E91E63', 'mean_sub' : '#1565C0',
}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def readDCA1000(path):
    with open(path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.int16)
    I = raw[0::4] + 1j * raw[2::4]
    Q = raw[1::4] + 1j * raw[3::4]
    LVDS = np.empty(I.size + Q.size, dtype=np.complex64)
    LVDS[0::2] = I; LVDS[1::2] = Q
    numChirps = raw.shape[0] // (2 * 512 * 4)
    LVDS = LVDS.reshape((512 * 4, numChirps), order='F').T
    out = np.zeros((4, numChirps * 512), dtype=LVDS.dtype)
    for r in range(4):
        for i in range(numChirps):
            out[r, i*512:(i+1)*512] = LVDS[i, r*512:(r+1)*512]
    return out

def beamform(if_data):
    rx_pos = np.arange(NUM_RX) * d
    ang = np.deg2rad(azimuth)
    W = np.exp(-1j * 2 * np.pi * rx_pos[:, None] * np.sin(ang)[None, :]) / NUM_RX
    return np.einsum('ra,rcs->acs', W, if_data)

def fast_rank1(mat):
    """Power-iteration rank-1 clutter removal (~44x faster than full SVD)."""
    v = np.sum(np.abs(mat), axis=0).astype(np.complex64)
    nv = np.linalg.norm(v)
    if nv < 1e-10:
        return mat.copy(), 0.0
    v /= nv; sigma = 0.0; u = np.zeros(mat.shape[0], dtype=np.complex64)
    for _ in range(5):
        u = mat @ v.conj(); nu = np.linalg.norm(u)
        if nu < 1e-10: break
        u /= nu; v = mat.conj().T @ u; sigma = np.linalg.norm(v)
        if sigma < 1e-10: break
        v /= sigma
    return (mat - np.outer(u * sigma, v.conj())).astype(np.complex64), sigma

def autocorr_hr(sig, fs):
    x = sig - np.mean(sig)
    r = np.correlate(x, x, mode='full'); r = r[r.size//2:]; r /= (r[0] + 1e-30)
    lags_s = np.arange(len(r)) / fs
    peaks, _ = find_peaks(r)
    if len(peaks) == 0: return None
    T = lags_s[peaks][0]
    return (1.0 / T) * 60 if T > 0 else None

FNAME_RE = re.compile(
    r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin'
)

def parse_meta(path):
    m = FNAME_RE.match(Path(path).name)
    if not m: return None
    return {
        'path': str(path), 'filename': Path(path).name,
        'dist_m': float(m.group('dist')), 'angle_deg': int(m.group('angle')),
        'hr_true': int(m.group('hr')), 'pm': int(m.group('pm')),
        'br_true': float(m.group('br')), 'has_br': float(m.group('br')) > 0,
    }

# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(path):
    if_data = readDCA1000(path).reshape(NUM_RX, M_chirps, N_chirps).astype(np.complex64)
    res = beamform(if_data)

    # Stage 1a: Mean subtraction per angle
    res_dc = np.zeros_like(res, dtype=np.complex64)
    for nn in range(num_angles):
        m = res[nn]
        res_dc[nn] = (m - np.mean(m, axis=0, keepdims=True)).astype(np.complex64)

    # Stage 2: Fast rank-1 SVD clutter removal per angle
    res_clean = np.zeros_like(res_dc, dtype=np.complex64)
    svd_sigmas = np.zeros(num_angles)
    for nn in range(num_angles):
        cleaned, sigma = fast_rank1(res_dc[nn])
        res_clean[nn] = cleaned; svd_sigmas[nn] = sigma

    # Range FFT
    rfft_cube = np.zeros((num_angles, M_chirps, N_FFT_RANGE // 2), dtype=np.complex64)
    rfft_mean = np.zeros((num_angles, N_FFT_RANGE // 2))
    for nn in range(num_angles):
        X = np.fft.fft(res_clean[nn] * win_range[np.newaxis, :], n=N_FFT_RANGE, axis=1)[:, :N_FFT_RANGE // 2]
        rfft_cube[nn] = X.astype(np.complex64)
        rfft_mean[nn] = np.mean(np.abs(X), axis=0)

    # Target detection (skip near-field bins < MIN_RANGE_M)
    min_bin = int(MIN_RANGE_M / R_max * (N_FFT_RANGE // 2))
    masked = rfft_mean.copy(); masked[:, :min_bin] = 0
    pf = np.argmax(masked)
    pa, pr = pf // (N_FFT_RANGE // 2), pf % (N_FFT_RANGE // 2)
    peak_range_m = range_axis[pr]; peak_angle_deg = azimuth[pa]

    # Stage 1b: Circle-fit DC at target bin
    slow_iq = rfft_cube[pa, :, pr]
    slow_iq_corr, _ = circle_fit_correction(slow_iq)

    # Phase demodulation
    demod = demodulate(slow_iq_corr, wavelength_m=lambda_c, remove_dc_in_phase=True)
    disp_full = demod['displacement_mm']

    # Skip startup transient + detrend
    skip = int(SKIP_S * fs_slow)
    disp = disp_full[skip:]
    t_arr = np.arange(len(disp))
    poly = np.polyfit(t_arr, disp, 1)
    disp = disp - np.polyval(poly, t_arr)
    t_plot = t_arr / fs_slow  # seconds

    # Vital sign extraction — improved with respiration harmonic suppression
    vs = extract_vital_signs(disp, fs=fs_slow,
                             resp_band=(0.1, 0.5), heart_band=(0.8, 2.5), estimator='fft')

    # ── Improved HR: suppress respiration harmonics before finding HR peak ──
    br_fft_hz = vs['resp_rate_bpm'] / 60.0          # estimated BR in Hz
    hf, hm = vs['heart_spectrum']                    # (freqs, magnitudes)
    hm_clean = hm.copy()
    if br_fft_hz > 0.05:                             # valid BR detected
        notch_bw = 0.04                              # ±0.04 Hz notch around each harmonic
        for k in range(2, 10):
            h_hz = br_fft_hz * k
            if h_hz > 2.5: break
            mask = np.abs(hf - h_hz) < notch_bw
            hm_clean[mask] = 0.0

    # Find HR peak in harmonic-suppressed spectrum (only in 0.8–2.5 Hz)
    valid = (hf >= 0.8) & (hf <= 2.5)
    if valid.any() and hm_clean[valid].max() > 0:
        hr_fft_hz = hf[valid][np.argmax(hm_clean[valid])]
        hr_fft = hr_fft_hz * 60.0
    else:
        hr_fft = vs['heart_rate_bpm']               # fallback to raw FFT

    # AutoCorr estimators
    hr_ac = autocorr_hr(vs['heart_signal_mm'], fs_slow)
    br_ac = autocorr_hr(vs['resp_signal_mm'],  fs_slow)

    # Use AC for HR if it's within ±15 bpm of FFT estimate and both are physiological
    if hr_ac and 48 <= hr_ac <= 150:
        if abs(hr_ac - hr_fft) <= 15:
            hr_fft = (hr_fft + hr_ac) / 2.0         # blend FFT + AC

    # Raw range profile at peak angle (for comparison plots)
    rp_raw = np.fft.fft(res[pa] * win_range[np.newaxis, :], n=N_FFT_RANGE, axis=1)[:, :N_FFT_RANGE // 2]
    rp_cln = rfft_cube[pa]

    return {
        'hr_fft': hr_fft, 'br_fft': vs['resp_rate_bpm'],
        'hr_ac': hr_ac, 'br_ac': br_ac,
        'peak_range_m': peak_range_m, 'peak_angle_deg': peak_angle_deg, 'peak_angle_idx': pa,
        'disp_mm': disp, 't_plot': t_plot,
        'resp_signal': vs['resp_signal_mm'], 'heart_signal': vs['heart_signal_mm'],
        'resp_spectrum': (hf, hm_clean),             # show harmonic-suppressed spectrum
        'heart_spectrum': vs['heart_spectrum'],
        'ra_heatmap': rfft_mean, 'slow_iq_raw': slow_iq, 'slow_iq_corr': slow_iq_corr,
        'rp_raw': rp_raw, 'rp_cln': rp_cln, 'svd_sigmas': svd_sigmas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS  (matching FMCW_Radar_Complete_Pipeline.ipynb sections)
# ══════════════════════════════════════════════════════════════════════════════
def savefig(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return str(path)

# ── Per-file figures ────────────────────────────────────────────────────────

def fig_raw_and_rst(r, meta, tag, od):
    avg_raw = 10*np.log10(np.mean(np.abs(r['rp_raw'])**2, axis=0) + 1e-30)
    avg_cln = 10*np.log10(np.mean(np.abs(r['rp_cln'])**2, axis=0) + 1e-30)
    rst_raw = 10*np.log10(np.abs(r['rp_raw'])**2 + 1e-30)
    rst_cln = 10*np.log10(np.abs(r['rp_cln'])**2 + 1e-30)
    t_rst   = np.arange(M_chirps) / fs_slow

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), tight_layout=True)

    # Raw beat signal
    axes[0,0].plot(np.real(r['rp_raw'][0]), color='steelblue', lw=0.9, alpha=0.85)
    axes[0,0].set(title='Raw ADC Beat Signal (Chirp 0)', xlabel='Sample index', ylabel='Amplitude')

    # Range profile before/after
    axes[0,1].plot(range_axis, avg_raw, color=COLORS['raw'], lw=1.5, alpha=0.7, label='Raw')
    axes[0,1].plot(range_axis, avg_cln, color=COLORS['svd'], lw=1.8, label='After SVD [12]')
    axes[0,1].axvline(meta['dist_m'], color='red', ls='--', lw=2, label=f'True {meta["dist_m"]}m')
    axes[0,1].axvline(r['peak_range_m'], color='cyan', ls=':', lw=1.5, label=f'Detected {r["peak_range_m"]:.2f}m')
    axes[0,1].set(xlabel='Range [m]', ylabel='Power [dB]', title='Range Profile — Before vs After Clutter Removal')
    axes[0,1].legend(fontsize=8)

    # RST heatmap — raw
    im0 = axes[1,0].pcolormesh(range_axis, t_rst, rst_raw, cmap='viridis', shading='auto', vmin=-80)
    axes[1,0].axvline(meta['dist_m'], color='red', ls='--', lw=2)
    fig.colorbar(im0, ax=axes[1,0], label='Power [dB]')
    axes[1,0].set(xlabel='Range [m]', ylabel='Slow Time [s]', title='Range-Slow-Time Heatmap (Raw) [14]')

    # RST heatmap — cleaned
    im1 = axes[1,1].pcolormesh(range_axis, t_rst, rst_cln, cmap='viridis', shading='auto', vmin=-80)
    axes[1,1].axvline(meta['dist_m'], color='red', ls='--', lw=2)
    axes[1,1].axvline(r['peak_range_m'], color='cyan', ls=':', lw=1.5)
    fig.colorbar(im1, ax=axes[1,1], label='Power [dB]')
    axes[1,1].set(xlabel='Range [m]', ylabel='Slow Time [s]', title='Range-Slow-Time Heatmap (After Stage1a+SVD) [12,14]')

    fig.suptitle(f'[Sec 15–17] Raw Signal & Clutter Suppression — {tag}', fontsize=13, fontweight='bold')
    return savefig(fig, od / f'{tag}_fig1_raw_rst.png')

def fig_svd_and_ra(r, meta, tag, od):
    pa = r['peak_angle_idx']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

    # SVD dominant singular value vs azimuth
    axes[0].bar(np.arange(num_angles), r['svd_sigmas'], color='steelblue', edgecolor='k', lw=0.4)
    axes[0].axvline(pa, color='red', ls='--', lw=2, label=f'Target bin {pa} ({azimuth[pa]:.0f}°)')
    axes[0].set(title='SVD Dominant Singular Value vs Azimuth [12]', xlabel='Angle bin', ylabel='σ₁')
    axes[0].legend()

    # Range-Azimuth map (zoom)
    data = np.abs(r['ra_heatmap']).T
    im1 = axes[1].imshow(data, extent=[azimuth[-1], azimuth[0], 0, R_max],
                         aspect='auto', origin='lower', cmap='jet')
    axes[1].set_ylim(0, min(R_max, meta['dist_m'] * 2))
    axes[1].axhline(r['peak_range_m'], color='white', ls='--', lw=2, label=f'Det {r["peak_range_m"]:.2f}m')
    axes[1].axhline(meta['dist_m'], color='red', ls=':', lw=2, label=f'True {meta["dist_m"]}m')
    axes[1].axvline(r['peak_angle_deg'], color='white', ls='--', lw=1.5)
    plt.colorbar(im1, ax=axes[1], label='Mean |FFT|')
    axes[1].set(title='Range-Azimuth Heatmap (zoomed) [18]', xlabel='Azimuth (°)', ylabel='Range (m)')
    axes[1].legend(fontsize=8)

    # Range-Azimuth map (full)
    im2 = axes[2].imshow(data, extent=[azimuth[-1], azimuth[0], 0, R_max],
                         aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(im2, ax=axes[2], label='Mean |FFT|')
    axes[2].set(title='Range-Azimuth Heatmap (full range) [18]', xlabel='Azimuth (°)', ylabel='Range (m)')

    fig.suptitle(f'[Sec 17–18] SVD Analysis & Range-Azimuth Map — {tag}', fontsize=13, fontweight='bold')
    return savefig(fig, od / f'{tag}_fig2_svd_ra.png')

def fig_iq_and_displacement(r, meta, tag, od):
    tv = r['t_plot']
    fig = plt.figure(figsize=(16, 10), tight_layout=True)
    gs_ = gridspec.GridSpec(2, 3, figure=fig)

    # IQ before
    ax1 = fig.add_subplot(gs_[0, 0])
    ax1.scatter(np.real(r['slow_iq_raw']), np.imag(r['slow_iq_raw']), s=5, alpha=0.5, color=COLORS['raw'])
    ax1.axhline(0, color='k', lw=0.8, ls='--'); ax1.axvline(0, color='k', lw=0.8, ls='--')
    ax1.plot(0, 0, 'r+', ms=14, markeredgewidth=2.5)
    ax1.set_aspect('equal'); ax1.set(title='IQ — Before Circle-Fit [12]', xlabel='I', ylabel='Q')

    # IQ after
    ax2 = fig.add_subplot(gs_[0, 1])
    ax2.scatter(np.real(r['slow_iq_raw']), np.imag(r['slow_iq_raw']), s=4, alpha=0.25, color=COLORS['raw'], label='Before')
    ax2.scatter(np.real(r['slow_iq_corr']), np.imag(r['slow_iq_corr']), s=4, alpha=0.7, color=COLORS['circle'], label='After [12]')
    ax2.axhline(0, color='k', lw=0.8, ls='--'); ax2.axvline(0, color='k', lw=0.8, ls='--')
    ax2.plot(0, 0, 'r+', ms=14, markeredgewidth=2.5)
    ax2.set_aspect('equal'); ax2.legend(fontsize=8)
    ax2.set(title='IQ — After Circle-Fit DC Correction [12]', xlabel='I', ylabel='Q')

    # Full displacement
    ax3 = fig.add_subplot(gs_[0, 2])
    ax3.plot(tv, r['disp_mm'], color='#9C27B0', lw=0.9)
    ax3.fill_between(tv, r['disp_mm'], alpha=0.12, color='#9C27B0')
    ax3.axhline(0, color='k', lw=0.5)
    ax3.set(title='Chest Displacement (detrended) [14]', xlabel='Time [s]', ylabel='Disp [mm]')

    # Respiration band
    ax4 = fig.add_subplot(gs_[1, 0])
    ax4.plot(tv, r['resp_signal'], color=COLORS['resp'], lw=1.0)
    br_str = f'True: {meta["br_true"]:.1f}' if meta['has_br'] else 'Breath-hold'
    ax4.set(title=f'Respiration Band → {r["br_fft"]:.1f} br/min\n({br_str})', xlabel='Time [s]', ylabel='mm')

    # Heartbeat band
    ax5 = fig.add_subplot(gs_[1, 1])
    ax5.plot(tv, r['heart_signal'], color=COLORS['heart'], lw=1.0)
    ax5.set(title=f'Heartbeat Band → {r["hr_fft"]:.1f} bpm\n(True: {meta["hr_true"]} bpm)', xlabel='Time [s]', ylabel='mm')

    # Summary box
    ax6 = fig.add_subplot(gs_[1, 2]); ax6.axis('off')
    hr_ac = f'{r["hr_ac"]:.1f}' if r['hr_ac'] else 'N/A'
    txt = (f"TARGET DETECTION\n{'─'*26}\n"
           f"Detected range : {r['peak_range_m']:.2f} m\n"
           f"True range     : {meta['dist_m']} m\n"
           f"Angle detected : {r['peak_angle_deg']:.1f}°\n"
           f"True angle     : {meta['angle_deg']}°\n"
           f"{'─'*26}\n"
           f"HR FFT  : {r['hr_fft']:.1f} bpm\n"
           f"HR AC   : {hr_ac} bpm\n"
           f"HR True : {meta['hr_true']} bpm\n"
           f"HR Err  : {abs(r['hr_fft']-meta['hr_true']):.1f} bpm\n"
           f"{'─'*26}\n"
           f"BR FFT  : {r['br_fft']:.1f} br/min\n"
           f"BR True : {meta['br_true'] if meta['has_br'] else 'N/A'}")
    ax6.text(0.05, 0.95, txt, transform=ax6.transAxes, fontsize=9, va='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))

    fig.suptitle(f'[Sec 19–20] IQ Correction, Phase Demodulation & Vital Signs — {tag}', fontsize=12, fontweight='bold')
    return savefig(fig, od / f'{tag}_fig3_iq_disp.png')

def fig_spectra(r, meta, tag, od):
    rf, rm = r['resp_spectrum']; hf, hm = r['heart_spectrum']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

    mr = (rf >= 0.05) & (rf <= 0.6)
    axes[0].fill_between(rf[mr]*60, rm[mr], alpha=0.3, color=COLORS['resp'])
    axes[0].plot(rf[mr]*60, rm[mr], color=COLORS['resp'], lw=1.8)
    axes[0].axvline(r['br_fft'], color='blue', ls='--', lw=2, label=f'Est: {r["br_fft"]:.1f} br/min')
    if meta['has_br']:
        axes[0].axvline(meta['br_true'], color='red', ls=':', lw=2, label=f'True: {meta["br_true"]:.1f}')
    axes[0].set(title='Respiration Spectrum [14]', xlabel='Rate (breaths/min)', ylabel='Magnitude')
    axes[0].legend()

    mh = (hf >= 0.7) & (hf <= 2.6)
    axes[1].fill_between(hf[mh]*60, hm[mh], alpha=0.3, color=COLORS['heart'])
    axes[1].plot(hf[mh]*60, hm[mh], color=COLORS['heart'], lw=1.8)
    axes[1].axvline(r['hr_fft'], color='blue', ls='--', lw=2, label=f'Est: {r["hr_fft"]:.1f} bpm')
    axes[1].axvline(meta['hr_true'], color='red', ls=':', lw=2, label=f'True: {meta["hr_true"]} bpm')
    err = abs(r['hr_fft'] - meta['hr_true'])
    axes[1].set(title=f'Heart Rate Spectrum [14] — MAE={err:.1f} bpm', xlabel='Rate (bpm)', ylabel='Magnitude')
    axes[1].legend()

    fig.suptitle(f'[Sec 20] Vital Sign Spectra — {tag}', fontsize=12, fontweight='bold')
    return savefig(fig, od / f'{tag}_fig4_spectra.png')

def fig_dashboard(r, meta, tag, od):
    """Final pipeline dashboard matching Cell 29/48 of FMCW_Radar_Complete_Pipeline.ipynb."""
    rf, rm = r['resp_spectrum']; hf, hm = r['heart_spectrum']
    tv = r['t_plot']
    avg_raw = 10*np.log10(np.mean(np.abs(r['rp_raw'])**2, axis=0) + 1e-30)
    avg_cln = 10*np.log10(np.mean(np.abs(r['rp_cln'])**2, axis=0) + 1e-30)

    fig = plt.figure(figsize=(16, 10))
    gs_ = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(gs_[0, 0])
    ax1.plot(range_axis, avg_raw, color=COLORS['raw'], lw=1.5, alpha=0.7, label='Raw')
    ax1.plot(range_axis, avg_cln, color=COLORS['svd'], lw=1.8, label='SVD cleaned [12]')
    ax1.axvline(meta['dist_m'], color='red', ls='--', lw=1.5, label=f'Target {meta["dist_m"]}m')
    ax1.set(title='Steps 2–3: Range Profile\nBefore vs After Clutter Removal', xlabel='Range [m]', ylabel='Power [dB]')
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs_[0, 1])
    ax2.scatter(np.real(r['slow_iq_raw']), np.imag(r['slow_iq_raw']), s=5, alpha=0.3, color=COLORS['raw'], label='Before DC')
    ax2.scatter(np.real(r['slow_iq_corr']), np.imag(r['slow_iq_corr']), s=5, alpha=0.7, color=COLORS['circle'], label='After [12]')
    ax2.axhline(0, color='k', lw=0.5); ax2.axvline(0, color='k', lw=0.5)
    ax2.plot(0, 0, 'r+', ms=12, markeredgewidth=2.5)
    ax2.set(title='Steps 4–5: IQ Constellation DC Corr [12]', xlabel='I', ylabel='Q', aspect='equal')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs_[0, 2])
    ax3.plot(tv, r['disp_mm'], color='#9C27B0', lw=1.2)
    ax3.fill_between(tv, r['disp_mm'], alpha=0.15, color='#9C27B0')
    ax3.axhline(0, color='k', lw=0.5)
    ax3.set(title='Step 6: Chest Displacement [mm] [14]', xlabel='Time [s]', ylabel='Disp [mm]')

    ax4 = fig.add_subplot(gs_[1, 0])
    mr = (rf >= 0.05) & (rf <= 0.6)
    ax4.fill_between(rf[mr], rm[mr], alpha=0.3, color=COLORS['resp'])
    ax4.plot(rf[mr], rm[mr], color=COLORS['resp'], lw=1.8)
    ax4.axvline(r['br_fft']/60, color='blue', ls='--', lw=2, label=f'Est: {r["br_fft"]:.1f} br/min')
    if meta['has_br']:
        ax4.axvline(meta['br_true']/60, color='red', ls=':', lw=2, label=f'True: {meta["br_true"]:.1f}')
    ax4.set(xlabel='Frequency [Hz]', ylabel='Magnitude', title=f'Step 7: Respiration [14]\n{r["br_fft"]:.1f} br/min')
    ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(gs_[1, 1])
    mh = (hf >= 0.7) & (hf <= 2.6)
    ax5.fill_between(hf[mh], hm[mh], alpha=0.3, color=COLORS['heart'])
    ax5.plot(hf[mh], hm[mh], color=COLORS['heart'], lw=1.8)
    ax5.axvline(r['hr_fft']/60, color='blue', ls='--', lw=2, label=f'Est: {r["hr_fft"]:.1f} bpm')
    ax5.axvline(meta['hr_true']/60, color='red', ls=':', lw=2, label=f'True: {meta["hr_true"]} bpm')
    err = abs(r['hr_fft'] - meta['hr_true'])
    ax5.set(xlabel='Frequency [Hz]', ylabel='Magnitude', title=f'Step 7: Heartbeat [14]\nMAE={err:.2f} bpm')
    ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs_[1, 2]); ax6.axis('off')
    hr_ac = f'{r["hr_ac"]:.1f} bpm' if r['hr_ac'] else 'N/A'
    br_ac = f'{r["br_ac"]:.1f} br/min' if r['br_ac'] else 'N/A'
    txt = (f"RESULTS — {tag}\n{'─'*30}\n"
           f"Range detected : {r['peak_range_m']:.2f} m  (true: {meta['dist_m']}m)\n"
           f"Angle detected : {r['peak_angle_deg']:.1f}°  (true: {meta['angle_deg']}°)\n"
           f"{'─'*30}\n"
           f"HR (FFT)  : {r['hr_fft']:.1f} bpm\n"
           f"HR (AC)   : {hr_ac}\n"
           f"HR True   : {meta['hr_true']} bpm  ±{meta['pm']}\n"
           f"HR Error  : {err:.1f} bpm\n"
           f"{'─'*30}\n"
           f"BR (FFT)  : {r['br_fft']:.1f} br/min\n"
           f"BR (AC)   : {br_ac}\n"
           f"BR True   : {meta['br_true'] if meta['has_br'] else 'Breath-hold'}\n"
           f"{'─'*30}\n"
           f"[12] Mercuri 2022\n[14] Alizadeh 2019")
    ax6.text(0.04, 0.97, txt, transform=ax6.transAxes, fontsize=8.5, va='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))
    ax6.set_title('Pipeline Results', fontweight='bold')

    fig.suptitle(f'FMCW QIDK Pipeline Dashboard — {tag}\n[12] Mercuri 2022  |  [14] Alizadeh 2019',
                 fontsize=13, fontweight='bold')
    return savefig(fig, od / f'{tag}_fig5_dashboard.png')

# ── Aggregate figures (all files in folder) ─────────────────────────────────

def fig_hr_scatter(df, folder_tag, od):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    lims = [max(40, df.hr_true.min()-5), min(160, df.hr_true.max()+5)]

    sc = axes[0].scatter(df.hr_true, df.hr_fft, c=df.dist_m, cmap='plasma',
                         s=60, edgecolors='k', lw=0.5, zorder=2, alpha=0.85)
    plt.colorbar(sc, ax=axes[0], label='Distance (m)')
    axes[0].plot(lims, lims, 'k--', lw=1.5, label='Perfect')
    for delta, alpha in [(5, 0.15), (10, 0.08)]:
        axes[0].fill_between(lims, [l-delta for l in lims], [l+delta for l in lims],
                             alpha=alpha, color='gray', label=f'±{delta} bpm')
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set(title=f'HR: Estimated vs True — {folder_tag}',
                xlabel='True HR (bpm)', ylabel='Estimated HR (bpm)')
    axes[0].legend(fontsize=9)

    errs = (df.hr_fft - df.hr_true).abs().sort_values()
    mae = errs.mean(); rmse = np.sqrt((errs**2).mean())
    w5 = (errs <= 5).mean()*100; w10 = (errs <= 10).mean()*100
    axes[1].bar(range(len(errs)), errs.values, color=COLORS['hr'], alpha=0.8, edgecolor='k', lw=0.3)
    axes[1].axhline(5,  color='green',  ls='--', lw=1.5, label='5 bpm')
    axes[1].axhline(10, color='orange', ls='--', lw=1.5, label='10 bpm')
    axes[1].set(title=f'HR Error (sorted)\nMAE={mae:.1f} | RMSE={rmse:.1f} | ≤5:{w5:.0f}% | ≤10:{w10:.0f}%',
                xlabel='File rank', ylabel='|HR Error| (bpm)')
    axes[1].legend(fontsize=9)

    fig.suptitle(f'[Sec 11] HR Accuracy — {folder_tag}', fontsize=13, fontweight='bold')
    return savefig(fig, od / 'agg_hr_scatter.png')

def fig_hr_by_distance(df, folder_tag, od):
    dists = sorted(df.dist_m.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

    maes = [df[df.dist_m==d]['hr_err'].mean() for d in dists]
    w10s = [(df[df.dist_m==d]['hr_err'] <= 10).mean()*100 for d in dists]

    axes[0].bar([str(d) for d in dists], maes, color='steelblue', edgecolor='k', lw=0.5, alpha=0.85)
    axes[0].axhline(10, color='red', ls='--', lw=1.5, label='10 bpm threshold')
    axes[0].set(title='HR MAE by Distance', xlabel='Distance (m)', ylabel='MAE (bpm)')
    axes[0].legend()

    axes[1].bar([str(d) for d in dists], w10s, color='#4CAF50', edgecolor='k', lw=0.5, alpha=0.85)
    axes[1].axhline(80, color='red', ls='--', lw=1.5, label='80% threshold')
    axes[1].set(title='HR Within 10 bpm by Distance', xlabel='Distance (m)', ylabel='% Files ≤10 bpm')
    axes[1].set_ylim(0, 105); axes[1].legend()

    fig.suptitle(f'[Sec 11] HR Accuracy vs Distance — {folder_tag}', fontsize=13, fontweight='bold')
    return savefig(fig, od / 'agg_hr_by_distance.png')

def fig_hr_by_angle(df, folder_tag, od):
    angles = sorted(df.angle_deg.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    maes = [df[df.angle_deg==a]['hr_err'].mean() for a in angles]
    w10s = [(df[df.angle_deg==a]['hr_err'] <= 10).mean()*100 for a in angles]

    axes[0].bar([f'{a}°' for a in angles], maes, color='darkorange', edgecolor='k', lw=0.5, alpha=0.85)
    axes[0].axhline(10, color='red', ls='--', lw=1.5)
    axes[0].set(title='HR MAE by Angle', xlabel='Angle (°)', ylabel='MAE (bpm)')

    axes[1].bar([f'{a}°' for a in angles], w10s, color='#9C27B0', edgecolor='k', lw=0.5, alpha=0.85)
    axes[1].set(title='HR Within 10 bpm by Angle', xlabel='Angle (°)', ylabel='% Files ≤10 bpm')
    axes[1].set_ylim(0, 105)

    fig.suptitle(f'[Sec 11] HR Accuracy vs Angle — {folder_tag}', fontsize=13, fontweight='bold')
    return savefig(fig, od / 'agg_hr_by_angle.png')

def fig_br_scatter(df, folder_tag, od):
    df_br = df[df.has_br]
    if len(df_br) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    lims = [df_br.br_true.min()-2, df_br.br_true.max()+2]

    axes[0].scatter(df_br.br_true, df_br.br_fft, s=60, edgecolors='k', lw=0.5,
                    color=COLORS['br'], alpha=0.85, zorder=2)
    axes[0].plot(lims, lims, 'k--', lw=1.5)
    axes[0].fill_between(lims, [l-3 for l in lims], [l+3 for l in lims], alpha=0.15, color='gray', label='±3 br/min')
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set(title='BR: Estimated vs True', xlabel='True BR (br/min)', ylabel='Estimated BR (br/min)')
    axes[0].legend()

    br_errs = (df_br.br_fft - df_br.br_true).abs().sort_values()
    mae = br_errs.mean()
    axes[1].bar(range(len(br_errs)), br_errs.values, color=COLORS['br'], alpha=0.8, edgecolor='k', lw=0.3)
    axes[1].axhline(3, color='red', ls='--', lw=1.5, label='3 br/min')
    axes[1].set(title=f'BR Error (sorted)\nMAE={mae:.1f} br/min', xlabel='File rank', ylabel='|BR Error|')
    axes[1].legend()

    fig.suptitle(f'[Sec 11] BR Accuracy — {folder_tag}', fontsize=13, fontweight='bold')
    return savefig(fig, od / 'agg_br_scatter.png')

def fig_summary_heatmap(df, folder_tag, od):
    dists  = sorted(df.dist_m.unique())
    angles = sorted(df.angle_deg.unique())
    mae_grid = np.zeros((len(dists), len(angles)))
    for i, dist in enumerate(dists):
        for j, ang in enumerate(angles):
            sub = df[(df.dist_m == dist) & (df.angle_deg == ang)]
            mae_grid[i, j] = sub.hr_err.mean() if len(sub) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    im = ax.imshow(mae_grid, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=30)
    ax.set_xticks(range(len(angles))); ax.set_xticklabels([f'{a}°' for a in angles])
    ax.set_yticks(range(len(dists)));  ax.set_yticklabels([f'{d}m' for d in dists])
    ax.set(xlabel='Angle', ylabel='Distance', title=f'HR MAE Heatmap (bpm) — {folder_tag}')
    plt.colorbar(im, ax=ax, label='MAE (bpm)')
    for i in range(len(dists)):
        for j in range(len(angles)):
            val = mae_grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='black')

    fig.suptitle(f'[Sec 11] HR MAE Grid: Distance × Angle — {folder_tag}', fontsize=12, fontweight='bold')
    return savefig(fig, od / 'agg_summary_heatmap.png')

def fig_ra_grid(df_loc, results_loc, folder_tag, od):
    """Grid of Range-Azimuth maps for all files in this location."""
    n = len(results_loc); ncols = min(5, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), tight_layout=True)
    axes = np.array(axes).flatten()
    for idx, (row, r) in enumerate(zip(df_loc.itertuples(), results_loc)):
        ax = axes[idx]
        data = np.abs(r['ra_heatmap']).T
        ax.imshow(data, extent=[azimuth[-1], azimuth[0], 0, R_max],
                  aspect='auto', origin='lower', cmap='jet')
        ax.set_ylim(0, min(6, R_max))
        ax.axhline(r['peak_range_m'], color='white', ls='--', lw=1.5)
        ax.axhline(row.dist_m, color='red', ls=':', lw=1.5)
        err = abs(r['hr_fft'] - row.hr_true)
        ax.set_title(f'HR {row.hr_true}→{r["hr_fft"]:.0f} (Δ{err:.0f})', fontsize=8, fontweight='bold')
        ax.set_xlabel('Az (°)', fontsize=7); ax.set_ylabel('Range (m)', fontsize=7)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f'[Sec 18] Range-Azimuth Grid — {folder_tag}', fontsize=12, fontweight='bold')
    return savefig(fig, od / f'{folder_tag.replace("/","_")}_ra_grid.png')

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_notebook(records, agg_figs, out_path):
    cells = []

    def md(text):
        return {'cell_type':'markdown','metadata':{},'source':[text]}
    def img(path, title=''):
        import base64
        try:
            with open(path,'rb') as f: data=base64.b64encode(f.read()).decode()
        except: return md(f'[Image not found: {path}]')
        return {
            'cell_type':'code','metadata':{},'source':[''],
            'outputs':[{'output_type':'display_data','metadata':{},
                        'data':{'image/png':data,'text/plain':['<Figure>']}}],
            'execution_count':None
        }

    cells.append(md('# FMCW QIDK Pipeline — Output Report\n**IC~WiBES Research Group**\n\n---'))
    cells.append(md(f'## Dataset Summary\n\n| | |\n|---|---|\n'
                    f'|Files processed|{len(records)}|\n'
                    f'|HR MAE|{np.mean([abs(r["hr_fft"]-r["hr_true"]) for r in records]):.1f} bpm|\n'
                    f'|HR RMSE|{np.sqrt(np.mean([(r["hr_fft"]-r["hr_true"])**2 for r in records])):.1f} bpm|'))

    # Aggregate figures first
    cells.append(md('## Aggregate Accuracy Analysis'))
    for p in agg_figs:
        if p: cells.append(img(p))

    # Per-location sections
    locs = {}
    for r in records:
        key = f"D{r['dist_m']}m_{r['angle_deg']}d"
        locs.setdefault(key,[]).append(r)

    for loc, loc_recs in sorted(locs.items()):
        cells.append(md(f'## Location: {loc}'))
        for r in loc_recs:
            cells.append(md(f'### {r["filename"]} — HR: {r["hr_fft"]:.1f} bpm (True: {r["hr_true"]})'))
            for fp in r.get('fig_paths', []):
                cells.append(img(fp))

    nb = {
        'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python','version':'3.x'}},
        'cells':cells
    }
    with open(out_path,'w') as f:
        json.dump(nb, f, indent=1)
    print(f'  Notebook saved: {out_path}')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_folder(folder_path: str):
    folder = Path(folder_path)
    assert folder.exists(), f"Folder not found: {folder}"

    # Find all .bin files
    all_bins = sorted(folder.rglob('*.bin'))
    valid = [p for p in all_bins if FNAME_RE.match(p.name)]
    print(f"\n{'='*60}")
    print(f"  Folder  : {folder}")
    print(f"  .bin files found : {len(valid)}")
    print(f"{'='*60}")

    od = Path('results/figures'); od.mkdir(parents=True, exist_ok=True)

    all_records = []
    for fp in valid:
        meta = parse_meta(fp)
        tag = fp.stem
        loc = f"D{meta['dist_m']}m_{meta['angle_deg']}d"
        loc_dir = od / loc; loc_dir.mkdir(exist_ok=True)

        print(f"  {fp.name} ...", end='', flush=True)
        t0 = time.perf_counter()
        try:
            r = run_pipeline(str(fp))
            elapsed = (time.perf_counter()-t0)*1e3
            err = abs(r['hr_fft']-meta['hr_true'])
            print(f" HR={r['hr_fft']:.1f} (true={meta['hr_true']}, err={err:.1f}) | {elapsed:.0f}ms")

            # Generate per-file figures
            fig_paths = []
            fig_paths.append(fig_raw_and_rst(r, meta, tag, loc_dir))
            fig_paths.append(fig_svd_and_ra(r, meta, tag, loc_dir))
            fig_paths.append(fig_iq_and_displacement(r, meta, tag, loc_dir))
            fig_paths.append(fig_spectra(r, meta, tag, loc_dir))
            fig_paths.append(fig_dashboard(r, meta, tag, loc_dir))

            rec = {**meta, 'hr_fft':r['hr_fft'], 'br_fft':r['br_fft'],
                   'hr_ac':r['hr_ac'], 'br_ac':r['br_ac'],
                   'hr_err':err, 'br_err':(abs(r['br_fft']-meta['br_true']) if meta['has_br'] else None),
                   'peak_range_m':r['peak_range_m'], 'peak_angle_deg':r['peak_angle_deg'],
                   'ra_heatmap':r['ra_heatmap'],
                   'fig_paths':fig_paths, 'hr_fft_r':r['hr_fft']}
            all_records.append(rec)
        except Exception as e:
            print(f" ERROR: {e}")

    if not all_records:
        print("No files processed."); return

    df = pd.DataFrame(all_records)
    df.to_csv('results/results.csv', index=False)
    print(f"\n  Results CSV saved: results/results.csv")

    # Aggregate figures
    folder_tag = folder.name
    print(f"\n  Generating aggregate figures...")
    agg = []
    agg.append(fig_hr_scatter(df, folder_tag, od))
    if len(df.dist_m.unique()) > 1:
        agg.append(fig_hr_by_distance(df, folder_tag, od))
        agg.append(fig_summary_heatmap(df, folder_tag, od))
    if len(df.angle_deg.unique()) > 1:
        agg.append(fig_hr_by_angle(df, folder_tag, od))
    agg.append(fig_br_scatter(df, folder_tag, od))

    # Per-location RA grid
    for loc, loc_df in df.groupby(lambda i: f"D{df.loc[i,'dist_m']}m_{df.loc[i,'angle_deg']}d"):
        loc_results = [all_records[i] for i in loc_df.index]
        agg.append(fig_ra_grid(loc_df, loc_results, loc, od))

    # Build notebook
    print(f"\n  Building output_report.ipynb ...")
    build_notebook(all_records, [a for a in agg if a], 'output_report.ipynb')

    # Summary
    hr_errs = df.hr_err
    print(f"\n{'─'*50}")
    print(f"  SUMMARY — {folder_tag}")
    print(f"  Files: {len(df)}")
    print(f"  HR MAE    : {hr_errs.mean():.2f} bpm")
    print(f"  HR RMSE   : {np.sqrt((hr_errs**2).mean()):.2f} bpm")
    print(f"  HR ≤5bpm  : {(hr_errs<=5).mean()*100:.1f}%")
    print(f"  HR ≤10bpm : {(hr_errs<=10).mean()*100:.1f}%")
    print(f"  Results   : results/")
    print(f"  Figures   : results/figures/")
    print(f"  Report    : output_report.ipynb")
    print(f"{'─'*50}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 run_folder.py <folder_path>")
        print("  e.g: python3 run_folder.py ./Single_Subject/")
        sys.exit(1)
    run_folder(sys.argv[1])
