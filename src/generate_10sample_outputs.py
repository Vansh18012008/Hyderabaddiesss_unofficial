"""
generate_10sample_outputs.py
============================
Runs the full CBF pipeline on 10 selected files from Single_Subject/,
saves every intermediate-stage figure as PNG into ./pipeline_outputs/figures/,
and writes a metrics JSON to ./pipeline_outputs/metrics_10.json.

Called by build_clean_notebooks.py.  Run standalone with:
    python generate_10sample_outputs.py
"""

import os, sys, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dc_offset_correction  import circle_fit_correction
from clutter_suppression   import svd_clutter_removal
from phase_demodulation    import demodulate
from vital_sign_extraction import extract_vital_signs

# ── Style ──────────────────────────────────────────────────────────────────
DARK    = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
FG      = '#c9d1d9'
ACCENT  = '#58a6ff'
GREEN   = '#3fb950'
AMBER   = '#d29922'
RED     = '#f85149'
PURPLE  = '#bc8cff'
TEAL    = '#39d353'
WHITE   = '#e6edf3'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': FG,
    'xtick.color': '#8b949e',  'ytick.color': '#8b949e',
    'text.color': FG,          'grid.color': BORDER,
    'grid.linestyle': '--',    'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans', 'figure.dpi': 130,
    'axes.titlepad': 8,
})

# ── Hardware constants ──────────────────────────────────────────────────────
F_MIN, F_MAX = 77e9, 79.982e9
C = 3e8; BW = F_MAX - F_MIN; TM = 100e-6; F_ADC = 600e3
LAMBDA_C   = C / ((F_MIN + F_MAX) / 2.0)
LAMBDA_MAX = C / F_MIN;  D_ANT = LAMBDA_MAX / 2
R_MAX      = (C * TM / 2.0) * (F_ADC / BW)
R_RES      = C / (2.0 * BW)
N_CHIRPS   = M_CHIRPS = 512
NUM_RX     = 4; N_FFT_RANGE = 512
FS_VITAL   = M_CHIRPS / 25.6   # 20.0 Hz

AZIMUTH_GRID = np.arange(90, -90 - 2.5, -2.5)
RANGE_AXIS   = np.linspace(0, R_MAX, N_FFT_RANGE // 2)
WIN_RANGE    = np.hanning(N_CHIRPS)
SINGLE_DIR   = Path('./Single_Subject')
OUT_DIR      = Path('./pipeline_outputs')
FIG_DIR      = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILES = [
    'D1.5m_0d_HR_89_pm_1_BR_18.bin',
    'D1.5m_10d_HR_88_pm_1_BR_15.bin',
    'D1.5m_20d_HR_89_pm_2_BR_16.bin',
    'D1.5m_30d_HR_98_pm_1_BR_14.bin',
    'D2m_0d_HR_97_pm_0_BR_15.bin',
    'D2.5m_10d_HR_100_pm_3_BR_16.bin',
    'D3m_0d_HR_84_pm_1_BR_15.bin',
    'D3.5m_0d_HR_85_pm_0_BR_16.bin',
    'D4m_0d_HR_90_pm_1_BR_15.bin',
    'D4.5m_0d_HR_90_pm_1_BR_15.bin',
]


# ── Helpers ─────────────────────────────────────────────────────────────────
_RE = re.compile(
    r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin'
)

def parse_meta(path):
    m = _RE.match(Path(path).name)
    if not m: return {}
    br = float(m.group('br'))
    return dict(dist_m=float(m.group('dist')), angle_deg=int(m.group('angle')),
                hr_true=int(m.group('hr')), br_true=br, has_br=br > 0)

def read_dca1000(path):
    raw  = np.fromfile(path, dtype=np.int16)
    I    = raw[0::4] + 1j * raw[2::4]
    Q    = raw[1::4] + 1j * raw[3::4]
    LVDS = np.empty(I.size + Q.size, dtype=np.complex64)
    LVDS[0::2] = I; LVDS[1::2] = Q
    nc   = raw.shape[0] // (2 * N_CHIRPS * NUM_RX)
    LVDS = LVDS.reshape((N_CHIRPS * NUM_RX, nc), order='F').T
    out  = np.zeros((NUM_RX, nc * N_CHIRPS), dtype=np.complex64)
    for rx in range(NUM_RX):
        for chp in range(nc):
            out[rx, chp*N_CHIRPS:(chp+1)*N_CHIRPS] = LVDS[chp, rx*N_CHIRPS:(rx+1)*N_CHIRPS]
    return out

def apply_cbf(if_data):
    rx_pos  = np.arange(NUM_RX) * (D_ANT / LAMBDA_C)
    ang_rad = np.deg2rad(AZIMUTH_GRID)
    W = np.exp(-1j * 2*np.pi * rx_pos[:, None] * np.sin(ang_rad)[None, :]) / NUM_RX
    return np.einsum('ra,rcs->acs', W, if_data)

def range_fft_cube(beamed):
    na   = beamed.shape[0]
    rfft = np.zeros((na, M_CHIRPS, N_FFT_RANGE//2), dtype=np.complex64)
    for nn in range(na):
        Xf = np.fft.fft(beamed[nn] * WIN_RANGE[np.newaxis, :], n=N_FFT_RANGE, axis=1)
        rfft[nn] = Xf[:, :N_FFT_RANGE//2].astype(np.complex64)
    return rfft

def autocorr_rate(sig, fs):
    x = sig - np.mean(sig)
    r = np.correlate(x, x, mode='full'); r = r[r.size//2:]; r /= r[0]+1e-30
    peaks, _ = find_peaks(r, distance=int(fs*0.4))
    if not len(peaks): return None
    T = peaks[0]/fs
    return round((1/T)*60, 2) if T > 0 else None

def fft_peak(sig, fs, fmin, fmax):
    n = len(sig); nf = 4*n
    spec  = np.abs(np.fft.fft(sig*np.hanning(n), n=nf))[:nf//2]
    freqs = np.arange(nf//2) * fs / nf
    mask  = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask): return 0.0, freqs, spec
    idx = np.where(mask)[0]; pk = idx[np.argmax(spec[idx])]
    return float(freqs[pk]), freqs, spec

def run_pipeline(bin_path):
    na = len(AZIMUTH_GRID)
    raw     = read_dca1000(bin_path)
    if_data = raw.reshape(NUM_RX, M_CHIRPS, N_CHIRPS).astype(np.complex64)

    beamed     = apply_cbf(if_data)
    beamed_raw = beamed.copy()
    for nn in range(na):
        m = beamed[nn]
        beamed[nn] = (m - np.mean(m, axis=0, keepdims=True)).astype(np.complex64)

    rfft_cube     = range_fft_cube(beamed)
    rfft_pre_svd  = range_fft_cube(beamed_raw)
    rfft_mean_pre = np.mean(np.abs(rfft_pre_svd), axis=1)

    rfft_mean = np.zeros((na, N_FFT_RANGE//2))
    for nn in range(na):
        cleaned, _ = svd_clutter_removal(rfft_cube[nn], rank=1)
        rfft_cube[nn]  = cleaned.astype(np.complex64)
        rfft_mean[nn]  = np.mean(np.abs(rfft_cube[nn]), axis=0)

    mb  = max(1, int(0.3/R_MAX * (N_FFT_RANGE//2)))
    msk = rfft_mean.copy(); msk[:, :mb] = 0
    fi  = int(np.argmax(msk))
    pai = fi // (N_FFT_RANGE//2); pri = fi % (N_FFT_RANGE//2)

    slow_iq_raw       = rfft_cube[pai, :, pri].copy()
    slow_iq_corr, cfi = circle_fit_correction(slow_iq_raw)

    dem     = demodulate(slow_iq_corr, wavelength_m=LAMBDA_C, remove_dc_in_phase=True)
    disp_mm = dem['displacement_mm']

    vs    = extract_vital_signs(disp_mm, fs=FS_VITAL,
                                resp_band=(0.1, 0.5), heart_band=(0.8, 2.5),
                                estimator='fft')
    hr_ac = autocorr_rate(vs['heart_signal_mm'], FS_VITAL)
    br_ac = autocorr_rate(vs['resp_signal_mm'],  FS_VITAL)
    br_hz, br_freqs, br_spec = fft_peak(vs['resp_signal_mm'],  FS_VITAL, 0.1, 0.5)
    hr_hz, hr_freqs, hr_spec = fft_peak(vs['heart_signal_mm'], FS_VITAL, 0.8, 2.5)

    return dict(
        slow_iq_raw=slow_iq_raw, slow_iq_corr=slow_iq_corr, cf_info=cfi,
        ra_pre=rfft_mean_pre, ra_post=rfft_mean,
        disp_mm=disp_mm, phase_clean=dem['phase_clean_rad'],
        resp_signal=vs['resp_signal_mm'], heart_signal=vs['heart_signal_mm'],
        hr_fft=round(vs['heart_rate_bpm'], 2), br_fft=round(vs['resp_rate_bpm'], 2),
        hr_ac=hr_ac, br_ac=br_ac,
        hr_hz=hr_hz, br_hz=br_hz,
        hr_freqs=hr_freqs, hr_spec=hr_spec,
        br_freqs=br_freqs, br_spec=br_spec,
        rng_m=round(float(RANGE_AXIS[pri]), 3), ang_d=round(float(AZIMUTH_GRID[pai]), 2),
    )


# ── Figure generators ────────────────────────────────────────────────────────

def fig_heatmaps(r, idx, meta, prefix):
    """Range–azimuth heatmap before and after SVD clutter suppression."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    fname = SAMPLE_FILES[idx]
    fig.suptitle(
        f'Sample {idx+1}  ·  {fname}\nRange–Azimuth Heatmap — Before vs After SVD Clutter Suppression',
        color=WHITE, fontsize=11, fontweight='bold', y=1.02
    )

    nr = min(80, N_FFT_RANGE//2)
    ext = [AZIMUTH_GRID[-1], AZIMUTH_GRID[0], 0, RANGE_AXIS[nr]]

    for ax, data, title, cmap, col in [
        (axes[0], r['ra_pre'],  'BEFORE SVD Clutter Removal', 'magma',  AMBER),
        (axes[1], r['ra_post'], 'AFTER SVD Clutter Removal',  'plasma', GREEN),
    ]:
        ax.set_facecolor(PANEL)
        dB = 20*np.log10(data[:, :nr] + 1e-30)
        im = ax.imshow(dB.T, aspect='auto', origin='lower', extent=ext,
                       cmap=cmap, vmin=np.percentile(dB, 5))
        plt.colorbar(im, ax=ax, label='Power [dB]', pad=0.02)
        ax.set_title(title, color=col, fontsize=10, fontweight='bold')
        ax.set_xlabel('Azimuth [°]', color=FG); ax.set_ylabel('Range [m]', color=FG)
        ax.grid(True, alpha=0.25)

    axes[1].axhline(r['rng_m'], color=GREEN, lw=1.5, ls='--',
                    label=f'Detected: {r["rng_m"]:.2f} m')
    axes[1].axvline(r['ang_d'], color=ACCENT, lw=1.5, ls='--',
                    label=f'Detected: {r["ang_d"]:.1f}°')
    axes[1].legend(fontsize=8, loc='upper right', facecolor=PANEL, edgecolor=BORDER)

    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_s{idx+1:02d}_heatmap.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_iq(r, idx, meta, prefix):
    """IQ constellation before and after circle-fit DC correction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f'Sample {idx+1}  ·  {SAMPLE_FILES[idx]}\nIQ Constellation — Before vs After Circle-Fit DC Correction',
        color=WHITE, fontsize=11, fontweight='bold', y=1.02
    )

    Ir, Qr = np.real(r['slow_iq_raw']),  np.imag(r['slow_iq_raw'])
    Ic, Qc = np.real(r['slow_iq_corr']), np.imag(r['slow_iq_corr'])
    t_idx  = np.arange(len(Ir))

    # Before
    ax = axes[0]; ax.set_facecolor(PANEL)
    sc = ax.scatter(Ir, Qr, c=t_idx, cmap='viridis', s=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Chirp index', pad=0.02)
    dc_i, dc_q = np.mean(Ir), np.mean(Qr)
    ax.scatter(dc_i, dc_q, color=RED, s=120, marker='x', linewidths=2.5, zorder=7,
               label=f'DC centre\n({dc_i:.3f}, {dc_q:.3f})')
    ax.axhline(0, color=BORDER, lw=0.8); ax.axvline(0, color=BORDER, lw=0.8)
    ax.set_aspect('equal', 'box'); ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER)
    ax.set_title('BEFORE DC Correction\n(centre offset from origin)', color=RED,
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('I (In-phase)'); ax.set_ylabel('Q (Quadrature)')
    ax.grid(True, alpha=0.25)

    # After
    ax = axes[1]; ax.set_facecolor(PANEL)
    sc = ax.scatter(Ic, Qc, c=t_idx, cmap='plasma', s=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Chirp index', pad=0.02)
    rad   = float(np.mean(np.sqrt(Ic**2 + Qc**2)))
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(rad*np.cos(theta), rad*np.sin(theta), color=GREEN, lw=1.5, ls='--',
            alpha=0.8, label=f'Fitted circle r={rad:.3f}')
    ax.scatter(0, 0, color=GREEN, s=130, marker='+', linewidths=2.5, zorder=7,
               label=f'DC removed\n({r["cf_info"]["dc_i"]:.3f}, {r["cf_info"]["dc_q"]:.3f})')
    ax.axhline(0, color=BORDER, lw=0.8); ax.axvline(0, color=BORDER, lw=0.8)
    ax.set_aspect('equal', 'box'); ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER)
    ax.set_title('AFTER Circle-Fit DC Correction\n(circle centred at origin)', color=GREEN,
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('I (In-phase)'); ax.set_ylabel('Q (Quadrature)')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_s{idx+1:02d}_iq.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_displacement(r, idx, meta, prefix):
    """Chest displacement signal + unwrapped phase."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f'Sample {idx+1}  ·  {SAMPLE_FILES[idx]}\nPhase Demodulation → Chest Displacement',
        color=WHITE, fontsize=11, fontweight='bold'
    )
    t = np.arange(M_CHIRPS) / FS_VITAL

    ax = axes[0]; ax.set_facecolor(PANEL)
    ax.plot(t, r['disp_mm'],      color=ACCENT,  lw=0.9, alpha=0.7, label='Total displacement [mm]')
    ax.plot(t, r['resp_signal'],  color=GREEN,   lw=1.5, label=f'Respiration  [{r["br_fft"]:.1f} br/min est.]')
    ax.plot(t, r['heart_signal'], color=PURPLE,  lw=1.0, alpha=0.9,
            label=f'Heartbeat  [{r["hr_fft"]:.1f} bpm est.]')
    ax.set_ylabel('Displacement [mm]'); ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER)
    ax.set_title('Bandpass-Decomposed Chest Motion', color=ACCENT, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)

    ax = axes[1]; ax.set_facecolor(PANEL)
    ax.plot(t, r['phase_clean'], color=TEAL, lw=1.0)
    ax.fill_between(t, r['phase_clean'], alpha=0.15, color=TEAL)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Phase [rad]')
    ax.set_title('Unwrapped Phase Signal (after DC removal)', color=TEAL, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_s{idx+1:02d}_displacement.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_spectra(r, idx, meta, prefix):
    """HR and BR frequency spectra with estimated vs ground-truth markers."""
    m      = meta
    hr_err = abs(r['hr_fft'] - m['hr_true'])
    br_err = abs(r['br_fft'] - m['br_true']) if m['has_br'] else None
    hr_col = GREEN if hr_err <= 5 else (AMBER if hr_err <= 10 else RED)
    br_col = (GREEN if br_err is not None and br_err <= 2
              else AMBER if br_err is not None and br_err <= 4 else RED) if br_err else ACCENT

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f'Sample {idx+1}  ·  {SAMPLE_FILES[idx]}\nVital Sign Frequency Spectra',
        color=WHITE, fontsize=11, fontweight='bold', y=1.02
    )

    # Respiration
    ax = axes[0]; ax.set_facecolor(PANEL)
    mb = (r['br_freqs'] >= 0.05) & (r['br_freqs'] <= 0.75)
    bx = r['br_freqs'][mb]*60; by = r['br_spec'][mb]
    ax.plot(bx, by, color=GREEN, lw=1.8)
    ax.fill_between(bx, by, alpha=0.2, color=GREEN)
    ax.axvline(r['br_hz']*60, color=br_col, lw=2.5, ls='--',
               label=f'Estimated BR = {r["br_fft"]:.1f} br/min')
    if m['has_br']:
        ax.axvline(m['br_true'], color=WHITE, lw=2, ls=':',
                   label=f'Ground truth = {m["br_true"]} br/min')
    err_txt = f'Error = {br_err:.1f} br/min' if br_err is not None else 'No BR ground truth'
    ax.set_title(f'Respiration Spectrum\n{err_txt}', color=GREEN, fontsize=10, fontweight='bold')
    ax.set_xlabel('Frequency [br/min]'); ax.set_ylabel('Spectral Magnitude')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER); ax.grid(True, alpha=0.25)
    ax.set_xlim(3, 45)

    # Heart rate
    ax = axes[1]; ax.set_facecolor(PANEL)
    mh = (r['hr_freqs'] >= 0.6) & (r['hr_freqs'] <= 3.2)
    hx = r['hr_freqs'][mh]*60; hy = r['hr_spec'][mh]
    ax.plot(hx, hy, color=PURPLE, lw=1.8)
    ax.fill_between(hx, hy, alpha=0.2, color=PURPLE)
    ax.axvline(r['hr_hz']*60, color=hr_col, lw=2.5, ls='--',
               label=f'Estimated HR = {r["hr_fft"]:.1f} bpm')
    ax.axvline(m['hr_true'], color=WHITE, lw=2, ls=':',
               label=f'Ground truth = {m["hr_true"]} bpm')
    ax.set_title(f'Heart Rate Spectrum\nError = {hr_err:.1f} bpm', color=PURPLE,
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Frequency [bpm]'); ax.set_ylabel('Spectral Magnitude')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER); ax.grid(True, alpha=0.25)
    ax.set_xlim(36, 192)

    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_s{idx+1:02d}_spectra.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_summary_errors(all_results, all_metas, prefix):
    """Summary: True vs Est HR bar chart + per-sample error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('10-Sample Error Summary — CBF Pipeline (No 2D MUSIC)',
                 color=WHITE, fontsize=13, fontweight='bold')
    for ax in axes: ax.set_facecolor(PANEL)

    lbls    = [f'S{i+1}' for i in range(len(all_results))]
    hr_est  = [r['hr_fft']          for r in all_results]
    hr_true = [m['hr_true']         for m in all_metas]
    hr_errs = [abs(e-t) for e,t in zip(hr_est, hr_true)]

    bi     = [i for i,m in enumerate(all_metas) if m['has_br']]
    br_est = [all_results[i]['br_fft']    for i in bi]
    br_tr  = [all_metas[i]['br_true']     for i in bi]
    br_ev  = [abs(e-t) for e,t in zip(br_est, br_tr)]
    bl     = [f'S{i+1}' for i in bi]

    x = np.arange(len(all_results))

    # True vs Est HR
    ax = axes[0]
    ax.bar(x-0.2, hr_true, 0.35, label='True HR', color=GREEN,  alpha=0.85, edgecolor=BORDER)
    ax.bar(x+0.2, hr_est,  0.35, label='Est HR',  color=ACCENT, alpha=0.85, edgecolor=BORDER)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel('Heart Rate [bpm]')
    ax.set_title('True vs Estimated Heart Rate', color=ACCENT, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER)
    ax.grid(True, alpha=0.25, axis='y')

    # HR error per sample
    ax = axes[1]
    cols = [GREEN if e<=5 else AMBER if e<=10 else RED for e in hr_errs]
    ax.bar(x, hr_errs, color=cols, alpha=0.9, edgecolor=BORDER)
    ax.axhline(np.mean(hr_errs), color=WHITE, lw=2, ls='--',
               label=f'MAE = {np.mean(hr_errs):.2f} bpm')
    ax.axhline(10, color=AMBER, lw=1, ls=':', alpha=0.7, label='±10 bpm')
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel('|Error| [bpm]')
    ax.set_title('Heart Rate Absolute Error per Sample', color=RED, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER)
    ax.grid(True, alpha=0.25, axis='y')

    # True vs Est BR
    ax = axes[2]
    xb = np.arange(len(bi))
    ax.bar(xb-0.2, br_tr,  0.35, label='True BR', color=TEAL,   alpha=0.85, edgecolor=BORDER)
    ax.bar(xb+0.2, br_est, 0.35, label='Est BR',  color=PURPLE, alpha=0.85, edgecolor=BORDER)
    ax.set_xticks(xb); ax.set_xticklabels(bl, fontsize=9)
    ax.set_ylabel('Breathing Rate [br/min]')
    ax.set_title('True vs Estimated Breathing Rate', color=PURPLE, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER)
    ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_summary_errors.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    prefix   = '10s'
    all_res  = []
    all_meta = []
    fig_paths = {i: {} for i in range(len(SAMPLE_FILES))}

    for idx, fname in enumerate(SAMPLE_FILES):
        fpath = str(SINGLE_DIR / fname)
        if not os.path.exists(fpath):
            print(f'  ⚠  Not found: {fname}'); continue
        print(f'  [{idx+1:>2}/10] Processing {fname} ...', end='', flush=True)
        try:
            meta = parse_meta(fpath)
            r    = run_pipeline(fpath)
            all_res.append(r)
            all_meta.append(meta)

            fp = fig_paths[idx]
            fp['heatmap']     = fig_heatmaps(r, idx, meta, prefix)
            fp['iq']          = fig_iq(r, idx, meta, prefix)
            fp['displacement']= fig_displacement(r, idx, meta, prefix)
            fp['spectra']     = fig_spectra(r, idx, meta, prefix)

            hr_e = abs(r['hr_fft'] - meta['hr_true'])
            br_e = abs(r['br_fft'] - meta['br_true']) if meta['has_br'] else None
            print(f"  HR={r['hr_fft']:.1f} (err={hr_e:.1f})  "
                  f"BR={r['br_fft']:.1f}" + (f" (err={br_e:.1f})" if br_e else ''))
        except Exception as e:
            print(f'\n    ❌ ERROR: {e}')

    # Summary error figure
    summary_path = fig_summary_errors(all_res, all_meta, prefix)

    # Metrics JSON
    metrics = []
    for idx, (r, meta) in enumerate(zip(all_res, all_meta)):
        hr_e = abs(r['hr_fft'] - meta['hr_true'])
        br_e = round(abs(r['br_fft'] - meta['br_true']), 2) if meta['has_br'] else None
        metrics.append(dict(
            sample_idx=idx+1, filename=SAMPLE_FILES[idx],
            dist_m=meta['dist_m'], angle_deg=meta['angle_deg'],
            hr_true=meta['hr_true'],  hr_est=r['hr_fft'],
            hr_err=round(hr_e, 2),
            br_true=meta['br_true'] if meta['has_br'] else None,
            br_est=r['br_fft'],       br_err=br_e,
            hr_ac=r['hr_ac'],         br_ac=r['br_ac'],
            rng_m=r['rng_m'],         ang_d=r['ang_d'],
            figs=fig_paths[idx],
        ))

    hr_errs = [m['hr_err'] for m in metrics]
    br_errs = [m['br_err'] for m in metrics if m['br_err'] is not None]

    output = dict(
        metrics=metrics,
        summary=dict(
            n_total=len(metrics),
            hr_mae=round(np.mean(hr_errs), 3),
            hr_rmse=round(float(np.sqrt(np.mean(np.array(hr_errs)**2))), 3),
            hr_median=round(float(np.median(hr_errs)), 3),
            br_mae=round(np.mean(br_errs), 3)  if br_errs else None,
            br_rmse=round(float(np.sqrt(np.mean(np.array(br_errs)**2))), 3) if br_errs else None,
            n_br=len(br_errs),
        ),
        summary_fig=summary_path,
    )
    out_json = OUT_DIR / 'metrics_10.json'
    with open(out_json, 'w') as f:
        # Convert numpy arrays to lists before serializing
        def _clean(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list): return [_clean(v) for v in o]
            return o
        json.dump(_clean(output), f, indent=2)

    print(f'\n✅ 10-sample figures saved to {FIG_DIR}')
    print(f'   Metrics JSON: {out_json}')
    print(f'   HR MAE = {output["summary"]["hr_mae"]:.2f} bpm')
    if br_errs: print(f'   BR MAE = {output["summary"]["br_mae"]:.2f} br/min')


if __name__ == '__main__':
    main()
