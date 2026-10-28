"""
make_notebooks.py
=================
Generates two Jupyter notebooks for the FMCW radar pipeline:

  1. pipeline_10_samples.ipynb
     - 10 files from Single_Subject/
     - Full pipeline visualization per sample: heatmaps, IQ constellations,
       displacement, spectra, QIDK display
     - Summary error table + bar charts
     - Markdown explanations of every pipeline step

  2. batch_all_320_qidk.ipynb
     - All 320 files from Single_Subject/
     - DC offset correction + SVD clutter suppression on every file
     - QIDK terminal display for all results
     - HR/BR error metrics: MAE, RMSE, within-band %
     - Aggregate plots: scatter, histogram, grouped bar by distance/angle
     - CSV export of all results

Run:
  python make_notebooks.py
"""

import nbformat as nbf
import textwrap, os

NB_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def md(src): return nbf.v4.new_markdown_cell(textwrap.dedent(src).strip())
def code(src): return nbf.v4.new_code_cell(textwrap.dedent(src).strip())
def nb(): return nbf.v4.new_notebook()


# ===========================================================================
#  NOTEBOOK 1 — 10-Sample Pipeline Deep-Dive
# ===========================================================================

def make_nb1():
    n = nb()

    # ── Title ──
    n.cells.append(md("""
    # 🎯 FMCW Radar Pipeline — 10 Sample Deep Dive
    ## DC Offset Correction → Static Clutter Suppression → HR/BR Estimation

    **Dataset:** AWR1843 / DCA1000 Single-Subject (320 recordings total)
    **This notebook:** 10 recordings spanning 1.5 m – 4.5 m, 0° – 30°
    **Method:** CBF only (no 2D MUSIC)

    ```
    .bin → Read DCA1000 → Mean-Sub DC (Stage-1a) → Range FFT
         → SVD Clutter Removal → Peak Detection → Circle-Fit DC (Stage-1b)
         → Phase Demodulation → Bandpass → FFT Vital-Sign Estimation
         → QIDK Display
    ```
    ---
    """))

    # ── Cell 1: Imports & constants ──
    n.cells.append(code("""
    # Cell 1 — Imports & Hardware Constants
    import os, sys, re
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    from scipy.signal import butter, filtfilt, find_peaks

    sys.path.insert(0, os.path.abspath('.'))
    from dc_offset_correction  import circle_fit_correction, mean_subtraction
    from clutter_suppression   import svd_clutter_removal
    from phase_demodulation    import demodulate
    from vital_sign_extraction import extract_vital_signs

    matplotlib.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',   'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
        'text.color': '#c9d1d9',       'grid.color': '#21262d',
        'grid.linestyle': '--',        'grid.linewidth': 0.5,
        'font.family': 'monospace',    'figure.dpi': 110,
    })

    # AWR1843 / DCA1000 hardware constants
    F_MIN, F_MAX = 77e9, 79.982e9
    C    = 3e8
    BW   = F_MAX - F_MIN
    TM   = 100e-6
    F_ADC = 600e3
    LAMBDA_C   = C / ((F_MIN + F_MAX) / 2.0)
    LAMBDA_MAX = C / F_MIN
    D_ANT      = LAMBDA_MAX / 2
    R_MAX      = (C * TM / 2.0) * (F_ADC / BW)
    R_RES      = C / (2.0 * BW)
    N_CHIRPS   = 512
    M_CHIRPS   = 512
    NUM_RX     = 4
    N_FFT_RANGE = 512
    FS_VITAL   = M_CHIRPS / 25.6      # 20.0 Hz

    AZIMUTH_GRID = np.arange(90, -90 - 2.5, -2.5)
    RANGE_AXIS   = np.linspace(0, R_MAX, N_FFT_RANGE // 2)
    WIN_RANGE    = np.hanning(N_CHIRPS)
    SINGLE_DIR   = Path('./Single_Subject')

    print(f'✅ R_MAX={R_MAX:.2f} m  R_RES={R_RES*100:.2f} cm  FS={FS_VITAL:.1f} Hz  λ={LAMBDA_C*1e3:.2f} mm')
    """))

    # ── Cell 2: Helper functions ──
    n.cells.append(code(r"""
    # Cell 2 — Helper Functions
    _RE = re.compile(r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin')

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
        W = np.exp(-1j * 2 * np.pi * rx_pos[:, None] * np.sin(ang_rad)[None, :]) / NUM_RX
        return np.einsum('ra,rcs->acs', W, if_data)

    def range_fft_cube(beamed):
        na, rfft = beamed.shape[0], np.zeros((beamed.shape[0], M_CHIRPS, N_FFT_RANGE//2), dtype=np.complex64)
        for nn in range(na):
            Xf = np.fft.fft(beamed[nn] * WIN_RANGE[np.newaxis, :], n=N_FFT_RANGE, axis=1)
            rfft[nn] = Xf[:, :N_FFT_RANGE//2].astype(np.complex64)
        return rfft

    def autocorr_rate(sig, fs):
        x = sig - np.mean(sig)
        r = np.correlate(x, x, mode='full'); r = r[r.size//2:]; r /= r[0] + 1e-30
        lags_s = np.arange(len(r)) / fs
        peaks, _ = find_peaks(r, distance=int(fs * 0.4))
        if not len(peaks): return None
        T = lags_s[peaks][0]
        return (1/T)*60 if T > 0 else None

    def fft_peak(sig, fs, fmin, fmax):
        n = len(sig); nf = 4 * n
        spec  = np.abs(np.fft.fft(sig * np.hanning(n), n=nf))[:nf//2]
        freqs = np.arange(nf//2) * fs / nf
        mask  = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask): return 0.0, freqs, spec
        idx = np.where(mask)[0]; pk = idx[np.argmax(spec[idx])]
        return float(freqs[pk]), freqs, spec

    print('✅ Helper functions ready')
    """))

    # ── Cell 3: Full pipeline function ──
    n.cells.append(code("""
    # Cell 3 — Full CBF Pipeline (all intermediate stages)
    def run_full_pipeline(bin_path):
        \"\"\"
        Returns a dict with every intermediate stage for visualization.
        Steps: Read → Stage-1a DC → Range FFT → SVD Clutter → Detect
               → Stage-1b DC (circle fit) → Phase Demod → Vital Signs
        \"\"\"
        meta = parse_meta(bin_path)
        na   = len(AZIMUTH_GRID)

        # 1. Read
        raw     = read_dca1000(bin_path)
        if_data = raw.reshape(NUM_RX, M_CHIRPS, N_CHIRPS).astype(np.complex64)

        # 2. CBF + Stage-1a mean-subtraction DC correction
        beamed     = apply_cbf(if_data)
        beamed_raw = beamed.copy()
        for nn in range(na):
            m = beamed[nn]
            beamed[nn] = (m - np.mean(m, axis=0, keepdims=True)).astype(np.complex64)

        # 3. Range FFT
        rfft_cube = range_fft_cube(beamed)
        rfft_raw  = range_fft_cube(beamed_raw)   # pre DC-correction, for viz

        # 4. SVD clutter removal
        rfft_pre_svd = rfft_cube.copy()
        rfft_mean    = np.zeros((na, N_FFT_RANGE//2))
        for nn in range(na):
            cleaned, _ = svd_clutter_removal(rfft_cube[nn], rank=1)
            rfft_cube[nn]  = cleaned.astype(np.complex64)
            rfft_mean[nn]  = np.mean(np.abs(rfft_cube[nn]), axis=0)

        # 5. Peak detection
        mb  = max(1, int(0.3/R_MAX * (N_FFT_RANGE//2)))
        msk = rfft_mean.copy(); msk[:, :mb] = 0
        fi  = int(np.argmax(msk))
        pai = fi // (N_FFT_RANGE//2); pri = fi % (N_FFT_RANGE//2)
        rng_m = float(RANGE_AXIS[pri]); ang_d = float(AZIMUTH_GRID[pai])

        # 6. Stage-1b: circle-fit DC correction on slow-time IQ
        slow_iq_raw       = rfft_cube[pai, :, pri].copy()
        slow_iq_corr, cfi = circle_fit_correction(slow_iq_raw)

        # 7. Phase demodulation
        dem     = demodulate(slow_iq_corr, wavelength_m=LAMBDA_C, remove_dc_in_phase=True)
        disp_mm = dem['displacement_mm']

        # 8. Vital sign extraction
        vs    = extract_vital_signs(disp_mm, fs=FS_VITAL,
                                    resp_band=(0.1, 0.5), heart_band=(0.8, 2.5),
                                    estimator='fft')
        hr_ac = autocorr_rate(vs['heart_signal_mm'], FS_VITAL)
        br_ac = autocorr_rate(vs['resp_signal_mm'],  FS_VITAL)
        br_hz, br_freqs, br_spec = fft_peak(vs['resp_signal_mm'],  FS_VITAL, 0.1, 0.5)
        hr_hz, hr_freqs, hr_spec = fft_peak(vs['heart_signal_mm'], FS_VITAL, 0.8, 2.5)

        return dict(
            meta=meta, bin_path=bin_path,
            slow_iq_raw=slow_iq_raw, slow_iq_corr=slow_iq_corr, cf_info=cfi,
            ra_pre=np.mean(np.abs(rfft_pre_svd), axis=1),
            ra_post=rfft_mean,
            disp_mm=disp_mm,
            phase_clean=dem['phase_clean_rad'],
            resp_signal=vs['resp_signal_mm'],
            heart_signal=vs['heart_signal_mm'],
            hr_fft=vs['heart_rate_bpm'], br_fft=vs['resp_rate_bpm'],
            hr_ac=hr_ac, br_ac=br_ac,
            hr_hz=hr_hz, br_hz=br_hz,
            hr_freqs=hr_freqs, hr_spec=hr_spec,
            br_freqs=br_freqs, br_spec=br_spec,
            rng_m=rng_m, ang_d=ang_d,
        )

    print('✅ Pipeline function ready')
    """))

    # ── Cell 4: Select & run 10 files ──
    n.cells.append(code("""
    # Cell 4 — Select 10 Representative Files & Run Pipeline
    # Spanning: 1.5 m – 4.5 m distance, 0° – 30° angle
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

    results = []
    print('Running pipeline on 10 files...\\n')
    for fname in SAMPLE_FILES:
        fpath = str(SINGLE_DIR / fname)
        if not os.path.exists(fpath):
            print(f'  ⚠  Not found: {fname}'); continue
        try:
            r = run_full_pipeline(fpath)
            results.append(r)
            m    = r['meta']
            hr_e = abs(r['hr_fft'] - m['hr_true'])
            br_e = abs(r['br_fft'] - m['br_true']) if m['has_br'] else float('nan')
            print(f"  ✅  {fname}")
            print(f"       True  HR={m['hr_true']} bpm  BR={m['br_true'] if m['has_br'] else '--'}")
            print(f"       Est.  HR={r['hr_fft']:.1f} bpm  BR={r['br_fft']:.1f} br/min")
            print(f"       Err   HR={hr_e:.1f} bpm  BR={'N/A' if np.isnan(br_e) else f'{br_e:.1f} br/min'}")
            print(f"       Det.  {r['rng_m']:.2f} m  @  {r['ang_d']:.1f}°\\n")
        except Exception as e:
            print(f'  ❌ {fname}: {e}\\n')

    print(f'Done — {len(results)}/10 files processed.')
    """))

    # ── Cell 5: Visualization helper ──
    n.cells.append(code(r"""
    # Cell 5 — Pipeline Visualization Helper
    C_ACCENT = '#58a6ff'; C_GREEN = '#3fb950'; C_AMBER = '#d29922'
    C_RED    = '#f85149'; C_PURPLE= '#bc8cff'; C_TEAL  = '#39d353'

    def plot_pipeline(r, idx):
        m      = r['meta']
        hr_err = abs(r['hr_fft'] - m['hr_true'])
        br_err = abs(r['br_fft'] - m['br_true']) if m['has_br'] else None
        hr_col = C_GREEN if hr_err <= 10 else (C_AMBER if hr_err <= 15 else C_RED)
        br_col = C_GREEN if (br_err is not None and br_err <= 3) else (
                 C_AMBER if (br_err is not None and br_err <= 5) else C_RED) \
                 if br_err is not None else C_ACCENT

        fig = plt.figure(figsize=(20, 14)); fig.patch.set_facecolor('#0d1117')
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.40)

        fig.suptitle(
            f"Sample {idx+1}/10  ·  {Path(r['bin_path']).name}\n"
            f"True: HR={m['hr_true']} bpm  BR={m['br_true'] if m['has_br'] else '—'} br/min"
            f"  |  Dist={m['dist_m']} m  Angle={m['angle_deg']}°",
            color='#e6edf3', fontsize=12, fontweight='bold', y=0.99)

        t = np.arange(M_CHIRPS) / FS_VITAL

        # ── P1: RA heatmap PRE-SVD ──
        ax = fig.add_subplot(gs[0, 0])
        Rp = 20 * np.log10(r['ra_pre'][:, :80] + 1e-30)
        im = ax.imshow(Rp.T, aspect='auto', origin='lower',
                       extent=[AZIMUTH_GRID[-1], AZIMUTH_GRID[0], 0, RANGE_AXIS[80]],
                       cmap='magma', vmin=np.percentile(Rp, 10))
        plt.colorbar(im, ax=ax, label='dB', pad=0.02)
        ax.set_title('Range–Azimuth  PRE-SVD', color=C_AMBER, fontsize=9)
        ax.set_xlabel('Azimuth [°]'); ax.set_ylabel('Range [m]'); ax.grid(True, alpha=0.3)

        # ── P2: RA heatmap POST-SVD ──
        ax = fig.add_subplot(gs[0, 1])
        Rq = 20 * np.log10(r['ra_post'][:, :80] + 1e-30)
        im = ax.imshow(Rq.T, aspect='auto', origin='lower',
                       extent=[AZIMUTH_GRID[-1], AZIMUTH_GRID[0], 0, RANGE_AXIS[80]],
                       cmap='plasma', vmin=np.percentile(Rq, 10))
        plt.colorbar(im, ax=ax, label='dB', pad=0.02)
        ax.axhline(r['rng_m'], color=C_GREEN,  lw=1.2, ls='--', label=f'{r["rng_m"]:.2f} m')
        ax.axvline(r['ang_d'], color=C_ACCENT, lw=1.2, ls='--', label=f'{r["ang_d"]:.1f}°')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title('Range–Azimuth  POST-SVD', color=C_GREEN, fontsize=9)
        ax.set_xlabel('Azimuth [°]'); ax.set_ylabel('Range [m]'); ax.grid(True, alpha=0.3)

        # ── P3: IQ PRE circle-fit DC correction ──
        ax = fig.add_subplot(gs[0, 2])
        Ir, Qr = np.real(r['slow_iq_raw']), np.imag(r['slow_iq_raw'])
        ax.scatter(Ir, Qr, c=np.arange(len(Ir)), cmap='viridis', s=2, alpha=0.8)
        ax.scatter(np.mean(Ir), np.mean(Qr), color=C_RED, s=90, marker='x', linewidths=2, zorder=6,
                   label=f'DC ({np.mean(Ir):.3f}, {np.mean(Qr):.3f})')
        ax.axhline(0, color='#30363d', lw=0.8); ax.axvline(0, color='#30363d', lw=0.8)
        ax.set_aspect('equal', 'box'); ax.legend(fontsize=7)
        ax.set_title('IQ Constellation  PRE DC-Correction', color=C_RED, fontsize=9)
        ax.set_xlabel('I'); ax.set_ylabel('Q'); ax.grid(True, alpha=0.3)

        # ── P4: IQ POST circle-fit DC correction ──
        ax = fig.add_subplot(gs[0, 3])
        Ic, Qc = np.real(r['slow_iq_corr']), np.imag(r['slow_iq_corr'])
        ax.scatter(Ic, Qc, c=np.arange(len(Ic)), cmap='plasma', s=2, alpha=0.8)
        rad   = float(np.mean(np.sqrt(Ic**2 + Qc**2)))
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), color=C_GREEN, lw=1.2, ls='--', alpha=0.8)
        ax.scatter(0, 0, color=C_GREEN, s=100, marker='+', linewidths=2, zorder=6,
                   label=f'DC removed\n({r["cf_info"]["dc_i"]:.3f}, {r["cf_info"]["dc_q"]:.3f})')
        ax.axhline(0, color='#30363d', lw=0.8); ax.axvline(0, color='#30363d', lw=0.8)
        ax.set_aspect('equal', 'box'); ax.legend(fontsize=7)
        ax.set_title('IQ Constellation  POST Circle-Fit DC', color=C_GREEN, fontsize=9)
        ax.set_xlabel('I'); ax.set_ylabel('Q'); ax.grid(True, alpha=0.3)

        # ── P5: Chest displacement ──
        ax = fig.add_subplot(gs[1, :2])
        ax.plot(t, r['disp_mm'],     color=C_ACCENT, lw=0.9, alpha=0.7, label='Total displacement')
        ax.plot(t, r['resp_signal'], color=C_GREEN,  lw=1.2, label=f'Resp [{r["br_fft"]:.1f} br/min]')
        ax.plot(t, r['heart_signal'],color=C_PURPLE, lw=1.0, alpha=0.9, label=f'Heart [{r["hr_fft"]:.1f} bpm]')
        ax.set_title('Phase-Demodulated Chest Displacement', color=C_ACCENT, fontsize=9)
        ax.set_xlabel('Slow time [s]'); ax.set_ylabel('Displacement [mm]')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # ── P6: Unwrapped phase ──
        ax = fig.add_subplot(gs[1, 2:])
        ax.plot(t, r['phase_clean'], color=C_TEAL, lw=1.0)
        ax.fill_between(t, r['phase_clean'], alpha=0.15, color=C_TEAL)
        ax.set_title('Unwrapped Phase (DC-removed)', color=C_TEAL, fontsize=9)
        ax.set_xlabel('Slow time [s]'); ax.set_ylabel('Phase [rad]'); ax.grid(True, alpha=0.3)

        # ── P7: Respiration spectrum ──
        ax = fig.add_subplot(gs[2, :2])
        mb = (r['br_freqs'] >= 0.05) & (r['br_freqs'] <= 0.7)
        ax.plot(r['br_freqs'][mb]*60, r['br_spec'][mb], color=C_GREEN, lw=1.5)
        ax.fill_between(r['br_freqs'][mb]*60, r['br_spec'][mb], alpha=0.2, color=C_GREEN)
        ax.axvline(r['br_hz']*60, color=br_col, lw=2, ls='--', label=f'Est BR = {r["br_fft"]:.1f}')
        if m['has_br']:
            ax.axvline(m['br_true'], color='#e6edf3', lw=1.5, ls=':', label=f'True BR = {m["br_true"]}')
        ttl = f'Respiration Spectrum  |  Err = {br_err:.1f} br/min' if br_err is not None else 'Respiration Spectrum'
        ax.set_title(ttl, color=C_GREEN, fontsize=9)
        ax.set_xlabel('Frequency [br/min]'); ax.set_ylabel('Magnitude')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # ── P8: Heart rate spectrum ──
        ax = fig.add_subplot(gs[2, 2:])
        mh = (r['hr_freqs'] >= 0.6) & (r['hr_freqs'] <= 3.0)
        ax.plot(r['hr_freqs'][mh]*60, r['hr_spec'][mh], color=C_PURPLE, lw=1.5)
        ax.fill_between(r['hr_freqs'][mh]*60, r['hr_spec'][mh], alpha=0.2, color=C_PURPLE)
        ax.axvline(r['hr_hz']*60, color=hr_col, lw=2, ls='--', label=f'Est HR = {r["hr_fft"]:.1f}')
        ax.axvline(m['hr_true'], color='#e6edf3', lw=1.5, ls=':', label=f'True HR = {m["hr_true"]}')
        ax.set_title(f'Heart Rate Spectrum  |  Err = {hr_err:.1f} bpm', color=C_PURPLE, fontsize=9)
        ax.set_xlabel('Frequency [bpm]'); ax.set_ylabel('Magnitude')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # ── QIDK text overlay ──
        qidk = (
            f"╔══════════ QIDK DISPLAY ════════════╗\n"
            f"║  Location: {r['rng_m']:.2f} m  @  {r['ang_d']:+.1f}°\n"
            f"║  HEART RATE (FFT): {r['hr_fft']:6.1f} bpm\n"
            f"║  HEART RATE (AC) : {str(round(r['hr_ac'],1)) if r['hr_ac'] else 'N/A':>6} bpm\n"
            f"║  BREATHING RATE  : {r['br_fft']:6.1f} br/min\n"
            f"║  True HR={m['hr_true']} bpm  Err={hr_err:.1f} bpm\n"
            f"╚════════════════════════════════════╝"
        )
        fig.text(0.5, 0.002, qidk, ha='center', va='bottom', fontsize=8,
                 color='#58a6ff', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22',
                           edgecolor='#58a6ff', alpha=0.92))
        plt.tight_layout(rect=[0, 0.08, 1, 0.98]); plt.show()

    print('✅ Visualization helper ready')
    """))

    # ── Markdown explanation ──
    n.cells.append(md("""
    ---
    ## 📊 Sample-by-Sample Deep Dive

    For each of the 10 recordings you will see 8 panels:

    | Panel | Content |
    |-------|---------|
    | 1 | Range–Azimuth heatmap **before** SVD clutter removal (dominant clutter peak visible) |
    | 2 | Range–Azimuth heatmap **after** SVD clutter removal (human target peak highlighted) |
    | 3 | IQ constellation **before** circle-fit DC correction (circle off-center) |
    | 4 | IQ constellation **after** circle-fit DC correction (circle centered at origin) |
    | 5 | Phase-demodulated chest displacement decomposed into breathing + heartbeat |
    | 6 | Unwrapped phase signal (after DC removal) |
    | 7 | Respiration spectrum — estimated vs true BR marker |
    | 8 | Heart rate spectrum — estimated vs true HR marker |

    The **QIDK display box** at the bottom replicates the embedded device terminal output.
    ---
    """))

    # ── Cell 6: Plot all 10 ──
    n.cells.append(code("""
    # Cell 6 — Plot All 10 Samples (8-panel each)
    for idx, r in enumerate(results):
        print(f"\\n{'='*72}")
        print(f"  SAMPLE {idx+1}/10 — {Path(r['bin_path']).name}")
        print(f"{'='*72}")
        plot_pipeline(r, idx)
    """))

    # ── Markdown: pipeline explanation ──
    n.cells.append(md(r"""
    ---
    ## 🔬 How Are We Getting Results? — Step-by-Step Explanation

    ### Step 1 · Read DCA1000 Binary
    The AWR1843 emits 77–80 GHz FMCW chirps; the DCA1000 board captures raw 16-bit
    IQ samples over LVDS. We demux 4 RX channels and reshape to
    `[NUM_RX=4, M_CHIRPS=512, N_CHIRPS=512]`.

    ### Step 2 · Stage-1a DC Offset Correction — Mean Subtraction [Alizadeh 2019]
    Static walls/furniture produce a **zero-Doppler** (constant-across-chirps) component
    at every range bin. Subtracting the temporal mean removes it:
    > `S_clean[k, b] = S[k, b] − (1/K) Σ_k S[k, b]`

    This is done **per beamformed angle** before the range FFT.

    ### Step 3 · CBF Beamforming + Range FFT
    A **Hanning window** along fast-time suppresses sidelobes.
    FFT maps each sample to a range bin:
    > `R[n] = n · c / (2 · Slope · N_FFT)`

    CBF steers the 4-element array across ±90° in 2.5° steps, giving a
    `[73 angles × 512 chirps × 256 range bins]` cube.

    ### Step 4 · SVD Rank-1 Clutter Suppression [Mercuri 2022]
    Residual static clutter is **low-rank**: its energy is concentrated in the
    first singular vector. Subtracting the rank-1 approximation isolates the
    time-varying Doppler signal from the human:
    > `S_clean = S − σ₁ u₁ v₁ᴴ`

    **Panel 1 vs Panel 2** shows the dramatic suppression of the static peak,
    revealing the weaker human target.

    ### Step 5 · Target Detection
    The peak in the range–azimuth mean-power map (excluding near-field < 0.3 m)
    gives the estimated **(range, angle)** of the subject.

    ### Step 6 · Stage-1b DC Correction — Circle Fit [Mercuri 2022]
    The slow-time IQ trajectory at the target range bin traces a **circular arc**
    in the complex plane. Its center = residual DC offset (from hardware + multipath).
    Taubin algebraic circle fitting finds the center `(cx, cy)` and subtracts it:
    > `s_corrected(t) = s(t) − (cx + j·cy)`

    **Panel 3 vs Panel 4** shows the IQ circle shifting to be centered at the origin.

    ### Step 7 · Phase Demodulation
    With the IQ circle centered:
    > `φ(t) = arctan2(Q(t), I(t))`  →  unwrap  →  subtract mean
    > `d(t) = φ(t) · λ / (4π)` [mm]

    Panel 5 shows the resulting displacement signal containing both
    breathing (~8 mm amplitude, slow) and heartbeat (~0.3 mm amplitude, fast) components.

    ### Step 8 · Vital Sign Extraction
    Two **Butterworth bandpass filters** isolate each band:

    | Vital sign | Band | Typical range |
    |------------|------|---------------|
    | Breathing rate (BR) | 0.1 – 0.5 Hz | 6–30 br/min |
    | Heart rate (HR)     | 0.8 – 2.5 Hz | 48–150 bpm  |

    The **dominant FFT peak** in each filtered band gives the final estimate.
    An **autocorrelation estimator** provides a secondary cross-check.

    Errors are then computed as `|estimated − true_from_filename|`.
    ---
    """))

    # ── Cell 7: Summary table ──
    n.cells.append(code("""
    # Cell 7 — Summary Error Table
    print('\\n' + '='*82)
    print('  SUMMARY — 10 SAMPLES')
    print('='*82)
    hdr = f"{'#':>2}  {'File':<40} {'True HR':>7} {'Est HR':>7} {'HR Err':>7}  {'True BR':>8} {'Est BR':>8} {'BR Err':>8}"
    print(hdr); print('-'*82)

    hr_errs, br_errs = [], []
    for i, r in enumerate(results):
        m    = r['meta']
        fn   = Path(r['bin_path']).name[:40]
        hr_e = abs(r['hr_fft'] - m['hr_true']); hr_errs.append(hr_e)
        if m['has_br']:
            br_e = abs(r['br_fft'] - m['br_true']); br_errs.append(br_e)
            br_s = f"{m['br_true']:>8.1f} {r['br_fft']:>8.1f} {br_e:>8.1f}"
        else:
            br_s = f"{'--':>8} {r['br_fft']:>8.1f} {'N/A':>8}"
        print(f"{i+1:>2}  {fn:<40} {m['hr_true']:>7} {r['hr_fft']:>7.1f} {hr_e:>7.1f}  {br_s}")

    print('-'*82)
    print(f"  HR MAE  = {np.mean(hr_errs):.2f} bpm    (n={len(hr_errs)})")
    print(f"  HR RMSE = {np.sqrt(np.mean(np.array(hr_errs)**2)):.2f} bpm")
    if br_errs:
        print(f"  BR MAE  = {np.mean(br_errs):.2f} br/min (n={len(br_errs)})")
        print(f"  BR RMSE = {np.sqrt(np.mean(np.array(br_errs)**2)):.2f} br/min")
    print('='*82)
    """))

    # ── Cell 8: QIDK terminal display ──
    n.cells.append(code(r"""
    # Cell 8 — QIDK Terminal-Style Display for All 10 Samples
    GR = '\033[92m'; CY = '\033[96m'; YE = '\033[93m'
    RE = '\033[91m'; BO = '\033[1m';  RS = '\033[0m'; WH = '\033[97m'

    for i, r in enumerate(results):
        m      = r['meta']
        hr_e   = abs(r['hr_fft'] - m['hr_true'])
        hc     = GR if hr_e <= 5 else (YE if hr_e <= 10 else RE)
        br_e   = abs(r['br_fft'] - m['br_true']) if m['has_br'] else None
        bc     = (GR if br_e is not None and br_e <= 2 else
                  YE if br_e is not None and br_e <= 4 else RE) if br_e is not None else CY
        sep    = '='*68
        print(f'\n{BO}{WH}{sep}{RS}')
        print(f'{BO}{CY}  FMCW RADAR — QIDK VITAL SIGNS  [Sample {i+1}/10]{RS}')
        print(f'{BO}{WH}{sep}{RS}')
        print(f'  {YE}File   : {WH}{Path(r["bin_path"]).name}{RS}')
        print(f'  {YE}Method : {WH}CBF (Conventional Beamforming){RS}')
        print(f'  {YE}Detect : {WH}{r["rng_m"]:.2f} m  |  {r["ang_d"]:+.1f}°{RS}')
        print(f'{BO}{WH}{"─"*68}{RS}')
        print(f'\n  {BO}{"HEART RATE":^30}  {"BREATHING RATE":^30}{RS}')
        print(f'  {BO}{hc}{"FFT: "+f"{r[\"hr_fft\"]:.1f} bpm":^30}{RS}  '
              f'{BO}{bc}{"FFT: "+f"{r[\"br_fft\"]:.1f} br/min":^30}{RS}')
        ach = f'{r["hr_ac"]:.1f} bpm'    if r['hr_ac'] else 'N/A'
        acb = f'{r["br_ac"]:.1f} br/min' if r['br_ac'] else 'N/A'
        print(f'  {"AC:  "+ach:^32}  {"AC:  "+acb:^32}')
        print(f'  {hc}{"True HR: "+str(m["hr_true"])+" bpm  |  Err: "+f"{hr_e:.1f} bpm":^64}{RS}')
        if br_e is not None:
            print(f'  {bc}{"True BR: "+str(m["br_true"])+f" br/min  |  Err: {br_e:.1f} br/min":^64}{RS}')
        print(f'{BO}{WH}{sep}{RS}\n')
    """))

    # ── Cell 9: Summary plots ──
    n.cells.append(code("""
    # Cell 9 — Error Summary Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('10-Sample Error Analysis — CBF Pipeline', color='#e6edf3', fontsize=14, fontweight='bold')

    lbls   = [f'S{i+1}' for i in range(len(results))]
    hr_est = [r['hr_fft']        for r in results]
    hr_tr  = [r['meta']['hr_true'] for r in results]
    hr_ev  = [abs(a-b) for a, b in zip(hr_est, hr_tr)]

    bi     = [i for i, r in enumerate(results) if r['meta']['has_br']]
    br_est = [results[i]['br_fft']            for i in bi]
    br_tr  = [results[i]['meta']['br_true']   for i in bi]
    br_ev  = [abs(a-b) for a, b in zip(br_est, br_tr)]
    bl     = [f'S{i+1}' for i in bi]

    x = np.arange(len(results))
    for ax in axes: ax.set_facecolor('#161b22')

    # True vs Est HR
    axes[0].bar(x-0.2, hr_tr,  0.35, label='True HR', color='#3fb950', alpha=0.85)
    axes[0].bar(x+0.2, hr_est, 0.35, label='Est HR',  color='#58a6ff', alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(lbls, fontsize=9)
    axes[0].set_ylabel('Heart Rate [bpm]')
    axes[0].set_title('True vs Estimated HR', color='#58a6ff', fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')

    # HR error per sample
    cols = ['#3fb950' if e<=5 else '#d29922' if e<=10 else '#f85149' for e in hr_ev]
    axes[1].bar(x, hr_ev, color=cols, alpha=0.9)
    axes[1].axhline(np.mean(hr_ev), color='#e6edf3', lw=1.5, ls='--', label=f'MAE={np.mean(hr_ev):.2f}')
    axes[1].axhline(10, color='#d29922', lw=1.0, ls=':', alpha=0.6, label='±10 bpm')
    axes[1].set_xticks(x); axes[1].set_xticklabels(lbls, fontsize=9)
    axes[1].set_ylabel('|HR Error| [bpm]')
    axes[1].set_title('HR Absolute Error per Sample', color='#f85149', fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')

    # True vs Est BR
    xb = np.arange(len(bi))
    axes[2].bar(xb-0.2, br_tr,  0.35, label='True BR', color='#39d353', alpha=0.85)
    axes[2].bar(xb+0.2, br_est, 0.35, label='Est BR',  color='#bc8cff', alpha=0.85)
    axes[2].set_xticks(xb); axes[2].set_xticklabels(bl, fontsize=9)
    axes[2].set_ylabel('Breathing Rate [br/min]')
    axes[2].set_title('True vs Estimated BR', color='#bc8cff', fontsize=11)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout(); plt.show()
    print(f'  HR MAE={np.mean(hr_ev):.2f} bpm  RMSE={np.sqrt(np.mean(np.array(hr_ev)**2)):.2f} bpm')
    if br_ev: print(f'  BR MAE={np.mean(br_ev):.2f} br/min  RMSE={np.sqrt(np.mean(np.array(br_ev)**2)):.2f}')
    """))

    return n


# ===========================================================================
#  NOTEBOOK 2 — Batch All 320 Files + QIDK
# ===========================================================================

def make_nb2():
    n = nb()

    n.cells.append(md("""
    # 🗂️ FMCW Radar — Batch Processing All 320 Recordings
    ## DC Offset Correction + Static Clutter Suppression → QIDK HR/BR → Error Metrics

    **Dataset:** AWR1843 / DCA1000 Single-Subject (320 `.bin` files)
    **Method:** CBF (Conventional Beamforming) — no 2D MUSIC

    ### What this notebook does
    1. Runs the full pipeline on **every file** in `Single_Subject/`
    2. Applies **DC offset correction** (mean subtraction + circle fit) and **SVD clutter suppression**
    3. Estimates **HR and BR** from phase-demodulated chest displacement
    4. Displays **QIDK-style** HR/BR output with color-coded error
    5. Computes **MAE, RMSE, within-band %** for HR and BR
    6. Generates **aggregate plots**: scatter, histogram, grouped bar by distance/angle
    7. Exports **CSV** of all results
    ---
    """))

    # ── Cell 1: Imports ──
    n.cells.append(code("""
    # Cell 1 — Imports & Constants
    import os, sys, re, csv, time
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.signal import butter, filtfilt, find_peaks

    sys.path.insert(0, os.path.abspath('.'))
    from dc_offset_correction  import circle_fit_correction
    from clutter_suppression   import svd_clutter_removal
    from phase_demodulation    import demodulate
    from vital_sign_extraction import extract_vital_signs

    matplotlib.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',   'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
        'text.color': '#c9d1d9',       'grid.color': '#21262d',
        'grid.linestyle': '--',        'grid.linewidth': 0.5,
        'font.family': 'monospace',    'figure.dpi': 110,
    })

    F_MIN, F_MAX = 77e9, 79.982e9
    C = 3e8; BW = F_MAX - F_MIN; TM = 100e-6; F_ADC = 600e3
    LAMBDA_C   = C / ((F_MIN + F_MAX) / 2.0)
    LAMBDA_MAX = C / F_MIN; D_ANT = LAMBDA_MAX / 2
    R_MAX = (C * TM / 2.0) * (F_ADC / BW)
    N_CHIRPS = M_CHIRPS = 512; NUM_RX = 4; N_FFT_RANGE = 512
    FS_VITAL  = M_CHIRPS / 25.6

    AZIMUTH_GRID = np.arange(90, -90 - 2.5, -2.5)
    RANGE_AXIS   = np.linspace(0, R_MAX, N_FFT_RANGE // 2)
    WIN_RANGE    = np.hanning(N_CHIRPS)
    SINGLE_DIR   = Path('./Single_Subject')

    print(f'✅ R_MAX={R_MAX:.2f} m  FS={FS_VITAL:.1f} Hz  Total bins={N_FFT_RANGE//2}')
    """))

    # ── Cell 2: All helpers + pipeline ──
    n.cells.append(code(r"""
    # Cell 2 — Helper Functions + Batch Pipeline
    _RE = re.compile(r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin')

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
        W = np.exp(-1j * 2 * np.pi * rx_pos[:, None] * np.sin(ang_rad)[None, :]) / NUM_RX
        return np.einsum('ra,rcs->acs', W, if_data)

    def autocorr_rate(sig, fs):
        x = sig - np.mean(sig)
        r = np.correlate(x, x, mode='full'); r = r[r.size//2:]; r /= r[0]+1e-30
        peaks, _ = find_peaks(r, distance=int(fs*0.4))
        if not len(peaks): return None
        T = peaks[0] / fs
        return (1/T)*60 if T > 0 else None

    def run_cbf_pipeline_fast(bin_path):
        \"\"\"Optimised single-pass CBF pipeline (no intermediate storage).\"\"\"
        na = len(AZIMUTH_GRID)
        raw     = read_dca1000(bin_path)
        if_data = raw.reshape(NUM_RX, M_CHIRPS, N_CHIRPS).astype(np.complex64)

        beamed = apply_cbf(if_data)
        for nn in range(na):
            m = beamed[nn]
            beamed[nn] = (m - np.mean(m, axis=0, keepdims=True)).astype(np.complex64)

        rfft_mean = np.zeros((na, N_FFT_RANGE//2))
        rp_cube   = np.zeros((na, M_CHIRPS, N_FFT_RANGE//2), dtype=np.complex64)
        for nn in range(na):
            Xf = np.fft.fft(beamed[nn] * WIN_RANGE[np.newaxis, :], n=N_FFT_RANGE, axis=1)
            raw_rp = Xf[:, :N_FFT_RANGE//2].astype(np.complex64)
            cleaned, _ = svd_clutter_removal(raw_rp, rank=1)
            rp_cube[nn]   = cleaned.astype(np.complex64)
            rfft_mean[nn] = np.mean(np.abs(rp_cube[nn]), axis=0)

        mb  = max(1, int(0.3/R_MAX * (N_FFT_RANGE//2)))
        msk = rfft_mean.copy(); msk[:, :mb] = 0
        fi  = int(np.argmax(msk))
        pai = fi // (N_FFT_RANGE//2); pri = fi % (N_FFT_RANGE//2)

        slow_iq      = rp_cube[pai, :, pri]
        slow_iq_corr, _ = circle_fit_correction(slow_iq)
        dem          = demodulate(slow_iq_corr, wavelength_m=LAMBDA_C, remove_dc_in_phase=True)
        vs           = extract_vital_signs(dem['displacement_mm'], fs=FS_VITAL,
                                           resp_band=(0.1, 0.5), heart_band=(0.8, 2.5),
                                           estimator='fft')
        hr_ac = autocorr_rate(vs['heart_signal_mm'], FS_VITAL)
        br_ac = autocorr_rate(vs['resp_signal_mm'],  FS_VITAL)

        return dict(
            hr_fft=vs['heart_rate_bpm'], br_fft=vs['resp_rate_bpm'],
            hr_ac=hr_ac, br_ac=br_ac,
            rng_m=float(RANGE_AXIS[pri]), ang_d=float(AZIMUTH_GRID[pai]),
        )

    print('✅ Pipeline + helpers ready')
    """))

    # ── Cell 3: Batch run ──
    n.cells.append(code("""
    # Cell 3 — Batch Run: All 320 Files
    bin_files = sorted(SINGLE_DIR.glob('*.bin'))
    bin_files = [p for p in bin_files if _RE.match(p.name)]
    print(f'Found {len(bin_files)} valid .bin files\\n')

    all_results = []
    errors_list  = []
    t_start      = time.time()

    for i, fpath in enumerate(bin_files):
        meta = parse_meta(str(fpath))
        try:
            res = run_cbf_pipeline_fast(str(fpath))
            hr_e = abs(res['hr_fft'] - meta['hr_true'])
            br_e = abs(res['br_fft'] - meta['br_true']) if meta['has_br'] else None
            entry = dict(
                filename  = fpath.name,
                dist_m    = meta['dist_m'],
                angle_deg = meta['angle_deg'],
                hr_true   = meta['hr_true'],
                br_true   = meta['br_true'],
                has_br    = meta['has_br'],
                hr_fft    = res['hr_fft'],
                br_fft    = res['br_fft'],
                hr_ac     = res['hr_ac'],
                br_ac     = res['br_ac'],
                hr_err    = hr_e,
                br_err    = br_e,
                rng_m     = res['rng_m'],
                ang_d     = res['ang_d'],
            )
            all_results.append(entry)
            if (i+1) % 50 == 0 or i == 0:
                elapsed = time.time() - t_start
                print(f'  [{i+1:>3}/{len(bin_files)}]  elapsed={elapsed:.1f}s  '
                      f'last HR={res["hr_fft"]:.1f} bpm  err={hr_e:.1f}')
        except Exception as e:
            errors_list.append(dict(file=fpath.name, error=str(e)))
            print(f'  ❌ {fpath.name}: {e}')

    elapsed = time.time() - t_start
    print(f'\\n✅ Done — {len(all_results)}/{len(bin_files)} processed in {elapsed:.1f}s')
    print(f'   Errors: {len(errors_list)}')
    """))

    # ── Cell 4: Aggregate error stats ──
    n.cells.append(code("""
    # Cell 4 — Aggregate Error Statistics
    import numpy as np

    hr_errs = np.array([r['hr_err'] for r in all_results])
    br_rows = [r for r in all_results if r['has_br'] and r['br_err'] is not None]
    br_errs = np.array([r['br_err'] for r in br_rows])

    def pct_within(errs, thresh):
        return 100.0 * np.mean(np.array(errs) <= thresh)

    print('='*60)
    print('  HEART RATE METRICS')
    print(f'  n            = {len(hr_errs)}')
    print(f'  MAE          = {np.mean(hr_errs):.2f} bpm')
    print(f'  RMSE         = {np.sqrt(np.mean(hr_errs**2)):.2f} bpm')
    print(f'  Median error = {np.median(hr_errs):.2f} bpm')
    print(f'  Max error    = {np.max(hr_errs):.2f} bpm')
    print(f'  Within  5 bpm: {pct_within(hr_errs,  5):.1f}%')
    print(f'  Within 10 bpm: {pct_within(hr_errs, 10):.1f}%')
    print(f'  Within 15 bpm: {pct_within(hr_errs, 15):.1f}%')
    print()
    print('  BREATHING RATE METRICS')
    print(f'  n            = {len(br_errs)}')
    if len(br_errs):
        print(f'  MAE          = {np.mean(br_errs):.2f} br/min')
        print(f'  RMSE         = {np.sqrt(np.mean(br_errs**2)):.2f} br/min')
        print(f'  Median error = {np.median(br_errs):.2f} br/min')
        print(f'  Within  2 br/min: {pct_within(br_errs, 2):.1f}%')
        print(f'  Within  4 br/min: {pct_within(br_errs, 4):.1f}%')
    print('='*60)
    """))

    # ── Cell 5: QIDK display sample ──
    n.cells.append(code(r"""
    # Cell 5 — QIDK Terminal Display (every file)
    # (Shows colored output in Jupyter; full terminal colors in TTY)
    GR = '\033[92m'; CY = '\033[96m'; YE = '\033[93m'
    RE = '\033[91m'; BO = '\033[1m';  RS = '\033[0m'; WH = '\033[97m'

    print(f'{BO}{WH}{"="*68}{RS}')
    print(f'{BO}{CY}  FMCW QIDK BATCH RESULTS — {len(all_results)} FILES{RS}')
    print(f'{BO}{WH}{"="*68}{RS}')
    hdr = f"{'File':<42} {'True HR':>7} {'Est HR':>7} {'Err':>6}  {'BR True':>7} {'BR Est':>7} {'Err':>6}"
    print(f'{BO}{WH}{hdr}{RS}')
    print('─'*90)

    for r in all_results:
        hr_e  = r['hr_err']
        hc    = GR if hr_e <= 5 else (YE if hr_e <= 10 else RE)
        br_e  = r['br_err']
        bc    = (GR if br_e is not None and br_e <= 2 else
                 YE if br_e is not None and br_e <= 4 else RE) if br_e is not None else ''
        fn    = r['filename'][:42]
        br_ts = f"{r['br_true']:>7.1f}" if r['has_br'] else f"{'--':>7}"
        br_es = f"{r['br_fft']:>7.1f}"
        br_er = f"{br_e:>6.1f}" if br_e is not None else f"{'N/A':>6}"
        print(f"{fn:<42} {r['hr_true']:>7} {hc}{r['hr_fft']:>7.1f}{RS} {hc}{hr_e:>6.1f}{RS}  "
              f"{br_ts} {bc}{br_es}{RS} {bc}{br_er}{RS}")

    print(f'\n{BO}{WH}{"="*68}{RS}')
    print(f'  HR MAE  = {np.mean(hr_errs):.2f} bpm   '
          f'BR MAE  = {np.mean(br_errs):.2f} br/min' if len(br_errs) else '')
    print(f'{BO}{WH}{"="*68}{RS}')
    """))

    # ── Cell 6: Aggregate plots ──
    n.cells.append(code("""
    # Cell 6 — Aggregate Plots (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f'Aggregate Results — {len(all_results)} Files — CBF Pipeline',
                 color='#e6edf3', fontsize=14, fontweight='bold')
    for ax in axes.flat: ax.set_facecolor('#161b22')

    hr_true_all = np.array([r['hr_true'] for r in all_results])
    hr_est_all  = np.array([r['hr_fft']  for r in all_results])
    hr_err_all  = np.abs(hr_true_all - hr_est_all)

    # P1: Scatter HR true vs estimated
    ax = axes[0, 0]
    sc = ax.scatter(hr_true_all, hr_est_all, c=hr_err_all, cmap='RdYlGn_r',
                    vmin=0, vmax=20, s=18, alpha=0.7)
    lo = min(hr_true_all.min(), hr_est_all.min()) - 5
    hi = max(hr_true_all.max(), hr_est_all.max()) + 5
    ax.plot([lo, hi], [lo, hi], color='#8b949e', lw=1.5, ls='--', label='Perfect estimate')
    ax.plot([lo, hi], [lo+10, hi+10], color='#d29922', lw=1, ls=':', alpha=0.6)
    ax.plot([lo, hi], [lo-10, hi-10], color='#d29922', lw=1, ls=':', alpha=0.6, label='±10 bpm band')
    plt.colorbar(sc, ax=ax, label='|HR Error| [bpm]', pad=0.02)
    ax.set_xlabel('True HR [bpm]'); ax.set_ylabel('Estimated HR [bpm]')
    ax.set_title('HR: True vs Estimated (all 320)', color='#58a6ff', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # P2: HR error histogram
    ax = axes[0, 1]
    ax.hist(hr_err_all, bins=30, color='#58a6ff', alpha=0.85, edgecolor='#30363d')
    ax.axvline(np.mean(hr_err_all), color='#3fb950', lw=2, ls='--', label=f'MAE={np.mean(hr_err_all):.2f}')
    ax.axvline(np.median(hr_err_all), color='#d29922', lw=2, ls=':', label=f'Median={np.median(hr_err_all):.2f}')
    ax.set_xlabel('|HR Error| [bpm]'); ax.set_ylabel('Count')
    ax.set_title('HR Error Distribution', color='#58a6ff', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # P3: MAE by distance group
    ax = axes[1, 0]
    dist_vals = sorted(set(r['dist_m'] for r in all_results))
    mae_by_dist = [np.mean([r['hr_err'] for r in all_results if r['dist_m'] == d]) for d in dist_vals]
    cnt_by_dist = [sum(1 for r in all_results if r['dist_m'] == d) for d in dist_vals]
    bars = ax.bar([str(d)+' m' for d in dist_vals], mae_by_dist, color='#bc8cff', alpha=0.88)
    for bar, cnt in zip(bars, cnt_by_dist):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={cnt}', ha='center', va='bottom', fontsize=8, color='#8b949e')
    ax.set_xlabel('Distance group'); ax.set_ylabel('HR MAE [bpm]')
    ax.set_title('HR MAE by Distance Group', color='#bc8cff', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # P4: MAE by angle group
    ax = axes[1, 1]
    ang_vals = sorted(set(r['angle_deg'] for r in all_results))
    mae_by_ang = [np.mean([r['hr_err'] for r in all_results if r['angle_deg'] == a]) for a in ang_vals]
    cnt_by_ang = [sum(1 for r in all_results if r['angle_deg'] == a) for a in ang_vals]
    bars = ax.bar([str(a)+'°' for a in ang_vals], mae_by_ang, color='#39d353', alpha=0.88)
    for bar, cnt in zip(bars, cnt_by_ang):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={cnt}', ha='center', va='bottom', fontsize=8, color='#8b949e')
    ax.set_xlabel('Angle group'); ax.set_ylabel('HR MAE [bpm]')
    ax.set_title('HR MAE by Angle Group', color='#39d353', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(); plt.show()
    """))

    # ── Cell 7: BR plots ──
    n.cells.append(code("""
    # Cell 7 — Breathing Rate Aggregate Plots
    if len(br_rows) > 0:
        br_true_all = np.array([r['br_true'] for r in br_rows])
        br_est_all  = np.array([r['br_fft']  for r in br_rows])
        br_err_all  = np.abs(br_true_all - br_est_all)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle(f'Breathing Rate Results — {len(br_rows)} files with BR > 0',
                     color='#e6edf3', fontsize=13, fontweight='bold')
        for ax in axes: ax.set_facecolor('#161b22')

        # Scatter
        sc = axes[0].scatter(br_true_all, br_est_all, c=br_err_all,
                             cmap='RdYlGn_r', vmin=0, vmax=8, s=22, alpha=0.75)
        lo = min(br_true_all.min(), br_est_all.min()) - 2
        hi = max(br_true_all.max(), br_est_all.max()) + 2
        axes[0].plot([lo, hi], [lo, hi], color='#8b949e', lw=1.5, ls='--', label='Perfect')
        axes[0].plot([lo, hi], [lo+3, hi+3], color='#d29922', lw=1, ls=':', alpha=0.6)
        axes[0].plot([lo, hi], [lo-3, hi-3], color='#d29922', lw=1, ls=':', alpha=0.6, label='±3 br/min')
        plt.colorbar(sc, ax=axes[0], label='|BR Error| [br/min]', pad=0.02)
        axes[0].set_xlabel('True BR [br/min]'); axes[0].set_ylabel('Estimated BR [br/min]')
        axes[0].set_title('BR: True vs Estimated', color='#39d353', fontsize=11)
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        # Histogram
        axes[1].hist(br_err_all, bins=20, color='#39d353', alpha=0.85, edgecolor='#30363d')
        axes[1].axvline(np.mean(br_err_all), color='#58a6ff', lw=2, ls='--',
                        label=f'MAE={np.mean(br_err_all):.2f}')
        axes[1].set_xlabel('|BR Error| [br/min]'); axes[1].set_ylabel('Count')
        axes[1].set_title('BR Error Distribution', color='#39d353', fontsize=11)
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()
    else:
        print('No files with non-zero BR found.')
    """))

    # ── Cell 8: CSV export ──
    n.cells.append(code("""
    # Cell 8 — Export Results to CSV
    csv_path = './batch_results_320.csv'
    fieldnames = ['filename','dist_m','angle_deg','hr_true','hr_fft','hr_ac',
                  'hr_err','br_true','br_fft','br_ac','br_err','has_br','rng_m','ang_d']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f'✅ Results saved → {csv_path}  ({len(all_results)} rows)')
    print(f'   Columns: {", ".join(fieldnames)}')
    """))

    return n


# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    nb1 = make_nb1()
    nb1_path = os.path.join(NB_DIR, 'pipeline_10_samples.ipynb')
    with open(nb1_path, 'w', encoding='utf-8') as f:
        nbf.write(nb1, f)
    print(f'✅ Written: {nb1_path}')

    nb2 = make_nb2()
    nb2_path = os.path.join(NB_DIR, 'batch_all_320_qidk.ipynb')
    with open(nb2_path, 'w', encoding='utf-8') as f:
        nbf.write(nb2, f)
    print(f'✅ Written: {nb2_path}')
    print('\\nDone. Run each notebook with: jupyter notebook')
