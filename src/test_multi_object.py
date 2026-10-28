"""
test_multi_object.py
====================
Multi-Object Detection, Localization & Living vs Non-Living Classification
using 2D MUSIC Super-Resolution Algorithm for FMCW Radar (AWR1843 / DCA1000).

Major Objectives:
  1. Joint 2D Range-Azimuth Super-Resolution Localization using 2D MUSIC.
  2. Distinguishing multiple objects in a single radar frame as LIVING (Human)
     or NON-LIVING (Static Object) via slow-time micromotion & vital sign periodicity.
  3. Contact-free extraction of Heart Rate (HR) and Breathing Rate (BR) for living targets.
  4. Generation of publication-grade multi-panel diagnostic plots and heatmaps.

Ground Truth Targets:
  - Person A   :  1.20 m,   0.0 deg,  HR=72 bpm,  BR=15 br/min  [LIVING]
  - Person B   :  2.00 m, +15.0 deg,  HR=55 bpm,  BR=12 br/min  [LIVING]
  - Person C   :  1.60 m, -20.0 deg,  HR=90 bpm,  BR=20 br/min  [LIVING]
  - Chair D    :  2.40 m, -10.0 deg,  Static reflector          [NON-LIVING]
  - Wall/Box E :  0.80 m, +25.0 deg,  Static obstacle           [NON-LIVING]
"""

import sys, os, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

# Ensure local imports
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SCRIPT_DIR)
from music_2d import compute_music_2d_fast, detect_targets_music

# Try importing matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Radar parameters (AWR1843 / DCA1000) ──────────────────────────────────────
F_MIN     = 77.0e9
F_MAX     = 79.982e9
BW        = F_MAX - F_MIN
TM        = 100e-6
FC        = (F_MIN + F_MAX) / 2.0
SPEED_C   = 3e8
LAMBDA    = SPEED_C / FC
D_ANT     = LAMBDA / 2.0
FS_ADC    = 600e3
SLOPE     = BW / TM

NUM_RX    = 4
N_CHIRPS  = 512
N_FAST    = 256
N_FFT     = 512

# Physical unambiguous range for complex IQ sampling (f_IF in [0, FS_ADC])
R_MAX      = (SPEED_C * TM / 2.0) * (FS_ADC / BW)   # = 3.018 m
FS_SLOW    = N_CHIRPS / 25.6                        # = 20.0 Hz
RANGE_AXIS = np.linspace(0, R_MAX, N_FFT, endpoint=False)
AZ_GRID    = np.arange(-60.0, 61.0, 1.0)

# ── Ground Truth Multi-Target Scene (Living + Non-Living) ─────────────────────
GT = [
    dict(id='A', label='Person A',   range_m=1.20, angle_deg=  0.0, amp=1.00, hr=72.0, br=15.0, is_living=True),
    dict(id='B', label='Person B',   range_m=2.00, angle_deg= 15.0, amp=0.80, hr=55.0, br=12.0, is_living=True),
    dict(id='C', label='Person C',   range_m=1.60, angle_deg=-20.0, amp=0.65, hr=90.0, br=20.0, is_living=True),
    dict(id='D', label='Chair D',    range_m=2.40, angle_deg=-10.0, amp=0.85, hr=0.0,  br=0.0,  is_living=False),
    dict(id='E', label='Wall/Box E', range_m=0.80, angle_deg= 25.0, amp=1.10, hr=0.0,  br=0.0,  is_living=False),
]

# ── ANSI Terminal Colours ─────────────────────────────────────────────────────
cW = '\033[97m'; cB = '\033[1m'; cY = '\033[93m'; cC = '\033[96m'
cG = '\033[92m'; cR = '\033[91m'; cM = '\033[95m'; cX = '\033[0m'
cDIM = '\033[2m'

SEP = "=" * 74

def print_header():
    print(f"\n{cB}{cW}{SEP}{cX}")
    print(f"{cB}{cC}  FMCW Multi-Object 2D MUSIC & Living/Non-Living Classification  {cX}")
    print(f"{cB}{cW}{SEP}{cX}")
    print(f"  {cY}Carrier  :{cW} {FC/1e9:.3f} GHz  BW={BW/1e9:.3f} GHz  "
          f"R_max={R_MAX:.2f} m  lambda={LAMBDA*1e3:.2f} mm{cX}")
    print(f"  {cY}Array    :{cW} {NUM_RX} RX  d={D_ANT*1e3:.2f} mm  Slow-FS={FS_SLOW:.1f} Hz  N_FFT={N_FFT}{cX}")
    print(f"\n  {cB}Ground Truth Targets in Scene:{cX}")
    print(f"  {'ID':>2}  {'Label':<12}  {'Category':<12}  {'Range':>7}  {'Angle':>8}  {'HR':>9}  {'BR':>11}  {'Amp':>5}")
    print(f"  {'─'*72}")
    for t in GT:
        cat_str = f"{cG}LIVING{cX}" if t['is_living'] else f"{cR}NON-LIVING{cX}"
        hr_str  = f"{t['hr']:.0f} bpm" if t['is_living'] else "---"
        br_str  = f"{t['br']:.0f} br/min" if t['is_living'] else "---"
        print(f"  {t['id']:>2}  {t['label']:<12}  {cat_str:<21}  {t['range_m']:>5.2f} m  {t['angle_deg']:>7.1f}°  "
              f"{hr_str:>9}  {br_str:>11}  {t['amp']:>5.2f}")
    print()

def generate_multi_object_iq():
    """Synthesise FMCW IQ data cube with monostatic array geometry."""
    print(f"  {cY}Step 1:{cX} Generating synthetic IQ data cube ({NUM_RX}x{N_CHIRPS}x{N_FAST})...")
    np.random.seed(42)

    t_fast = np.arange(N_FAST, dtype=np.float64) / FS_ADC
    t_slow = np.arange(N_CHIRPS, dtype=np.float64) / FS_SLOW
    cube   = np.zeros((NUM_RX, N_CHIRPS, N_FAST), dtype=np.complex128)

    for tgt in GT:
        R_0   = float(tgt['range_m'])
        theta = float(np.deg2rad(tgt['angle_deg']))
        A     = float(tgt['amp'])

        if tgt['is_living']:
            hr_hz = float(tgt['hr']) / 60.0
            br_hz = float(tgt['br']) / 60.0
            # Realistic cardiopulmonary chest displacement:
            # Respiration ~2.0 mm, heartbeat ~0.35 mm
            disp = 2.0e-3 * np.sin(2.0 * np.pi * br_hz * t_slow) + \
                   0.35e-3 * np.sin(2.0 * np.pi * hr_hz * t_slow)
        else:
            # Static non-living object: zero chest displacement
            disp = np.zeros_like(t_slow)

        for rx in range(NUM_RX):
            # Monostatic array: TX at origin, RX at rx*d.
            # Round-trip delay tau(rx) = (2*R - rx*d*sin(theta)) / c
            tau_base = (2.0 * R_0 - float(rx) * float(D_ANT) * float(np.sin(theta))) / float(SPEED_C)
            tau_disp = (2.0 * disp) / float(SPEED_C)

            for ch in range(N_CHIRPS):
                tau = tau_base + tau_disp[ch]
                cube[rx, ch] += A * np.exp(
                    1j * 2.0 * np.pi * (float(FC) * tau + float(SLOPE) * tau * t_fast)
                )

    # Add Gaussian thermal receiver noise (SNR ~ 25 dB relative to Target A)
    noise_std = float(GT[0]['amp']) / float(np.sqrt(2.0 * 10.0**(25.0 / 10.0)))
    cube += noise_std * (np.random.randn(*cube.shape) + 1j * np.random.randn(*cube.shape))
    cube  = cube.astype(np.complex64)
    print(f"    Cube shape: {cube.shape} | SNR_A ~ 25 dB | Dynamic range: {len(GT)} targets")
    return cube, t_slow

def run_range_fft(cube):
    """Compute Range FFT across all fast-time samples, preserving full complex spectrum."""
    print(f"  {cY}Step 2:{cX} Range FFT (N_FFT={N_FFT}, Hanning window across full band)...")
    win = np.hanning(N_FAST)
    rp  = np.zeros((NUM_RX, N_CHIRPS, N_FFT), dtype=np.complex64)
    for rx in range(NUM_RX):
        Xf = np.fft.fft(cube[rx] * win[np.newaxis, :], n=N_FFT, axis=1)
        rp[rx] = Xf.astype(np.complex64)
    print(f"    Range profile cube: {rp.shape} | Unambiguous range: 0 to {R_MAX:.2f} m")
    return rp

def bandpass_filter(sig, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [max(1e-4, lo/nyq), min(0.9999, hi/nyq)], btype='band')
    return filtfilt(b, a, sig)

def fft_rate_bpm(sig, fs, lo_hz, hi_hz):
    N = len(sig)
    X = np.abs(np.fft.rfft(sig * np.hanning(N), n=8*N))
    f = np.fft.rfftfreq(8*N, d=1.0/fs)
    mask = (f >= lo_hz) & (f <= hi_hz)
    if not mask.any():
        return 0.0
    return float(f[mask][np.argmax(X[mask])]) * 60.0

def classify_and_extract_vitals(peaks, rp, t_slow):
    """
    Distinguish LIVING vs NON-LIVING targets and extract cardiopulmonary rates.

    Classification Metrics:
      1. Displacement Standard Deviation (sigma_d):
         Living humans exhibit chest excursions > 0.5 mm from respiration.
         Static targets show displacement jitter < 0.05 mm.
      2. AC-to-Total Power Ratio (CSR - Clutter Suppression Ratio):
         Living targets have periodic slow-time phasor variance (AC/DC ratio > -10 dB).
         Static targets have concentrated DC phasor with AC/DC ratio < -25 dB.
      3. Vital Sign Periodicity SNR (VS-SNR):
         Ratio of peak spectral power in respiration/cardiac bands to out-of-band noise floor.
    """
    print(f"  {cY}Step 5:{cX} Living vs Non-Living Object Classification & Vital Sign Extraction...")
    results = []

    for p in peaks:
        ri = p['range_idx']
        # Beamforming steering vector to target angle
        sv = np.exp(-1j * 2.0 * np.pi * (D_ANT / LAMBDA)
                    * np.arange(NUM_RX) * np.sin(np.deg2rad(p['angle_deg'])))
        slow_iq = np.einsum('r,rc->c', sv.conj(), rp[:, :, ri]) / NUM_RX

        # Raw slow-time phase demodulation
        raw_phase = np.unwrap(np.angle(slow_iq))
        disp_mm   = (LAMBDA / (4.0 * np.pi)) * (raw_phase - np.mean(raw_phase)) * 1e3

        # 1. Displacement amplitude std
        std_disp = float(np.std(disp_mm))

        # 2. AC-to-Total Power Ratio
        p_tot = float(np.mean(np.abs(slow_iq)**2))
        p_ac  = float(np.var(slow_iq))
        ac_dc_ratio = float(10.0 * np.log10(p_ac / (p_tot + 1e-12) + 1e-12))

        # 3. Frequency Spectrum & Vital Sign Periodicity SNR
        N = len(disp_mm)
        X = np.abs(np.fft.rfft(disp_mm * np.hanning(N), n=8*N))
        f = np.fft.rfftfreq(8*N, d=1.0/FS_SLOW)

        resp_mask  = (f >= 0.10) & (f <= 0.50)   # 6 to 30 br/min
        hr_mask    = (f >= 0.80) & (f <= 2.50)   # 48 to 150 bpm
        noise_mask = (f >= 2.60) & (f <= 8.00)

        p_resp  = float(np.max(X[resp_mask])) if resp_mask.any() else 1e-12
        p_hr    = float(np.max(X[hr_mask]))   if hr_mask.any()   else 1e-12
        p_noise = float(np.median(X[noise_mask])) if noise_mask.any() else float(np.median(X))
        vital_snr = float(10.0 * np.log10((p_resp + p_hr) / (p_noise + 1e-12) + 1e-6))

        # Decision rule: Living requires significant micromotion and harmonic periodicity
        is_living = (std_disp > 0.40) and (ac_dc_ratio > -15.0) and (vital_snr >= 12.0)
        category  = "LIVING" if is_living else "NON-LIVING"

        if is_living:
            resp_sig  = bandpass_filter(disp_mm, 0.10, 0.50, FS_SLOW)
            heart_sig = bandpass_filter(disp_mm, 0.80, 2.50, FS_SLOW)
            hr_est    = fft_rate_bpm(heart_sig, FS_SLOW, 0.80, 2.50)
            br_est    = fft_rate_bpm(resp_sig,  FS_SLOW, 0.10, 0.50)
        else:
            resp_sig  = np.zeros_like(disp_mm)
            heart_sig = np.zeros_like(disp_mm)
            hr_est    = 0.0
            br_est    = 0.0

        results.append({
            **p,
            'is_living':   is_living,
            'category':    category,
            'std_disp':    std_disp,
            'ac_dc_ratio': ac_dc_ratio,
            'vital_snr':   vital_snr,
            'hr':          hr_est,
            'br':          br_est,
            'disp_mm':     disp_mm,
            'resp_sig':    resp_sig,
            'heart_sig':   heart_sig,
            'freqs':       f,
            'psd':         X,
        })
    return results

def match_with_ground_truth(results):
    """Associate detected targets with closest ground-truth objects."""
    used_ids = set()
    for res in results:
        best_gt, min_dist = None, 1e9
        for gt in GT:
            if gt['id'] in used_ids:
                continue
            # Weighted spatial distance
            dist = abs(res['range_m'] - gt['range_m']) + 0.03 * abs(res['angle_deg'] - gt['angle_deg'])
            if dist < min_dist:
                min_dist = dist
                best_gt = gt
        if best_gt and min_dist < 0.35:
            res['gt'] = best_gt
            used_ids.add(best_gt['id'])
        else:
            res['gt'] = None
    return results

def print_qidk_display(results, t_music):
    """Render structured terminal dashboard."""
    print()
    print(f"{cB}{cW}{SEP}{cX}")
    print(f"{cB}{cC}  QIDK VITAL SIGNS & OBJECT CLASSIFICATION DISPLAY  {cX}")
    print(f"{cB}{cW}{SEP}{cX}")
    print(f"  {cY}Algorithm :{cW} 2D MUSIC (Range-Azimuth Joint Super-Resolution){cX}")
    print(f"  {cY}Detections:{cW} {len(results)} objects localized in {t_music:.0f} ms{cX}")
    print(f"{cB}{cW}{'─'*74}{cX}")

    header = f"  {'#':>2}  {'Location':^16}  {'CLASSIFICATION':^15}  {'HEART RATE':^12}  {'BREATH RATE':^12}  {'Match':^7}"
    print(f"{cB}{header}{cX}")
    print(f"  {'─'*72}")

    for res in results:
        gt  = res['gt']
        loc = f"{res['range_m']:.2f}m / {res['angle_deg']:+.0f}°"
        mtag = gt['label'] if gt else "Unknown"

        if res['is_living']:
            cat_color = cG
            hr_s = f"{res['hr']:.1f} bpm"
            br_s = f"{res['br']:.1f} br/min"
            hc = cG if gt and abs(res['hr'] - gt['hr']) <= 5.0 else cY
            bc = cG if gt and abs(res['br'] - gt['br']) <= 2.0 else cY
        else:
            cat_color = cR
            hr_s = "---"
            br_s = "---"
            hc = cDIM
            bc = cDIM

        print(f"  {res['target_id']:>2}  {loc:^16}  {cB}{cat_color}{res['category']:^15}{cX}  "
              f"{cB}{hc}{hr_s:^12}{cX}  {cB}{bc}{br_s:^12}{cX}  {cM}{mtag:^7}{cX}")

    print(f"\n{cB}{cW}{'─'*74}{cX}")
    print(f"  {cB}{cY}DETAILED OBJECT ANALYSIS:{cX}")
    for res in results:
        gt = res['gt']
        tag = gt['label'] if gt else f"Target #{res['target_id']}"
        cat_badge = f"{cG}[LIVING]{cX}" if res['is_living'] else f"{cR}[NON-LIVING]{cX}"

        print(f"\n  {cB}{cM}#{res['target_id']} -- {tag} {cat_badge}{cX}")
        print(f"    Detected Loc  : {res['range_m']:.2f} m  |  {res['angle_deg']:+.1f}°  (MUSIC Peak: {res['peak_db']:.1f} dB)")
        if gt:
            r_err = abs(res['range_m'] - gt['range_m']) * 100.0
            a_err = abs(res['angle_deg'] - gt['angle_deg'])
            print(f"    True Location : {gt['range_m']:.2f} m  |  {gt['angle_deg']:+.1f}°  "
                  f"(Range Err: {r_err:.1f} cm | Angle Err: {a_err:.1f}°)")

        print(f"    Micromotion   : Disp Std = {res['std_disp']:.3f} mm  |  AC/DC Ratio = {res['ac_dc_ratio']:.1f} dB  |  Vital SNR = {res['vital_snr']:.1f} dB")
        if res['is_living']:
            hr_err = abs(res['hr'] - gt['hr']) if gt else 0.0
            br_err = abs(res['br'] - gt['br']) if gt else 0.0
            print(f"    {cB}Vital Signs   : Heart Rate = {cG}{res['hr']:.1f} bpm{cX} (err: {hr_err:.1f})  |  "
                  f"Breath Rate = {cG}{res['br']:.1f} br/min{cX} (err: {br_err:.1f})")
        else:
            print(f"    {cB}Vital Signs   : {cDIM}N/A (Static reflector / stationary clutter){cX}")

    # Accuracy Summary
    living_pairs = [(r, r['gt']) for r in results if r['gt'] and r['is_living']]
    all_pairs    = [(r, r['gt']) for r in results if r['gt']]

    print(f"\n{cB}{cW}{SEP}{cX}")
    print(f"  {cB}{cY}PIPELINE ACCURACY & PERFORMANCE BENCHMARK{cX}")
    print(f"  {'─'*72}")
    print(f"  Matched Targets : {len(all_pairs)} / {len(GT)} Ground-Truth Objects")
    class_correct = sum(1 for r, gt in all_pairs if r['is_living'] == gt['is_living'])
    print(f"  Classification  : {class_correct} / {len(all_pairs)} Correct ({class_correct/len(all_pairs)*100:.0f}%)")

    if all_pairs:
        r_errs = [abs(r['range_m'] - gt['range_m']) * 100.0 for r, gt in all_pairs]
        a_errs = [abs(r['angle_deg'] - gt['angle_deg']) for r, gt in all_pairs]
        print(f"  Range MAE       : {cB}{cG}{np.mean(r_errs):.1f} cm{cX}     (Max: {np.max(r_errs):.1f} cm)")
        print(f"  Angle MAE       : {cB}{cG}{np.mean(a_errs):.1f}°{cX}      (Max: {np.max(a_errs):.1f}°)")

    if living_pairs:
        hr_errs = [abs(r['hr'] - gt['hr']) for r, gt in living_pairs]
        br_errs = [abs(r['br'] - gt['br']) for r, gt in living_pairs]
        print(f"  Heart Rate MAE  : {cB}{cG}{np.mean(hr_errs):.2f} bpm{cX}  (Errors: {', '.join(f'{e:.1f}' for e in hr_errs)})")
        print(f"  Breath Rate MAE : {cB}{cG}{np.mean(br_errs):.2f} br/min{cX} (Errors: {', '.join(f'{e:.1f}' for e in br_errs)})")
    print(f"{cB}{cW}{SEP}{cX}\n")

def generate_visualizations(spec_music_db, cbf_map_db, results, output_path="music_2d_multi_object_analysis.png"):
    """
    Generate comprehensive 6-panel visualization figure:
      1. 2D Range-Azimuth MUSIC Super-Resolution Spectrum
      2. Conventional Beamforming (CBF) Comparison Heatmap
      3. Radar Bird's-Eye View (BEV) Living vs Non-Living Map
      4. Slow-Time Displacement Waveforms per Object
      5. Micro-Doppler Vital Sign Periodicity Power Spectra
      6. Clutter Attenuation & Classification Metrics Bar Chart
    """
    if not HAS_MATPLOTLIB:
        print("  [Warning] matplotlib not installed; skipping plot generation.")
        return None

    print(f"  {cY}Step 6:{cX} Generating publication-grade visualization graphs ({output_path})...")
    fig = plt.figure(figsize=(19, 12), dpi=140)
    fig.patch.set_facecolor('#0d1117')

    plt.rcParams.update({
        'text.color': '#e6edf3',
        'axes.labelcolor': '#e6edf3',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'axes.edgecolor': '#30363d',
        'font.family': 'sans-serif',
        'font.size': 10
    })

    # ── Panel 1: 2D MUSIC Pseudo-Spectrum Heatmap ─────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(
        spec_music_db,
        extent=[AZ_GRID[0], AZ_GRID[-1], RANGE_AXIS[-1], RANGE_AXIS[0]],
        aspect='auto',
        cmap='inferno',
        vmin=-30.0, vmax=0.0
    )
    ax1.set_ylim(0, R_MAX)
    ax1.set_title("1. 2D MUSIC Super-Resolution Spectrum", fontsize=12, fontweight='bold', color='#58a6ff')
    ax1.set_xlabel("Azimuth Angle (deg)")
    ax1.set_ylabel("Range (m)")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Pseudo-Power (dB)", color='#e6edf3')

    # Annotate detected targets
    for res in results:
        marker_color = '#3fb950' if res['is_living'] else '#f85149'
        ax1.plot(res['angle_deg'], res['range_m'], marker='o', markersize=8,
                 markeredgecolor='white', markeredgewidth=1.5, markerfacecolor=marker_color)
        lbl = res['gt']['label'] if res['gt'] else f"#{res['target_id']}"
        ax1.text(res['angle_deg'] + 2.5, res['range_m'] + 0.05, lbl,
                 color='#ffffff', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#161b22', alpha=0.8, edgecolor=marker_color))

    # ── Panel 2: Conventional Beamforming (CBF) Comparison ────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(
        cbf_map_db,
        extent=[AZ_GRID[0], AZ_GRID[-1], RANGE_AXIS[-1], RANGE_AXIS[0]],
        aspect='auto',
        cmap='magma',
        vmin=-30.0, vmax=0.0
    )
    ax2.set_ylim(0, R_MAX)
    ax2.set_title("2. Conventional Beamforming (CBF) Heatmap", fontsize=12, fontweight='bold', color='#d29922')
    ax2.set_xlabel("Azimuth Angle (deg)")
    ax2.set_ylabel("Range (m)")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Power (dB)", color='#e6edf3')

    # ── Panel 3: Radar Bird's-Eye View (BEV) Classification Map ───────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#161b22')
    # Range rings
    for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        th = np.linspace(-np.pi/3, np.pi/3, 100)
        ax3.plot(r * np.sin(th), r * np.cos(th), color='#30363d', linestyle='--', linewidth=0.8)
        ax3.text(0.05, r, f"{r:.1f}m", color='#8b949e', fontsize=8)

    # Angle radials
    for a in [-45, -30, -15, 0, 15, 30, 45]:
        rad = np.deg2rad(a)
        ax3.plot([0, 3.0 * np.sin(rad)], [0, 3.0 * np.cos(rad)], color='#30363d', linestyle=':', linewidth=0.8)

    # Plot objects
    for res in results:
        rad = np.deg2rad(res['angle_deg'])
        x = res['range_m'] * np.sin(rad)
        y = res['range_m'] * np.cos(rad)
        if res['is_living']:
            ax3.scatter(x, y, s=140, color='#3fb950', edgecolors='white', linewidth=1.5, zorder=5)
            ax3.annotate(f"{res['gt']['label'] if res['gt'] else ''}\nHR:{res['hr']:.0f} BR:{res['br']:.0f}",
                         (x, y), textcoords="offset points", xytext=(8, -5),
                         fontsize=8, color='#3fb950', fontweight='bold')
        else:
            ax3.scatter(x, y, s=140, color='#f85149', marker='s', edgecolors='white', linewidth=1.5, zorder=5)
            ax3.annotate(f"{res['gt']['label'] if res['gt'] else ''}\n[STATIC]",
                         (x, y), textcoords="offset points", xytext=(8, -5),
                         fontsize=8, color='#f85149', fontweight='bold')

    # Radar position
    ax3.scatter(0, 0, marker='^', s=160, color='#58a6ff', edgecolors='white', zorder=6)
    ax3.text(0, -0.15, "RADAR (AWR1843)", color='#58a6ff', fontsize=9, ha='center', fontweight='bold')
    ax3.set_xlim(-2.2, 2.2)
    ax3.set_ylim(-0.25, 3.2)
    ax3.set_title("3. BEV Map: Living vs Non-Living Targets", fontsize=12, fontweight='bold', color='#3fb950')
    ax3.set_xlabel("Cross-Range X (m)")
    ax3.set_ylabel("Down-Range Y (m)")
    ax3.grid(False)

    # ── Panel 4: Chest Micromotion Waveforms ───────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#161b22')
    t = np.arange(len(results[0]['disp_mm'])) / FS_SLOW
    colors = ['#3fb950', '#58a6ff', '#bc8cff', '#f85149', '#e3b341']

    for i, res in enumerate(results[:4]):
        label_text = f"{res['gt']['label'] if res['gt'] else f'#{res['target_id']}'} ({res['category']})"
        ax4.plot(t[:200], res['disp_mm'][:200] + i * 4.0, label=label_text, color=colors[i % len(colors)], linewidth=1.5)

    ax4.set_title("4. Demodulated Displacement Waveforms d(t)", fontsize=12, fontweight='bold', color='#bc8cff')
    ax4.set_xlabel("Slow Time (s)")
    ax4.set_ylabel("Displacement (mm + offset)")
    ax4.legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
    ax4.grid(True, color='#30363d', linestyle='--', alpha=0.5)

    # ── Panel 5: Micro-Doppler Vital Sign Power Spectra ────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#161b22')
    for i, res in enumerate(results[:4]):
        f = res['freqs']
        psd_db = 10 * np.log10(res['psd'] / (np.max(res['psd']) + 1e-12) + 1e-6)
        label_text = f"{res['gt']['label'] if res['gt'] else f'#{res['target_id']}'}"
        ax5.plot(f[f <= 3.0], psd_db[f <= 3.0], label=label_text, color=colors[i % len(colors)], linewidth=1.4)

    # Band indicators
    ax5.axvspan(0.10, 0.50, color='#3fb950', alpha=0.15, label='Breathing Band (0.1-0.5 Hz)')
    ax5.axvspan(0.80, 2.50, color='#f85149', alpha=0.15, label='Heartbeat Band (0.8-2.5 Hz)')
    ax5.set_title("5. Micromotion Frequency Spectra (FFT)", fontsize=12, fontweight='bold', color='#ff7b72')
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Normalized Power (dB)")
    ax5.set_xlim(0, 3.0)
    ax5.set_ylim(-40, 2)
    ax5.legend(loc='lower right', facecolor='#0d1117', edgecolor='#30363d', fontsize=7)
    ax5.grid(True, color='#30363d', linestyle='--', alpha=0.5)

    # ── Panel 6: Clutter Attenuation & Classification Metrics ──────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#161b22')
    obj_labels = [res['gt']['label'] if res['gt'] else f"#{res['target_id']}" for res in results]
    disp_stds  = [res['std_disp'] for res in results]
    vital_snrs = [res['vital_snr'] for res in results]
    bar_colors = ['#3fb950' if res['is_living'] else '#f85149' for res in results]

    x_pos = np.arange(len(obj_labels))
    bars = ax6.bar(x_pos, disp_stds, color=bar_colors, edgecolor='white', width=0.55)
    ax6.axhline(0.40, color='#e3b341', linestyle='--', linewidth=1.5, label='Living Threshold (0.40 mm)')

    for bar, snr in zip(bars, vital_snrs):
        yval = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{snr:.0f}dB",
                 ha='center', va='bottom', fontsize=8, color='#e6edf3', fontweight='bold')

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(obj_labels, rotation=20, ha='right', fontsize=9)
    ax6.set_title("6. Classification Metric: Disp Std & Vital SNR", fontsize=12, fontweight='bold', color='#e3b341')
    ax6.set_ylabel("Displacement Std (mm)")
    ax6.legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
    ax6.grid(True, axis='y', color='#30363d', linestyle='--', alpha=0.5)

    plt.tight_layout()
    abs_out = os.path.abspath(output_path)
    plt.savefig(abs_out, facecolor=fig.get_facecolor(), edgecolor='none', dpi=140)
    plt.close()
    print(f"    Visualization figure saved successfully -> {abs_out}")
    return abs_out

def main():
    print_header()

    # Step 1: Synthesise IQ Cube
    cube, t_slow = generate_multi_object_iq()

    # Step 2: Range FFT
    rp = run_range_fft(cube)

    # Step 3: 2D MUSIC Joint Range-Azimuth Super-Resolution Spectrum
    print(f"  {cY}Step 3:{cX} 2D MUSIC spectrum calculation (m1=4, range-power weighted)...")
    t0 = time.perf_counter()
    spec_music_db, _, _, K_est = compute_music_2d_fast(
        rp,
        fc=float(FC), d=float(D_ANT), c=float(SPEED_C), lambda_c=float(LAMBDA),
        range_axis=RANGE_AXIS,
        azimuth_grid=AZ_GRID,
        m1=4,
        num_targets=len(GT),
        num_targets_method="fixed",
        range_weighting=True,
    )
    t_music = (time.perf_counter() - t0) * 1e3
    print(f"    Done in {t_music:.1f} ms | Spectrum shape: {spec_music_db.shape}")

    # Generate Conventional Beamforming (CBF) heatmap for comparison
    az_rad = np.deg2rad(AZ_GRID)
    W_cbf = np.exp(-1j * 2.0 * np.pi * (D_ANT / LAMBDA) * np.arange(NUM_RX)[:, None] * np.sin(az_rad)[None, :]) / NUM_RX
    # Beamform range profile: [N_angles, N_chirps, N_range]
    cbf_rp = np.einsum('ma,mcr->acr', W_cbf.conj(), rp)
    cbf_pwr = np.mean(np.abs(cbf_rp)**2, axis=1).T  # [N_range, N_angles]
    cbf_map_db = 10.0 * np.log10(np.maximum(cbf_pwr / np.max(cbf_pwr), 1e-12))

    # Step 4: Multi-Target Peak Detection
    print(f"  {cY}Step 4:{cX} Multi-target peak detection from 2D MUSIC spectrum...")
    peaks = detect_targets_music(
        spec_music_db, RANGE_AXIS, AZ_GRID,
        num_targets=len(GT),
        min_range_m=0.4,
        min_peak_db=-8.0,
        min_range_sep_m=0.20,
        min_angle_sep_deg=4.0,
    )
    print(f"    Detected {len(peaks)} candidate targets in 2D spectrum")

    # Step 5: Classification (Living vs Non-Living) & Vital Sign Extraction
    results = classify_and_extract_vitals(peaks, rp, t_slow)

    # Step 6: Ground Truth Association & Display
    results = match_with_ground_truth(results)
    print_qidk_display(results, t_music)

    # Step 7: Visualizations & Diagnostic Heatmaps
    out_img = os.path.join(SCRIPT_DIR, "music_2d_multi_object_analysis.png")
    generate_visualizations(spec_music_db, cbf_map_db, results, output_path=out_img)

if __name__ == "__main__":
    main()
