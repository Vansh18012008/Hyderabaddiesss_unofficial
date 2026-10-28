"""
generate_batch_outputs.py
=========================
Runs the CBF pipeline on ALL 320 .bin files in Single_Subject/,
saves aggregate figures and writes metrics_320.json.

Called by build_clean_notebooks.py.  Run standalone with:
    python generate_batch_outputs.py
"""

import os, sys, re, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dc_offset_correction  import circle_fit_correction
from clutter_suppression   import svd_clutter_removal
from phase_demodulation    import demodulate
from vital_sign_extraction import extract_vital_signs

# ── Style ──────────────────────────────────────────────────────────────────
DARK = '#0d1117'; PANEL = '#161b22'; BORDER = '#30363d'; FG = '#c9d1d9'
ACCENT = '#58a6ff'; GREEN = '#3fb950'; AMBER = '#d29922'; RED = '#f85149'
PURPLE = '#bc8cff'; TEAL = '#39d353'; WHITE = '#e6edf3'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': FG,
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'text.color': FG, 'grid.color': BORDER,
    'grid.linestyle': '--', 'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans', 'figure.dpi': 130,
})

# ── Hardware constants ──────────────────────────────────────────────────────
F_MIN, F_MAX = 77e9, 79.982e9
C = 3e8; BW = F_MAX - F_MIN; TM = 100e-6; F_ADC = 600e3
LAMBDA_C   = C / ((F_MIN + F_MAX) / 2.0)
LAMBDA_MAX = C / F_MIN; D_ANT = LAMBDA_MAX / 2
R_MAX = (C * TM / 2.0) * (F_ADC / BW)
N_CHIRPS = M_CHIRPS = 512; NUM_RX = 4; N_FFT_RANGE = 512
FS_VITAL = M_CHIRPS / 25.6

AZIMUTH_GRID = np.arange(90, -90 - 2.5, -2.5)
RANGE_AXIS   = np.linspace(0, R_MAX, N_FFT_RANGE // 2)
WIN_RANGE    = np.hanning(N_CHIRPS)
SINGLE_DIR   = Path('./Single_Subject')
OUT_DIR      = Path('./pipeline_outputs')
FIG_DIR      = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

def autocorr_rate(sig, fs):
    x = sig - np.mean(sig)
    r = np.correlate(x, x, mode='full'); r = r[r.size//2:]; r /= r[0]+1e-30
    peaks, _ = find_peaks(r, distance=int(fs*0.4))
    if not len(peaks): return None
    T = peaks[0]/fs
    return round((1/T)*60, 2) if T > 0 else None

def run_fast(bin_path):
    """Optimized pipeline — single pass, no intermediate storage."""
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
        hr_fft=round(vs['heart_rate_bpm'], 2),
        br_fft=round(vs['resp_rate_bpm'],  2),
        hr_ac=hr_ac, br_ac=br_ac,
        rng_m=round(float(RANGE_AXIS[pri]), 3),
        ang_d=round(float(AZIMUTH_GRID[pai]), 2),
    )


# ── Aggregate figure generators ──────────────────────────────────────────────

def fig_hr_scatter(all_results, prefix='320'):
    hr_true = np.array([r['hr_true'] for r in all_results])
    hr_est  = np.array([r['hr_fft']  for r in all_results])
    hr_err  = np.abs(hr_true - hr_est)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    sc = ax.scatter(hr_true, hr_est, c=hr_err, cmap='RdYlGn_r', vmin=0, vmax=20,
                    s=20, alpha=0.75, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|HR Error| [bpm]', pad=0.02)
    lo = min(hr_true.min(), hr_est.min()) - 5
    hi = max(hr_true.max(), hr_est.max()) + 5
    ax.plot([lo, hi], [lo, hi], color='#8b949e', lw=2, ls='--', label='Perfect estimate')
    ax.plot([lo, hi], [lo+10, hi+10], color=AMBER, lw=1.2, ls=':', alpha=0.6)
    ax.plot([lo, hi], [lo-10, hi-10], color=AMBER, lw=1.2, ls=':', alpha=0.6, label='±10 bpm band')
    ax.set_xlabel('Ground-Truth HR [bpm]'); ax.set_ylabel('Estimated HR [bpm]')
    ax.set_title(f'Heart Rate: True vs Estimated  (n={len(all_results)})',
                 color=ACCENT, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER)
    ax.grid(True, alpha=0.25)

    # Annotation box
    mae  = np.mean(hr_err)
    rmse = np.sqrt(np.mean(hr_err**2))
    within10 = 100*np.mean(hr_err <= 10)
    ax.text(0.03, 0.97,
            f'MAE  = {mae:.2f} bpm\nRMSE = {rmse:.2f} bpm\n≤10 bpm: {within10:.1f}%',
            transform=ax.transAxes, va='top', fontsize=10,
            color=WHITE, bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL,
                                   edgecolor=ACCENT, alpha=0.9))
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_hr_scatter.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_hr_histogram(all_results, prefix='320'):
    hr_err = np.abs([r['hr_true'] - r['hr_fft'] for r in all_results])
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    n, bins, patches = ax.hist(hr_err, bins=35, edgecolor=BORDER, linewidth=0.5)
    # Color bars by magnitude
    for patch, b in zip(patches, bins):
        if b <= 5:   patch.set_facecolor(GREEN)
        elif b <= 10: patch.set_facecolor(AMBER)
        else:         patch.set_facecolor(RED)
    ax.axvline(np.mean(hr_err),   color=WHITE, lw=2.5, ls='--',
               label=f'MAE = {np.mean(hr_err):.2f} bpm')
    ax.axvline(np.median(hr_err), color=ACCENT, lw=2,   ls=':',
               label=f'Median = {np.median(hr_err):.2f} bpm')
    ax.set_xlabel('|HR Error| [bpm]'); ax.set_ylabel('Number of files')
    ax.set_title(f'Distribution of HR Estimation Errors  (n={len(all_results)})',
                 color=ACCENT, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_hr_hist.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_mae_by_distance(all_results, prefix='320'):
    dist_vals = sorted(set(r['dist_m'] for r in all_results))
    mae_hr  = [np.mean([abs(r['hr_true']-r['hr_fft']) for r in all_results if r['dist_m']==d])
               for d in dist_vals]
    cnt     = [sum(1 for r in all_results if r['dist_m']==d) for d in dist_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    bars = ax.bar([f'{d} m' for d in dist_vals], mae_hr,
                  color=PURPLE, alpha=0.88, edgecolor=BORDER)
    for bar, c, m in zip(bars, cnt, mae_hr):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.25,
                f'n={c}\n{m:.1f}', ha='center', va='bottom', fontsize=9, color=FG)
    ax.set_xlabel('Distance from Radar'); ax.set_ylabel('HR MAE [bpm]')
    ax.set_title('Heart Rate MAE by Distance Group', color=PURPLE, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_mae_dist.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_mae_by_angle(all_results, prefix='320'):
    ang_vals = sorted(set(r['angle_deg'] for r in all_results))
    mae_hr   = [np.mean([abs(r['hr_true']-r['hr_fft']) for r in all_results if r['angle_deg']==a])
                for a in ang_vals]
    cnt      = [sum(1 for r in all_results if r['angle_deg']==a) for a in ang_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    bars = ax.bar([f'{a}°' for a in ang_vals], mae_hr,
                  color=TEAL, alpha=0.88, edgecolor=BORDER)
    for bar, c, m in zip(bars, cnt, mae_hr):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.25,
                f'n={c}\n{m:.1f}', ha='center', va='bottom', fontsize=9, color=FG)
    ax.set_xlabel('Azimuth Angle'); ax.set_ylabel('HR MAE [bpm]')
    ax.set_title('Heart Rate MAE by Angle Group', color=TEAL, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_mae_angle.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_br_scatter(br_results, prefix='320'):
    if not br_results: return None
    br_true = np.array([r['br_true'] for r in br_results])
    br_est  = np.array([r['br_fft']  for r in br_results])
    br_err  = np.abs(br_true - br_est)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    sc = ax.scatter(br_true, br_est, c=br_err, cmap='RdYlGn_r', vmin=0, vmax=8,
                    s=22, alpha=0.78, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='|BR Error| [br/min]', pad=0.02)
    lo, hi = br_true.min()-1, br_true.max()+1
    ax.plot([lo, hi], [lo, hi], color='#8b949e', lw=2, ls='--', label='Perfect')
    ax.plot([lo, hi], [lo+3, hi+3], color=AMBER, lw=1.2, ls=':', alpha=0.6)
    ax.plot([lo, hi], [lo-3, hi-3], color=AMBER, lw=1.2, ls=':', alpha=0.6, label='±3 br/min')
    ax.set_xlabel('Ground-Truth BR [br/min]'); ax.set_ylabel('Estimated BR [br/min]')
    ax.set_title(f'Breathing Rate: True vs Estimated  (n={len(br_results)})',
                 color=GREEN, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=BORDER); ax.grid(True, alpha=0.25)
    mae = np.mean(br_err); rmse = np.sqrt(np.mean(br_err**2))
    ax.text(0.03, 0.97, f'MAE  = {mae:.2f} br/min\nRMSE = {rmse:.2f} br/min',
            transform=ax.transAxes, va='top', fontsize=10, color=WHITE,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL, edgecolor=GREEN, alpha=0.9))
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_br_scatter.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_br_histogram(br_results, prefix='320'):
    if not br_results: return None
    br_err = np.abs([r['br_true']-r['br_fft'] for r in br_results])
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    n, bins, patches = ax.hist(br_err, bins=25, edgecolor=BORDER, linewidth=0.5)
    for patch, b in zip(patches, bins):
        patch.set_facecolor(GREEN if b<=2 else AMBER if b<=4 else RED)
    ax.axvline(np.mean(br_err), color=WHITE, lw=2.5, ls='--',
               label=f'MAE = {np.mean(br_err):.2f} br/min')
    ax.set_xlabel('|BR Error| [br/min]'); ax.set_ylabel('Number of files')
    ax.set_title(f'Distribution of BR Estimation Errors  (n={len(br_results)})',
                 color=GREEN, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor=BORDER); ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_br_hist.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


def fig_within_band(all_results, prefix='320'):
    """Stacked bar: % of files within various error thresholds."""
    hr_err = np.array([abs(r['hr_true']-r['hr_fft']) for r in all_results])
    thresholds = [5, 10, 15, 20]
    pcts = [100*np.mean(hr_err <= t) for t in thresholds]
    colors = [GREEN, TEAL, AMBER, RED]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    bars = ax.bar([f'≤{t} bpm' for t in thresholds], pcts, color=colors, alpha=0.88, edgecolor=BORDER)
    for bar, p in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{p:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color=WHITE)
    ax.set_ylim(0, 108)
    ax.set_xlabel('Error Threshold'); ax.set_ylabel('% of Files within Threshold')
    ax.set_title(f'HR Accuracy — % Within Error Band  (n={len(all_results)})',
                 color=ACCENT, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    path = FIG_DIR / f'{prefix}_within_band.png'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK); plt.close(fig)
    return str(path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    prefix = '320'
    bins   = sorted(SINGLE_DIR.glob('*.bin'))
    bins   = [p for p in bins if _RE.match(p.name)]
    print(f'Found {len(bins)} valid files\n')

    all_results = []
    errors_list = []
    t0 = time.time()

    for i, fpath in enumerate(bins):
        meta = parse_meta(str(fpath))
        try:
            res = run_fast(str(fpath))
            hr_e = round(abs(res['hr_fft'] - meta['hr_true']), 2)
            br_e = round(abs(res['br_fft'] - meta['br_true']), 2) if meta['has_br'] else None
            all_results.append(dict(
                filename=fpath.name,
                dist_m=meta['dist_m'], angle_deg=meta['angle_deg'],
                hr_true=meta['hr_true'], hr_fft=res['hr_fft'],
                hr_ac=res['hr_ac'],    hr_err=hr_e,
                br_true=meta['br_true'], br_fft=res['br_fft'],
                br_ac=res['br_ac'],    br_err=br_e,
                has_br=meta['has_br'],
                rng_m=res['rng_m'],    ang_d=res['ang_d'],
            ))
            if (i+1) % 40 == 0 or i == 0:
                el = time.time() - t0
                print(f'  [{i+1:>3}/{len(bins)}]  {el:.0f}s  '
                      f'HR={res["hr_fft"]:.1f} (err={hr_e:.1f})')
        except Exception as e:
            errors_list.append(dict(file=fpath.name, error=str(e)))
            if (i+1) % 40 == 0:
                print(f'  [{i+1:>3}/{len(bins)}]  ❌ {fpath.name}: {e}')

    print(f'\n✅ {len(all_results)}/{len(bins)} processed in {time.time()-t0:.0f}s  '
          f'({len(errors_list)} errors)')

    # Aggregate figures
    print('Generating aggregate figures...')
    figs = {}
    figs['hr_scatter']  = fig_hr_scatter(all_results, prefix)
    figs['hr_hist']     = fig_hr_histogram(all_results, prefix)
    figs['mae_dist']    = fig_mae_by_distance(all_results, prefix)
    figs['mae_angle']   = fig_mae_by_angle(all_results, prefix)
    figs['within_band'] = fig_within_band(all_results, prefix)

    br_rows = [r for r in all_results if r['has_br'] and r['br_err'] is not None]
    figs['br_scatter']  = fig_br_scatter(br_rows, prefix)
    figs['br_hist']     = fig_br_histogram(br_rows, prefix)

    # Metrics
    hr_errs = np.array([r['hr_err'] for r in all_results])
    br_errs = np.array([r['br_err'] for r in br_rows]) if br_rows else np.array([])

    def pct(arr, t): return round(100*float(np.mean(arr <= t)), 2)

    summary = dict(
        n_total      = len(all_results),
        n_errors     = len(errors_list),
        hr_mae       = round(float(np.mean(hr_errs)), 3),
        hr_rmse      = round(float(np.sqrt(np.mean(hr_errs**2))), 3),
        hr_median    = round(float(np.median(hr_errs)), 3),
        hr_max       = round(float(np.max(hr_errs)), 3),
        hr_within5   = pct(hr_errs, 5),
        hr_within10  = pct(hr_errs, 10),
        hr_within15  = pct(hr_errs, 15),
        n_with_br    = len(br_rows),
        br_mae       = round(float(np.mean(br_errs)), 3)  if len(br_errs) else None,
        br_rmse      = round(float(np.sqrt(np.mean(br_errs**2))), 3) if len(br_errs) else None,
        br_within2   = pct(br_errs, 2) if len(br_errs) else None,
        br_within4   = pct(br_errs, 4) if len(br_errs) else None,
    )

    out = dict(summary=summary, results=all_results, errors=errors_list, figures=figs)
    out_json = OUT_DIR / 'metrics_320.json'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\n📊 Summary:')
    print(f'   HR MAE  = {summary["hr_mae"]:.2f} bpm')
    print(f'   HR RMSE = {summary["hr_rmse"]:.2f} bpm')
    print(f'   ≤5 bpm  = {summary["hr_within5"]:.1f}%')
    print(f'   ≤10 bpm = {summary["hr_within10"]:.1f}%')
    if summary['br_mae']:
        print(f'   BR MAE  = {summary["br_mae"]:.2f} br/min')
    print(f'   Metrics → {out_json}')


if __name__ == '__main__':
    main()
