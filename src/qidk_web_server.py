"""
qidk_web_server.py
==================
Local web dashboard for FMCW Radar QIDK Vital-Signs Pipeline.

Run:
    python qidk_web_server.py                      # process all 320 files
    python qidk_web_server.py --file D1.5m_...bin  # single file
    python qidk_web_server.py --max 20             # first 20 files

What happens:
  1. Server starts on http://localhost:5050
  2. Browser opens automatically (no mouse needed)
  3. Pipeline runs in background thread, CBF method
  4. Results stream live to dashboard via Server-Sent Events
  5. Dashboard shows HR/BR, heatmap, displacement, spectra, error table
"""

import argparse, base64, io, json, os, queue, re, sys, threading, time, webbrowser
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
try:
    from flask import Flask, Response, render_template_string, jsonify
except ImportError:
    os.system('pip install flask -q')
    from flask import Flask, Response, render_template_string, jsonify

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dc_offset_correction  import circle_fit_correction
from clutter_suppression   import svd_clutter_removal
from phase_demodulation    import demodulate
from vital_sign_extraction import extract_vital_signs
from heterogeneous         import HeterogeneousPipeline

# ── Hardware constants ────────────────────────────────────────────────────────
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

_RE = re.compile(
    r'D(?P<dist>[\d.]+)m_(?P<angle>\d+)d_HR_(?P<hr>\d+)_pm_(?P<pm>\d+)_BR_(?P<br>[\d.]+)\.bin'
)

# ── Global state ─────────────────────────────────────────────────────────────
app             = Flask(__name__)
results_list    = []                        # accumulated result dicts
event_history   = []                        # ALL past SSE events (for replay)
client_queues   = []                        # one queue per connected SSE client
client_lock     = threading.Lock()
pipeline_done   = threading.Event()
SINGLE_DIR      = Path('./Single_Subject')


# ── Pipeline helpers ──────────────────────────────────────────────────────────

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
    return round((1/T)*60, 1) if T > 0 else None

def fft_peak(sig, fs, fmin, fmax):
    n = len(sig); nf = 4*n
    spec  = np.abs(np.fft.fft(sig*np.hanning(n), n=nf))[:nf//2]
    freqs = np.arange(nf//2)*fs/nf
    mask  = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask): return 0.0, freqs, spec
    idx = np.where(mask)[0]; pk = idx[np.argmax(spec[idx])]
    return float(freqs[pk]), freqs, spec

_tri_pipe_server = None

def get_server_pipeline():
    global _tri_pipe_server
    if _tri_pipe_server is None:
        _tri_pipe_server = HeterogeneousPipeline()
    return _tri_pipe_server

def run_pipeline(bin_path):
    t0 = time.perf_counter()
    pipe = get_server_pipeline()
    res = pipe.process_single_frame(bin_path)

    br_hz, br_freqs, br_spec = fft_peak(res['resp_signal'], FS_VITAL, 0.1, 0.5)
    hr_hz, hr_freqs, hr_spec = fft_peak(res['heart_signal'], FS_VITAL, 0.8, 2.5)
    elapsed = (time.perf_counter() - t0) * 1e3

    return dict(
        rng_m=round(float(res['range_m']), 3),
        ang_d=round(float(res['angle_deg']), 2),
        hr_fft=round(res['hr_fft'], 1),
        br_fft=round(res['br_fft'], 1),
        hr_ac=res['hr_ac'],
        br_ac=res['br_ac'],
        disp_mm=res['disp_mm'],
        resp_signal=res['resp_signal'],
        heart_signal=res['heart_signal'],
        ra_heatmap=res['ra_heatmap'],
        br_hz=br_hz,
        br_freqs=br_freqs,
        br_spec=br_spec,
        hr_hz=hr_hz,
        hr_freqs=hr_freqs,
        hr_spec=hr_spec,
        elapsed_ms=round(elapsed, 1),
    )


# ── Figure generators (return base64 PNG) ────────────────────────────────────
DARK='#0d1117'; PANEL='#161b22'; FG='#c9d1d9'; BORDER='#30363d'
ACCENT='#58a6ff'; GREEN='#3fb950'; RED='#f85149'; PURPLE='#bc8cff'
AMBER='#d29922'; TEAL='#39d353'; WHITE='#e6edf3'

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_heatmap_fig(r):
    fig, ax = plt.subplots(figsize=(7, 4)); fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    nr  = min(80, N_FFT_RANGE//2)
    dB  = 20*np.log10(r['ra_heatmap'][:, :nr]+1e-30)
    ext = [AZIMUTH_GRID[-1], AZIMUTH_GRID[0], 0, RANGE_AXIS[nr]]
    im  = ax.imshow(dB.T, aspect='auto', origin='lower', extent=ext,
                    cmap='plasma', vmin=np.percentile(dB, 5))
    plt.colorbar(im, ax=ax, label='Power [dB]', pad=0.01)
    ax.axhline(r['rng_m'], color=GREEN, lw=1.5, ls='--', label=f'{r["rng_m"]:.2f} m')
    ax.axvline(r['ang_d'], color=ACCENT, lw=1.5, ls='--', label=f'{r["ang_d"]:.1f}°')
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER)
    ax.set_title('Range–Azimuth Heatmap (Post SVD)', color=GREEN, fontsize=9, fontweight='bold')
    ax.set_xlabel('Azimuth [°]', color=FG); ax.set_ylabel('Range [m]', color=FG)
    ax.tick_params(colors='#8b949e'); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return _fig_to_b64(fig)

def make_displacement_fig(r):
    t = np.arange(M_CHIRPS)/FS_VITAL
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.patch.set_facecolor(DARK)
    for ax in axes: ax.set_facecolor(PANEL); ax.grid(True, alpha=0.2); ax.tick_params(colors='#8b949e')
    axes[0].plot(t, r['disp_mm'],     color=ACCENT,  lw=0.9, alpha=0.7, label='Total')
    axes[0].plot(t, r['resp_signal'], color=GREEN,   lw=1.4, label=f'Resp [{r["br_fft"]:.1f} br/min]')
    axes[0].plot(t, r['heart_signal'],color=PURPLE,  lw=1.0, label=f'Heart [{r["hr_fft"]:.1f} bpm]')
    axes[0].set_ylabel('Displacement [mm]', color=FG)
    axes[0].set_title('Chest Displacement', color=ACCENT, fontsize=9, fontweight='bold')
    axes[0].legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER)
    axes[1].plot(t, r['resp_signal'], color=GREEN,  lw=1.4)
    axes[1].fill_between(t, r['resp_signal'], alpha=0.2, color=GREEN)
    axes[1].set_xlabel('Time [s]', color=FG); axes[1].set_ylabel('Resp [mm]', color=FG)
    axes[1].set_title('Respiration Component', color=GREEN, fontsize=9, fontweight='bold')
    plt.tight_layout()
    return _fig_to_b64(fig)

def make_spectra_fig(r, meta):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(DARK)
    for ax in axes: ax.set_facecolor(PANEL); ax.grid(True, alpha=0.2); ax.tick_params(colors='#8b949e')

    mb = (r['br_freqs']>=0.05)&(r['br_freqs']<=0.75)
    bx = r['br_freqs'][mb]*60; by = r['br_spec'][mb]
    axes[0].plot(bx, by, color=GREEN, lw=1.8)
    axes[0].fill_between(bx, by, alpha=0.2, color=GREEN)
    axes[0].axvline(r['br_hz']*60, color=AMBER, lw=2.5, ls='--', label=f'Est {r["br_fft"]:.1f} br/min')
    if meta.get('has_br'):
        axes[0].axvline(meta['br_true'], color=WHITE, lw=2, ls=':', label=f'True {meta["br_true"]} br/min')
    axes[0].set_title('Respiration Spectrum', color=GREEN, fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Frequency [br/min]', color=FG); axes[0].set_ylabel('Magnitude', color=FG)
    axes[0].legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER); axes[0].set_xlim(3, 45)

    mh = (r['hr_freqs']>=0.6)&(r['hr_freqs']<=3.2)
    hx = r['hr_freqs'][mh]*60; hy = r['hr_spec'][mh]
    hr_err = abs(r['hr_fft'] - meta.get('hr_true', r['hr_fft']))
    hr_col = GREEN if hr_err<=5 else (AMBER if hr_err<=10 else RED)
    axes[1].plot(hx, hy, color=PURPLE, lw=1.8)
    axes[1].fill_between(hx, hy, alpha=0.2, color=PURPLE)
    axes[1].axvline(r['hr_hz']*60, color=hr_col, lw=2.5, ls='--', label=f'Est {r["hr_fft"]:.1f} bpm')
    if meta.get('hr_true'):
        axes[1].axvline(meta['hr_true'], color=WHITE, lw=2, ls=':', label=f'True {meta["hr_true"]} bpm')
    axes[1].set_title('Heart Rate Spectrum', color=PURPLE, fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Frequency [bpm]', color=FG); axes[1].set_ylabel('Magnitude', color=FG)
    axes[1].legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER); axes[1].set_xlim(36, 192)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ── HTML Dashboard ────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FMCW Radar QIDK Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#c9d1d9;
    --accent:#58a6ff;--green:#3fb950;--amber:#d29922;--red:#f85149;
    --purple:#bc8cff;--teal:#39d353;--white:#e6edf3;
  }
  body{background:var(--bg);color:var(--fg);font-family:'Inter',sans-serif;min-height:100vh}
  header{
    background:linear-gradient(135deg,#1a1f2e 0%,#0d1117 100%);
    border-bottom:1px solid var(--border);padding:18px 28px;
    display:flex;align-items:center;justify-content:space-between;
    position:sticky;top:0;z-index:100;backdrop-filter:blur(12px);
  }
  header h1{font-size:1.25rem;font-weight:700;color:var(--white);letter-spacing:.5px}
  header h1 span{color:var(--accent)}
  .status-pill{
    padding:5px 14px;border-radius:20px;font-size:.78rem;font-weight:600;
    font-family:'JetBrains Mono',monospace;letter-spacing:.3px;
    display:flex;align-items:center;gap:7px;
  }
  .status-pill.running{background:rgba(63,185,80,.15);border:1px solid var(--green);color:var(--green)}
  .status-pill.idle   {background:rgba(88,166,255,.12);border:1px solid var(--accent);color:var(--accent)}
  .status-pill.done   {background:rgba(188,140,255,.12);border:1px solid var(--purple);color:var(--purple)}
  .pulse{width:8px;height:8px;border-radius:50%;background:currentColor;
         animation:pulse 1.2s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
  .main{padding:24px 28px;display:grid;gap:20px}
  /* Vitals bar */
  .vitals-bar{
    display:grid;grid-template-columns:repeat(4,1fr);gap:16px;
  }
  .vital-card{
    background:var(--panel);border:1px solid var(--border);border-radius:12px;
    padding:20px 22px;position:relative;overflow:hidden;transition:.3s;
  }
  .vital-card::before{
    content:'';position:absolute;inset:0;opacity:.06;
    background:radial-gradient(circle at 80% 20%,var(--glow),transparent 70%);
  }
  .vital-card.hr{--glow:var(--purple)}
  .vital-card.br{--glow:var(--green)}
  .vital-card.loc{--glow:var(--accent)}
  .vital-card.err{--glow:var(--amber)}
  .vital-label{font-size:.72rem;font-weight:600;letter-spacing:1.2px;
               text-transform:uppercase;color:#8b949e;margin-bottom:8px}
  .vital-value{font-size:2.4rem;font-weight:700;line-height:1;
               font-family:'JetBrains Mono',monospace;color:var(--white)}
  .vital-value small{font-size:.85rem;font-weight:400;color:#8b949e;margin-left:4px}
  .vital-sub{font-size:.78rem;margin-top:6px;color:#8b949e}
  .vital-sub b{color:var(--fg)}
  .badge{
    display:inline-block;padding:2px 8px;border-radius:6px;font-size:.7rem;
    font-weight:600;margin-top:8px;
  }
  .badge.good{background:rgba(63,185,80,.2);color:var(--green);border:1px solid rgba(63,185,80,.3)}
  .badge.warn{background:rgba(210,153,34,.2);color:var(--amber);border:1px solid rgba(210,153,34,.3)}
  .badge.bad {background:rgba(248,81,73,.2); color:var(--red); border:1px solid rgba(248,81,73,.3)}
  /* Progress */
  .progress-section{background:var(--panel);border:1px solid var(--border);
                    border-radius:12px;padding:18px 22px}
  .progress-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
  .progress-header span{font-size:.82rem;color:#8b949e}
  .progress-header strong{color:var(--white);font-family:'JetBrains Mono',monospace}
  .progress-bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden}
  .progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--purple));
                 border-radius:3px;transition:width .4s ease;width:0%}
  /* QIDK terminal */
  .qidk-section{background:var(--panel);border:1px solid var(--border);
                border-radius:12px;padding:20px 22px}
  .qidk-section h2{font-size:.85rem;font-weight:600;color:var(--accent);
                   letter-spacing:.8px;text-transform:uppercase;margin-bottom:14px}
  .qidk-terminal{
    background:#010409;border:1px solid var(--border);border-radius:8px;
    padding:16px 18px;font-family:'JetBrains Mono',monospace;font-size:.8rem;
    line-height:1.7;color:#c9d1d9;min-height:120px;max-height:260px;
    overflow-y:auto;white-space:pre-wrap;word-break:break-all;
  }
  /* Figures */
  .figures-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .figure-card{background:var(--panel);border:1px solid var(--border);
               border-radius:12px;padding:16px 18px}
  .figure-card h2{font-size:.82rem;font-weight:600;color:var(--fg);
                  text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px}
  .figure-card img{width:100%;border-radius:6px;display:block}
  .figure-card .placeholder{
    height:200px;display:flex;align-items:center;justify-content:center;
    color:#30363d;font-size:.85rem;border:1px dashed var(--border);border-radius:6px;
  }
  /* Spectra full-width */
  .spectra-card{background:var(--panel);border:1px solid var(--border);
                border-radius:12px;padding:16px 18px}
  .spectra-card h2{font-size:.82rem;font-weight:600;color:var(--fg);
                   text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px}
  .spectra-card img{width:100%;border-radius:6px}
  /* Results table */
  .table-section{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:18px 22px}
  .table-section h2{font-size:.85rem;font-weight:600;color:var(--fg);
                    text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px}
  .results-wrap{max-height:320px;overflow-y:auto;border-radius:6px}
  table{width:100%;border-collapse:collapse;font-size:.8rem}
  th{background:#010409;color:#8b949e;font-weight:600;padding:9px 12px;
     text-align:left;position:sticky;top:0;letter-spacing:.5px;font-size:.72rem;
     border-bottom:1px solid var(--border)}
  td{padding:8px 12px;border-bottom:1px solid #21262d;font-family:'JetBrains Mono',monospace}
  tr:last-child td{border-bottom:none}
  tr.new-row{animation:fadeRow .4s ease}
  @keyframes fadeRow{from{background:rgba(88,166,255,.12)}to{background:transparent}}
  .err-good{color:var(--green)} .err-warn{color:var(--amber)} .err-bad{color:var(--red)}
  /* File label */
  .file-label{
    background:#010409;border:1px solid var(--border);border-radius:8px;
    padding:8px 14px;font-family:'JetBrains Mono',monospace;font-size:.78rem;
    color:var(--accent);margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;
  }
  /* Scrollbar */
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:var(--panel)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<header>
  <h1>📡 FMCW Radar · <span>QIDK</span> Vital Signs Dashboard</h1>
  <div id="status-pill" class="status-pill idle">
    <div class="pulse"></div>
    <span id="status-text">Connecting...</span>
  </div>
</header>

<div class="main">

  <!-- File label -->
  <div class="file-label" id="file-label">Waiting for pipeline...</div>

  <!-- Vitals bar -->
  <div class="vitals-bar">
    <div class="vital-card hr">
      <div class="vital-label">Heart Rate (FFT)</div>
      <div class="vital-value" id="hr-val">—<small>bpm</small></div>
      <div class="vital-sub">AC: <b id="hr-ac">—</b></div>
      <div id="hr-badge"></div>
    </div>
    <div class="vital-card br">
      <div class="vital-label">Breathing Rate (FFT)</div>
      <div class="vital-value" id="br-val">—<small>br/min</small></div>
      <div class="vital-sub">AC: <b id="br-ac">—</b></div>
      <div id="br-badge"></div>
    </div>
    <div class="vital-card loc">
      <div class="vital-label">Target Location</div>
      <div class="vital-value" id="loc-val" style="font-size:1.6rem">—</div>
      <div class="vital-sub" id="loc-sub">Range · Azimuth</div>
    </div>
    <div class="vital-card err">
      <div class="vital-label">Timing</div>
      <div class="vital-value" id="timing-val" style="font-size:1.6rem">—<small>s</small></div>
      <div class="vital-sub" id="progress-sub">0 / 0 processed</div>
    </div>
  </div>

  <!-- Progress bar -->
  <div class="progress-section">
    <div class="progress-header">
      <span id="prog-label">Initialising pipeline...</span>
      <strong id="prog-pct">0%</strong>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
  </div>

  <!-- QIDK terminal -->
  <div class="qidk-section">
    <h2>📟 QIDK Terminal Output</h2>
    <div class="qidk-terminal" id="qidk-term">Waiting for results...<br/></div>
  </div>

  <!-- Heatmap + Displacement -->
  <div class="figures-grid">
    <div class="figure-card">
      <h2>🌡️ Range–Azimuth Heatmap</h2>
      <div id="heatmap-wrap"><div class="placeholder">Processing...</div></div>
    </div>
    <div class="figure-card">
      <h2>📈 Chest Displacement</h2>
      <div id="disp-wrap"><div class="placeholder">Processing...</div></div>
    </div>
  </div>

  <!-- Spectra -->
  <div class="spectra-card">
    <h2>📊 Frequency Spectra — Heart Rate &amp; Breathing Rate</h2>
    <div id="spectra-wrap"><div class="placeholder" style="height:180px">Processing...</div></div>
  </div>

  <!-- Results table -->
  <div class="table-section">
    <h2>📋 Results Log</h2>
    <div class="results-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>File</th><th>Dist</th><th>Angle</th>
            <th>True HR</th><th>Est HR</th><th>HR Err</th>
            <th>True BR</th><th>Est BR</th><th>BR Err</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody id="results-body"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
let rowCount = 0, total = 0, done = 0;

function errClass(e, t1, t2){
  if(e===null||e===undefined) return '';
  return e<=t1?'err-good':e<=t2?'err-warn':'err-bad';
}
function badge(e, t1, t2, unit){
  if(e===null||e===undefined) return '';
  const cls = e<=t1?'good':e<=t2?'warn':'bad';
  return `<span class="badge ${cls}">${e<=t1?'✓ ':''}±${e.toFixed(1)} ${unit}</span>`;
}
function escHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;') }

const term = document.getElementById('qidk-term');
function appendTerm(txt){ term.innerHTML += txt; term.scrollTop=term.scrollHeight; }

const evtSrc = new EventSource('/stream');

evtSrc.addEventListener('status', e=>{
  const d = JSON.parse(e.data);
  const pill = document.getElementById('status-pill');
  const st   = document.getElementById('status-text');
  pill.className = 'status-pill ' + d.state;
  st.textContent = d.message;
  total = d.total || total;
  document.getElementById('progress-sub').textContent = `${d.done||0} / ${total} processed`;
});

evtSrc.addEventListener('progress', e=>{
  const d = JSON.parse(e.data);
  done = d.done; total = d.total;
  const pct = total>0 ? Math.round(done/total*100) : 0;
  document.getElementById('prog-fill').style.width = pct+'%';
  document.getElementById('prog-pct').textContent  = pct+'%';
  document.getElementById('prog-label').textContent =
    `Processing file ${done} of ${total}: ${d.filename}`;
  document.getElementById('file-label').textContent = '📁 ' + d.filepath;
  document.getElementById('progress-sub').textContent = `${done} / ${total} processed`;
});

evtSrc.addEventListener('result', e=>{
  const d = JSON.parse(e.data);
  const r = d.result, m = d.meta;

  // Vitals
  document.getElementById('hr-val').innerHTML = `${r.hr_fft.toFixed(1)}<small>bpm</small>`;
  document.getElementById('br-val').innerHTML = `${r.br_fft.toFixed(1)}<small>br/min</small>`;
  document.getElementById('hr-ac').textContent = r.hr_ac ? r.hr_ac.toFixed(1)+' bpm' : 'N/A';
  document.getElementById('br-ac').textContent = r.br_ac ? r.br_ac.toFixed(1)+' br/min' : 'N/A';
  document.getElementById('loc-val').textContent = r.rng_m.toFixed(2)+'m';
  document.getElementById('loc-sub').textContent = `@ ${r.ang_d.toFixed(1)}°`;
  document.getElementById('timing-val').innerHTML = `${(r.elapsed_ms/1000).toFixed(1)}<small>s</small>`;

  if(m && m.hr_true){
    const he = Math.abs(r.hr_fft - m.hr_true);
    document.getElementById('hr-badge').innerHTML = badge(he,5,10,'bpm');
  }
  if(m && m.has_br){
    const be = Math.abs(r.br_fft - m.br_true);
    document.getElementById('br-badge').innerHTML = badge(be,2,4,'br/min');
  }

  // Figures
  if(d.heatmap_b64)
    document.getElementById('heatmap-wrap').innerHTML=`<img src="data:image/png;base64,${d.heatmap_b64}" alt="heatmap"/>`;
  if(d.disp_b64)
    document.getElementById('disp-wrap').innerHTML=`<img src="data:image/png;base64,${d.disp_b64}" alt="displacement"/>`;
  if(d.spectra_b64)
    document.getElementById('spectra-wrap').innerHTML=`<img src="data:image/png;base64,${d.spectra_b64}" alt="spectra"/>`;

  // Terminal
  const he = m&&m.hr_true ? Math.abs(r.hr_fft-m.hr_true) : null;
  const be = m&&m.has_br  ? Math.abs(r.br_fft-m.br_true) : null;
  const hc = he===null?'#58a6ff':he<=5?'#3fb950':he<=10?'#d29922':'#f85149';
  const bc = be===null?'#58a6ff':be<=2?'#3fb950':be<=4?'#d29922':'#f85149';
  appendTerm(
    `<span style="color:#58a6ff">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>\n`+
    `<span style="color:#d29922">📁 ${escHtml(d.filename)}</span>\n`+
    `<span style="color:#8b949e">   Loc  : ${r.rng_m.toFixed(2)} m  |  ${r.ang_d.toFixed(1)}°</span>\n`+
    `<span style="color:${hc}">   HEART: FFT=${r.hr_fft.toFixed(1)} bpm`+
    (m&&m.hr_true ? `  true=${m.hr_true}  err=${he.toFixed(1)} bpm`:'')+'</span>\n'+
    `<span style="color:${bc}">   RESP : FFT=${r.br_fft.toFixed(1)} br/min`+
    (m&&m.has_br  ? `  true=${m.br_true}  err=${be.toFixed(1)} br/min`:'')+'</span>\n'+
    `<span style="color:#8b949e">   Time : ${(r.elapsed_ms/1000).toFixed(1)}s</span>\n`
  );

  // Table row
  rowCount++;
  const he_s = he!==null ? `<span class="${errClass(he,5,10)}">${he.toFixed(1)}</span>` : '—';
  const be_s = be!==null ? `<span class="${errClass(be,2,4)}">${be.toFixed(1)}</span>` : '—';
  const row = `<tr class="new-row">
    <td>${rowCount}</td>
    <td style="color:var(--accent)">${escHtml(d.filename.replace('.bin','').substring(0,30))}</td>
    <td>${m?m.dist_m:'—'}</td><td>${m?m.angle_deg+'°':'—'}</td>
    <td>${m&&m.hr_true?m.hr_true:'—'}</td>
    <td style="color:var(--white);font-weight:600">${r.hr_fft.toFixed(1)}</td>
    <td>${he_s}</td>
    <td>${m&&m.has_br?m.br_true:'—'}</td>
    <td style="color:var(--white)">${r.br_fft.toFixed(1)}</td>
    <td>${be_s}</td>
    <td style="color:#8b949e">${(r.elapsed_ms/1000).toFixed(1)}s</td>
  </tr>`;
  const tbody = document.getElementById('results-body');
  tbody.insertAdjacentHTML('afterbegin', row);
});

evtSrc.addEventListener('done', e=>{
  const d = JSON.parse(e.data);
  const pill = document.getElementById('status-pill');
  pill.className = 'status-pill done';
  document.getElementById('status-text').textContent = `Done — ${d.n_processed} files processed`;
  document.getElementById('prog-fill').style.width = '100%';
  document.getElementById('prog-pct').textContent  = '100%';
  document.getElementById('prog-label').textContent = `✅ Complete — ${d.n_processed} files`;
  appendTerm(`\n<span style="color:#3fb950">✅ PIPELINE COMPLETE — ${d.n_processed} files processed</span>\n`+
             `<span style="color:#3fb950">   HR MAE = ${d.hr_mae.toFixed(2)} bpm</span>\n`);
});

evtSrc.onerror = ()=>{
  document.getElementById('status-text').textContent = 'Connection error';
};
</script>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/stream')
def stream():
    """SSE endpoint — new clients immediately receive ALL past events (replay),
    then keep listening for future ones. This ensures QIDK Chrome always gets
    the full data even if it connects after the pipeline has already started."""
    q = queue.Queue()
    with client_lock:
        client_queues.append(q)
        # Snapshot of everything that happened before this client connected
        history_snapshot = list(event_history)

    def generate():
        # ── Replay past events immediately ────────────────────────────────────
        for evt in history_snapshot:
            yield f"event: {evt['type']}\ndata: {json.dumps(evt['data'])}\n\n"

        # ── Stream future events ───────────────────────────────────────────────
        try:
            while True:
                try:
                    evt = q.get(timeout=25)
                    yield f"event: {evt['type']}\ndata: {json.dumps(evt['data'])}\n\n"
                    if evt['type'] == 'done':
                        break
                except queue.Empty:
                    yield ": keepalive\n\n"   # keep connection alive
        finally:
            with client_lock:
                try: client_queues.remove(q)
                except ValueError: pass

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/state')
def api_state():
    """REST fallback: returns current pipeline state + all results (no images)."""
    safe = []
    for r in results_list:
        entry = {k: v for k, v in r.items() if k not in ('heatmap_b64','disp_b64','spectra_b64')}
        safe.append(entry)
    return jsonify({'done': pipeline_done.is_set(), 'n': len(safe), 'results': safe})

@app.route('/results')
def results_route():
    return jsonify(results_list)


# ── Pipeline worker thread ───────────────────────────────────────────────────

def push(evt_type, data):
    """Save event to history AND fan-out to every connected SSE client.
    Late-connecting clients (e.g. QIDK Chrome) will get the full history
    replayed automatically when they connect via /stream.
    """
    evt = {'type': evt_type, 'data': data}
    with client_lock:
        event_history.append(evt)          # ← persisted for replay
        for q in client_queues:
            q.put(evt)                     # ← delivered to live clients


def pipeline_worker(bin_files):
    total = len(bin_files)
    push('status', {'state': 'running', 'message': f'Running CBF on {total} files', 'total': total, 'done': 0})
    hr_errors = []

    for idx, fpath in enumerate(bin_files):
        fname = Path(fpath).name
        push('progress', {'done': idx+1, 'total': total, 'filename': fname, 'filepath': str(fpath)})

        meta = parse_meta(str(fpath))
        try:
            r = run_pipeline(str(fpath))

            # Generate figures
            hm_b64  = make_heatmap_fig(r)
            dis_b64 = make_displacement_fig(r)
            sp_b64  = make_spectra_fig(r, meta)

            hr_e = round(abs(r['hr_fft'] - meta['hr_true']), 2) if meta.get('hr_true') else None
            br_e = round(abs(r['br_fft'] - meta['br_true']), 2) if meta.get('has_br') else None
            if hr_e is not None: hr_errors.append(hr_e)

            entry = dict(filename=fname, meta=meta,
                         result={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in r.items()
                                 if not isinstance(v, np.ndarray) or k not in
                                 ('ra_heatmap','disp_mm','resp_signal','heart_signal',
                                  'br_freqs','br_spec','hr_freqs','hr_spec')},
                         hr_err=hr_e, br_err=br_e)
            results_list.append(entry)

            push('result', dict(
                filename=fname, meta=meta,
                result={k: v for k, v in r.items()
                        if not isinstance(v, np.ndarray)},
                heatmap_b64=hm_b64, disp_b64=dis_b64, spectra_b64=sp_b64,
            ))

        except Exception as exc:
            push('result', dict(
                filename=fname, meta=meta,
                result={'hr_fft': 0, 'br_fft': 0, 'hr_ac': None, 'br_ac': None,
                        'rng_m': 0, 'ang_d': 0, 'elapsed_ms': 0},
                heatmap_b64=None, disp_b64=None, spectra_b64=None,
                error=str(exc),
            ))

    hr_mae = round(float(np.mean(hr_errors)), 2) if hr_errors else 0.0
    push('done',   {'n_processed': len(bin_files), 'hr_mae': hr_mae})
    push('status', {'state': 'done', 'message': f'Done — {len(bin_files)} files', 'total': total, 'done': total})
    pipeline_done.set()


# ── QIDK Display Launcher ────────────────────────────────────────────────────

def _get_local_ip():
    """Best-effort: return the machine's LAN IP so QIDK can reach the server."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def open_on_qidk_display(url: str, display: str = ':0',
                          qidk_ip: str = None, qidk_user: str = 'root',
                          ssh_key: str = None):
    """
    Open the dashboard on the QIDK's own physical display — not the PC.

    Strategy (tried in order):
      1. DISPLAY=:0  — QIDK display server already running locally
                       (works when this script runs ON the QIDK itself)
      2. SSH + DISPLAY=:0  — server runs on PC, QIDK connected via SSH,
                             browser launched remotely on QIDK screen
      3. ADB shell  — QIDK connected over USB/ADB (Android/embedded Linux)
      4. Chromium kiosk via weston  — Wayland-based QIDK display

    Args:
        url      : URL to open (e.g. 'http://192.168.1.x:5050')
        display  : X11 display string on QIDK (default ':0')
        qidk_ip  : QIDK IP for SSH mode (None = local mode)
        qidk_user: SSH username for QIDK (default 'root')
        ssh_key  : Path to SSH private key (optional)
    """
    import subprocess, shutil

    browsers = ['chromium-browser', 'chromium', 'google-chrome', 'weston-browser', 'minibrowser']
    kiosk_flags = [
        '--kiosk', '--no-sandbox', '--disable-infobars',
        '--noerrdialogs', '--disable-session-crashed-bubble',
        '--disable-restore-session-state', '--autoplay-policy=no-user-gesture-required',
        '--disable-features=TranslateUI',
    ]

    # ── Mode 1: Running ON the QIDK (local display) ──────────────────────────
    if qidk_ip is None:
        env = {**os.environ, 'DISPLAY': display}
        for browser in browsers:
            if shutil.which(browser):
                cmd = [browser] + kiosk_flags + [url]
                print(f'  [QIDK Display] Launching {browser} on DISPLAY={display} ...')
                try:
                    subprocess.Popen(cmd, env=env,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    print(f'  [QIDK Display] ✅ Browser opened on QIDK screen ({display})')
                    return
                except Exception as e:
                    print(f'  [QIDK Display] {browser} failed: {e}')

        # Wayland fallback via weston-terminal
        if shutil.which('weston-terminal'):
            cmd = ['weston-terminal', '--shell', f'chromium {" ".join(kiosk_flags)} {url}']
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('  [QIDK Display] ✅ Launched via weston-terminal')
            return

        print('  [QIDK Display] ⚠️  No browser found on QIDK local display.')
        print(f'  [QIDK Display]    Manually run: DISPLAY={display} chromium --kiosk {url}')
        return

    # ── Mode 2: Remote QIDK via SSH ──────────────────────────────────────────
    if shutil.which('ssh'):
        for browser in browsers:
            ssh_key_arg = ['-i', ssh_key] if ssh_key else []
            check_cmd = ['ssh', '-o', 'ConnectTimeout=4', '-o', 'StrictHostKeyChecking=no',
                         *ssh_key_arg, f'{qidk_user}@{qidk_ip}',
                         f'which {browser} 2>/dev/null']
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                browser_path = result.stdout.strip()
                launch_cmd = (
                    f'DISPLAY={display} {browser_path} '
                    + ' '.join(kiosk_flags)
                    + f' {url} &'
                )
                run_cmd = ['ssh', '-o', 'ConnectTimeout=4', '-o', 'StrictHostKeyChecking=no',
                           *ssh_key_arg, f'{qidk_user}@{qidk_ip}', launch_cmd]
                print(f'  [QIDK SSH] Launching {browser} on {qidk_ip} DISPLAY={display} ...')
                subprocess.Popen(run_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f'  [QIDK SSH] ✅ Browser opened on QIDK display via SSH')
                return

        print(f'  [QIDK SSH] ⚠️  Could not find a browser on QIDK ({qidk_ip}).')
        print(f'  [QIDK SSH]    Manually: ssh {qidk_user}@{qidk_ip}')
        print(f'                          DISPLAY={display} chromium --kiosk {url}')
        return

    # ── Mode 3: ADB (USB-connected Android/Linux QIDK) ───────────────────────
    if shutil.which('adb'):
        print('  [QIDK ADB] Trying to open URL via ADB ...')
        adb_cmd = ['adb', 'shell', 'am', 'start', '-a', 'android.intent.action.VIEW',
                   '-d', url, '--activity-clear-top']
        result = subprocess.run(adb_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f'  [QIDK ADB] ✅ URL opened on QIDK display via ADB')
        else:
            print(f'  [QIDK ADB] ⚠️  ADB failed: {result.stderr.strip()}')
        return

    print('  [QIDK Display] ⚠️  No SSH/ADB available. Use --no-qidk-display for PC browser.')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FMCW QIDK Web Dashboard — opens on QIDK display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ON the QIDK itself (opens browser on QIDK screen automatically):
  python qidk_web_server.py --max 5

  # Run on PC, open dashboard on QIDK connected via SSH:
  python qidk_web_server.py --qidk-ip 192.168.1.42 --max 5

  # Run on PC, open on QIDK via ADB (USB):
  python qidk_web_server.py --adb --max 5

  # Debug on PC only (no QIDK, use PC browser):
  python qidk_web_server.py --no-qidk-display --max 5
"""
    )
    parser.add_argument('--file',             help='Single .bin file to process')
    parser.add_argument('--folder',           default='./Single_Subject')
    parser.add_argument('--max',              type=int,   default=None)
    parser.add_argument('--port',             type=int,   default=8080)
    parser.add_argument('--display',          default=':0',
                        help='X11 DISPLAY on QIDK (default :0)')
    parser.add_argument('--qidk-ip',          default=None,
                        help='QIDK IP address for SSH mode (omit if running ON the QIDK)')
    parser.add_argument('--qidk-user',        default='root',
                        help='QIDK SSH username (default: root)')
    parser.add_argument('--ssh-key',          default=None,
                        help='Path to SSH private key for QIDK')
    parser.add_argument('--no-qidk-display',  action='store_true',
                        help='Fall back to opening browser on THIS PC (debug mode)')
    args = parser.parse_args()

    # ── Collect files ─────────────────────────────────────────────────────────
    if args.file:
        bin_files = [Path(args.file)]
    else:
        folder = Path(args.folder)
        bin_files = sorted(p for p in folder.glob('*.bin') if _RE.match(p.name))
        if args.max:
            bin_files = bin_files[:args.max]

    if not bin_files:
        print(f'❌  No valid .bin files found in {args.folder}')
        sys.exit(1)

    local_ip = _get_local_ip()
    url      = f'http://{local_ip}:{args.port}'

    print(f'\n{"="*62}')
    print(f'  FMCW Radar — QIDK Web Dashboard')
    print(f'  Files     : {len(bin_files)}')
    print(f'  Server IP : {local_ip}:{args.port}')
    print(f'  Dashboard : {url}')
    if args.qidk_ip:
        print(f'  QIDK IP   : {args.qidk_ip}  (SSH mode)')
    elif args.no_qidk_display:
        print(f'  Display   : PC browser (debug mode)')
    else:
        print(f'  Display   : QIDK screen  DISPLAY={args.display}')
    print(f'{"="*62}\n')

    # ── Start pipeline worker ─────────────────────────────────────────────────
    t = threading.Thread(target=pipeline_worker, args=(bin_files,), daemon=True)
    t.start()

    # ── Open browser on BOTH PC and QIDK simultaneously ───────────────────────
    def launch_display():
        import subprocess, shutil
        time.sleep(2.0)   # wait for Flask to be ready

        # ── 1. Always open on PC browser ──────────────────────────────────────
        print(f'  [PC Browser]  Opening {url} ...')
        webbrowser.open(url)

        # ── 2. Also open on QIDK if not explicitly disabled ───────────────────
        if not args.no_qidk_display:
            if args.qidk_ip:
                # SSH mode
                open_on_qidk_display(
                    url=url, display=args.display,
                    qidk_ip=args.qidk_ip, qidk_user=args.qidk_user,
                    ssh_key=args.ssh_key,
                )
            elif shutil.which('adb'):
                # ADB mode — set reverse tunnel so QIDK localhost:5050 → PC:5050
                try:
                    r = subprocess.run(
                        ['adb', 'reverse', f'tcp:{args.port}', f'tcp:{args.port}'],
                        capture_output=True, text=True, timeout=5
                    )
                    if r.returncode == 0:
                        print(f'  [QIDK ADB]    Reverse tunnel localhost:{args.port} → PC:{args.port} ✅')
                    else:
                        print(f'  [QIDK ADB]    Reverse tunnel warning: {r.stderr.strip()}')

                    # Open Chrome on QIDK screen
                    qidk_url = f'http://localhost:{args.port}'
                    intent = subprocess.run([
                        'adb', 'shell', 'am', 'start',
                        '-n', 'com.android.chrome/org.chromium.chrome.browser.ChromeTabbedActivity',
                        '-a', 'android.intent.action.VIEW',
                        '-d', qidk_url,
                        '--activity-clear-task',
                    ], capture_output=True, text=True, timeout=8)

                    if intent.returncode == 0:
                        print(f'  [QIDK ADB]    Chrome opened on QIDK screen → {qidk_url} ✅')
                        # Full-screen immersive mode
                        subprocess.run([
                            'adb', 'shell', 'settings', 'put', 'global',
                            'policy_control', 'immersive.full=com.android.chrome'
                        ], timeout=5)
                    else:
                        print(f'  [QIDK ADB]    Chrome launch failed: {intent.stderr.strip()}')
                except Exception as ex:
                    print(f'  [QIDK ADB]    Error: {ex}')
            else:
                # Fallback: local DISPLAY=:0
                open_on_qidk_display(url=url, display=args.display)

    threading.Thread(target=launch_display, daemon=True).start()

    # ── Start Flask ───────────────────────────────────────────────────────────
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

