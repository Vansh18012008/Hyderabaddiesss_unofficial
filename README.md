# FMCW Radar Vital Sign Estimation
## IC~WiBES Research Group

A research-grade, end-to-end pipeline for **contact-free heart rate (HR) and breathing rate (BR)**
estimation using a 77 GHz mmWave FMCW radar (TI AWR1843 + DCA1000 capture board).

> The pipeline is built from the ground up by integrating the seniors' beamforming framework with
> peer-reviewed preprocessing methods from references [12] and [14].

---

## Processing Pipeline

```
Raw .bin  (DCA1000 capture)
    │
    ▼  readDCA1000 / processIFSignal
(num_rx=4, num_chirps=512, num_samples=512)

    │
    ▼  performBeamforming  [CBF — seniors' design]
(num_angles=73, num_chirps, num_samples)        ← +90° to -90° in 2.5° steps
    │
    ▼ ══════════════ STAGE 1a ══════════════════
    │  DC Offset Correction — Mean Subtraction
    │  subtract temporal mean per sample [14]
    │
    ▼ ══════════════ STAGE 2  ══════════════════
    │  Static Clutter Removal — SVD rank-1
    │  remove dominant zero-Doppler component [12]
    │
    ▼  Range FFT (Hanning window, N=512)
(num_angles, num_chirps, 256 range bins)
    │
    ▼ ══════════════ STAGE 1b ══════════════════
    │  DC Correction at target range bin
    │  Taubin circle-fit on IQ constellation [12]
    │
    ▼  Phase Demodulation  [arctan + unwrap + DC removal]
    │
    ▼  Bandpass Filters
    │   Respiration : 0.1 – 0.5 Hz   (6 – 30 breaths/min)
    │   Heartbeat   : 0.8 – 2.0 Hz   (48 – 120 bpm)
    │
    ▼  FFT peak + AutoCorr cross-check
 BR (bpm)  +  HR (bpm)
```

---

## Project Structure

```
fmcw_radar_preprocessing/
├── Hr_Br_Stage1_Stage2.ipynb  ← Main notebook (Stage 1 + Stage 2 integrated)
├── README.md
├── requirements.txt
│
├── dc_offset_correction.py    ← Stage 1: 4 DC correction methods [12,13,14,15]
├── clutter_suppression.py     ← Stage 2: 4 static clutter methods [12,14,15]
├── range_fft.py               ← Range FFT + windowing utilities
├── phase_demodulation.py      ← arctangent phase extraction + unwrapping
├── vital_sign_extraction.py   ← bandpass filtering + FFT/autocorr rate estimation
├── data_generator.py          ← synthetic FMCW data cube (for testing)
├── evaluation.py              ← SCNR, SNR, MAE metrics
├── visualization.py           ← diagnostic plots
└── main.py                    ← standalone end-to-end script
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy scipy matplotlib Flask` (plus `jupyter` for notebooks).

---

## QIDK Pipeline and Live Dashboard

The QIDK runner starts the web dashboard and processes captures in one command.
Run it on the QIDK to keep the SQLite database and generated graph images on the
device. A PC on the same network can open the printed dashboard URL.

```bash
python3 -m pip install -r requirements.txt
python3 src/qidk_web_server.py --folder ./Single_Subject
```

The dashboard opens locally when a browser is available, listens on all network
interfaces at port `8080`, and continues to run even when no matching captures
are present. Use `--file path/to/capture.bin`, `--max 5`, `--port 8081`,
`--db /path/on/qidk/results.sqlite3`, or `--no-browser` as needed. Keep the
process running and use the same LAN URL from both the QIDK and PC. Network
firewall rules must allow the selected port.

Results, per-stage latency measurements, and dashboard PNG plots are written to
`data/qidk_results.sqlite3` on the machine running the server by default. The
database also retains compressed numeric heatmap, signal, and spectrum arrays;
each result exposes a downloadable `.npz` at `/data/<result-id>`. Restarting
the runner preserves prior runs; `/results` returns stored results.

The runner overlaps CPU file ingestion, the FFT/beamforming/clutter stage,
target detection, and CPU vital-sign extraction using bounded queues to limit
memory use. CUDA processing is enabled only when PyTorch with CUDA is actually
available. Qualcomm QNN/HTP detection requires a compatible ONNX model set via
`QIDK_NPU_MODEL=/path/to/model.onnx`; otherwise the dashboard reports the CPU
detection fallback. A QIDK GPU backend is not claimed when the installed
runtime does not expose a compatible compute provider.

---

## Quick Start

### Option 1 — Jupyter Notebook (recommended)

```bash
jupyter notebook Hr_Br_Stage1_Stage2.ipynb
```

Set `USE_SYNTHETIC = True` at the top of the notebook to run a complete
synthetic demo (no `.bin` file needed).

### Option 2 — Standalone Script

```bash
# Full demo with synthetic data
python main.py --mode=demo

# Full evaluation comparing all methods
python main.py --mode=evaluate --plot

# Custom parameters
python main.py --mode=demo --fc 77e9 --bandwidth 4e9 --n_chirps 128 --n_samples 256
```

---

## Radar Hardware Parameters (AWR1843 Configuration)

| Parameter | Value |
|-----------|-------|
| Start Frequency | 77 GHz |
| Stop Frequency | 79.982 GHz |
| Bandwidth | 2.982 GHz |
| Chirp Duration Tm | 100 µs |
| ADC Samples | 512 |
| Number of Chirps | 512 |
| ADC Sampling Rate | 600 kHz |
| RX Antennas | 4 |
| Range Resolution | ~5 cm |
| Max Range R_max | ~30 m |

---

## Stage 1 — DC Offset Correction

DC offset in FMCW homodyne receivers arises from LO self-mixing and IQ imbalance.
Additionally, **[14]** shows that strong static reflectors (walls) translate into
DC offsets in the synthesised slow-time IQ signal.

| Method | Reference | Description |
|--------|-----------|-------------|
| `mean_subtraction` | [14] | Subtract temporal mean per range/sample bin |
| `ema_subtraction` | general | Exponential moving average removal |
| `circle_fit_correction` | **[12]** | Taubin algebraic circle fit on IQ constellation ← **recommended** |

### Applied in the notebook:
- **Stage 1a** (before Range FFT): `mean_subtraction` on the raw beamformed chirp matrix  
- **Stage 1b** (after Range FFT): `circle_fit_correction` on the slow-time IQ at the target range bin

---

## Stage 2 — Static Clutter Removal

Static reflectors produce constant signals across all chirps (zero-Doppler), masking the
weak chest motion signal (~mm level).

| Method | Reference | Description |
|--------|-----------|-------------|
| `mean_clutter_removal` | [14] | Subtract mean range profile across chirps |
| `svd_clutter_removal` | **[12]** | Low-rank SVD decomposition ← **recommended** |

---

## References

| # | Citation |
|---|----------|
| [12] | M. Mercuri et al., "Automatic radar-based 2-D localization exploiting vital signs signatures," *Scientific Reports*, vol. 12, p. 7651, 2022. |
| [14] | M. Alizadeh et al., "Remote Monitoring of Human Vital Signs Using mm-Wave FMCW Radar," *IEEE Access*, vol. 7, pp. 54958–54968, 2019. |

---

## Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| Raw data + Beamforming | DCA1000 parser + CBF | ✅ Done (seniors) |
| Stage 1: DC Correction | Mean subtraction + Circle-fit | ✅ Done |
| Stage 2: Clutter Removal | SVD rank-1 | ✅ Done |
| Phase Demodulation | arctan + unwrap | ✅ Done |
| Vital Sign Estimation | FFT peak + AutoCorr | ✅ Done |
| **Stage 3** | Automated range/angle bin selection (CFAR / ML) | 🔲 Planned |
| **Stage 4** | RDA cube + multi-target tracking + Doppler FFT | 🔲 Planned |
