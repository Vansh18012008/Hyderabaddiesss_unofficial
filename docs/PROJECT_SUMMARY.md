# Project Summary — FMCW Radar Vital Sign Estimation
## IC~WiBES Research Group

---

## 1. What We Have Done

### Overview

This project implements a **contact-free, 77 GHz FMCW radar pipeline** for real-time estimation
of **Breathing Rate (BR)** and **Heart Rate (HR)** from raw radar binary captures.

The work is structured in four progressive stages. Stages 1 and 2 are **fully implemented**
in this iteration. Stages 3 and 4 are **roadmapped** for future work.

---

## 2. Functions Implemented

### 2.1 Senior-Designed Functions (Inherited, Preserved)

These functions were developed by the previous research team and are kept **unchanged**.

| Function | File | Description | Reference |
|----------|------|-------------|-----------|
| `readDCA1000` | Notebook | Reads raw 16-bit IQ binary from DCA1000 capture board. Handles 4-lane LVDS interleaving, complex-mode reshaping, and sign extension for <16-bit ADCs. | DCA1000 datasheet |
| `processIFSignal` | Notebook | Wraps `readDCA1000` and reshapes output to `(num_rx, num_chirps, num_samples)`. | — |
| `cbfweights` | Notebook | Conventional Beamformer (CBF) steering vector weights. Computes `w(m, θ) = exp(−j·2π·pos_m·sin θ) / N_ele`. Optional phase quantisation for hardware phase shifters. | — |
| `performBeamforming` | Notebook | Applies CBF weights via einsum to produce `(num_angles, num_chirps, num_samples)` from the 4-RX input. | — |
| `AutoCorr` / `autocorr` | Notebook | Normalised autocorrelation-based rate estimator. Finds first peak in the ACF and converts lag to bpm. | — |

---

### 2.2 Stage 1 — DC Offset Correction

> **Reference: [12] Mercuri 2022, [14] Alizadeh 2019**

DC offset corrupts the IQ constellation: the circular arc traced by chest motion is shifted
off-centre, making `arctan(Q/I)` demodulation inaccurate.

**Physical cause:**  
LO self-mixing in the homodyne receiver creates a constant complex phasor bias. Additionally,
**[14]** proves that strong stationary reflectors (walls) are *translated into DC offsets* in the
synthesised slow-time IQ signals — the two artefacts are inseparable before clutter removal.

| Function | Module | Method | Reference |
|----------|--------|--------|-----------|
| `mean_subtraction` | `dc_offset_correction.py` | Subtract temporal mean of complex slow-time signal. Simple and fast. | [14] |
| `circle_fit_correction` | `dc_offset_correction.py` | **Taubin algebraic circle fit** on IQ constellation. Finds exact centre (DC offset) by fitting a circle to the IQ arc. Most accurate method. | **[12]** |

**Applied in the notebook (two stages):**

- **Stage 1a** (before Range FFT): `mean_subtraction` applied to the full beamformed matrix
  `(num_chirps, num_samples)` per angle bin — removes the average DC across chirps per sample.
- **Stage 1b** (after Range FFT, at target range bin): `circle_fit_correction` on the slow-time IQ
  signal `(num_chirps,)` extracted from the peak range bin — gives the most precise DC removal
  before arctangent phase demodulation.

---

### 2.3 Stage 2 — Static Clutter Removal

> **Reference: [12] Mercuri 2022, [14] Alizadeh 2019**

Static reflectors (walls, furniture, radar housing) produce **zero-Doppler, constant-phase**
returns that dominate every range bin. They obscure the weak chest micro-motion signal
(~mm-level chest displacement corresponds to milli-degree phase shifts).
autocorrelation
| Function | Module | Method | Reference |
|----------|--------|--------|-----------|
| `mean_clutter_removal` | `clutter_suppression.py` | Subtract mean range profile across all chirps. Simplest baseline. | [14] |
| `svd_clutter_removal` | `clutter_suppression.py` | **SVD low-rank decomposition**: `S = UΣVᴴ`. Remove rank-k clutter approximation. Best performance for strong, specular clutter. | **[12]** |

**Applied in the notebook:**  
`svd_clutter_removal(rank=1)` on `(num_chirps, num_samples)` per angle bin —
removes the dominant static component (rank-1 clutter).

---

### 2.4 Supporting Functions (Preprocessing Module)

| Function | Module | Description |
|----------|--------|-------------|
| `range_fft` | `range_fft.py` | Windowed FFT along fast-time axis; converts to physical range. |
| `make_window` | `range_fft.py` | Hanning / Hamming / Blackman-Harris window generation. |
| `find_target_range_bin` | `range_fft.py` | Max-power or known-range bin selection. |
| `extract_slow_time_signal` | `range_fft.py` | Extract complex slow-time signal from range bin. |
| `extract_phase` | `phase_demodulation.py` | `arctan2(Q, I)` wrapped phase extraction. |
| `unwrap_phase` | `phase_demodulation.py` | `np.unwrap` for continuous phase. |
| `phase_to_displacement` | `phase_demodulation.py` | `d(t) = φ(t)·λ / (4π)` |
| `demodulate` | `phase_demodulation.py` | End-to-end: extract → unwrap → DC remove → displacement. |
| `bandpass_filter` | `vital_sign_extraction.py` | Zero-phase Butterworth bandpass. |
| `extract_respiration_signal` | `vital_sign_extraction.py` | 0.1–0.5 Hz respiration bandpass. |
| `extract_heartbeat_signal` | `vital_sign_extraction.py` | 0.8–2.0 Hz heartbeat bandpass. |
| `fft_peak_frequency` | `vital_sign_extraction.py` | 4× zero-padded FFT peak detection within frequency band. |
| `autocorr_peak_frequency` | `vital_sign_extraction.py` | Autocorrelation-based rate estimation (more robust for short signals). |
| `extract_vital_signs` | `vital_sign_extraction.py` | Full vital sign extraction pipeline: filter → estimate → return BR + HR. |
| `FMCWDataGenerator` | `data_generator.py` | Synthetic data cube generator (beat signal model from [12, 14]). |

---

## 3. Complete Processing Chain (As Implemented)

```
Step  Input shape                  Operation                         Reference
────  ─────────────────────────    ──────────────────────────────    ─────────
 0    .bin file                    readDCA1000 + processIFSignal     Seniors
      → (4, 512, 512)

 1    (4, 512, 512)                performBeamforming (CBF)          Seniors
      → (73, 512, 512)             73 angles: +90° to -90° @ 2.5°

STAGE 1a
 2    (73, 512, 512)               mean_subtraction per angle bin    [14]
      → res_dc (73, 512, 512)      subtract temporal mean per sample

STAGE 2
 3    (73, 512, 512)               svd_clutter_removal rank=1        [12]
      → res_clean (73, 512, 512)   remove dominant static clutter

 4    (73, 512, 512)               Range FFT (Hanning, N=512)        standard
      → (73, 512, 256)             one-sided spectrum

STAGE 1b
 5    (512,) at [peak_angle,       circle_fit_correction             [12]
              peak_range]          Taubin IQ circle fit

 6    (512,) corrected IQ          demodulate                        [12, 14]
      → displacement (mm)          arctan + unwrap + DC + λ/4π

 7    displacement (mm)            extract_vital_signs               [12, 14]
      → BR (bpm), HR (bpm)         bandpass + FFT peak + AutoCorr
```

---

## 4. Stage 3 Roadmap — Automated Subject Localisation

> **Goal:** Automatically determine the range bin and angle bin of the subject(s)
> without manual selection.

### Problem
The current pipeline requires manual identification of the peak in the Range-Azimuth heatmap.
This is impractical for real-world deployment with moving subjects or multiple people.

### Approach 1 — Signal Processing (CFAR)

**Constant False Alarm Rate (CFAR) Detection:**
- Apply 2D Cell-Averaging CFAR on the Range-Azimuth power map.
- Outputs a list of detected (range_bin, angle_bin) pairs above the adaptive threshold.
- Advantage: computationally light, no training data needed.
- Reference: standard radar CFAR — also described in [12] for multi-target scenarios.

```
Steps:
1. Compute RA heatmap: rangefft_angle_bins_mean  [73 × 256]
2. 2D CA-CFAR with guard cells (2, 2) and noise cells (4, 4)
3. Peak grouping: merge adjacent detections (non-maximum suppression)
4. Output: list of (range_m, angle_deg) per detected subject
```

### Approach 2 — AI/ML (Preferred) — Deep Learning Localisation

Train a lightweight CNN or U-Net on labelled Range-Azimuth heatmaps to output a
**bounding box or keypoint** for each subject.

```
Architecture (proposed):
  Input:  RA heatmap  [73 × 256 × 1]
  Backbone: MobileNetV2 or small ResNet-18
  Head: regression → (range_bin, angle_bin) per subject
  Output: [(r1, θ1), (r2, θ2), ...] — up to N subjects

Training data:
  - Collect labelled captures with known subject positions
  - Augment: random SNR, clutter levels, subject counts

Metrics:
  - Mean Absolute Position Error (MAPE) in metres and degrees
```

### Extension to Stage 3
Per **[12]** (Section IV), the angular separation can also be refined using
**Bartlett or Capon beamformers** after initial CFAR detection — giving sub-bin accuracy
without a neural network.

---

## 5. Stage 4 Roadmap — 3D RDA Data Cube & Multi-Target Tracking

> **Goal:** Extend from a 2D Range-Azimuth (RA) map to a full 3D
> **Range-Doppler-Azimuth (RDA)** data cube, and track subjects as they move.

### 5.1 RDA Data Cube Construction

The current pipeline processes chirps individually (slow-time axis = chirps within one frame).
For a full RDA cube, we add a **Doppler FFT** along the slow-time axis:

```
Step 1: Range FFT  (fast-time, per chirp)
        (num_chirps, num_samples) → (num_chirps, num_range_bins)

Step 2: Doppler FFT  (slow-time, per range bin)
        (num_chirps, num_range_bins) → (num_doppler_bins, num_range_bins)
        Doppler resolution: Δv = λ / (2 · num_chirps · T_chirp_interval)

Step 3: Angle FFT / Beamforming  (across RX antennas)
        → (num_angles, num_doppler_bins, num_range_bins)  [RDA cube]
```

The **RDA cube** provides:
- **Range bins**: target distance
- **Doppler bins**: target radial velocity (separates breathing from walking)
- **Angle bins**: target bearing

### 5.2 Modified Processing for RDA

Modifications required to the existing pipeline:

| Current | Modified for RDA |
|---------|-----------------|
| Range FFT per angle bin (serial) | Range FFT → Doppler FFT → Angle FFT (3D pipeline) |
| `svd_clutter_removal` on 2D matrix | Apply SVD on 3D cube (slice-wise or tensor decomposition) |
| Peak detection on RA map | Peak detection on RDA cube (3D CFAR) |
| Phase demodulation at fixed bin | Phase demodulation on tracked bin trajectory |

### 5.3 Multi-Target Tracking

For a moving subject, the detected range/angle bin changes over time.
A **tracker** associates detections across frames:

```
Proposed Architecture:
  Detector : 3D CFAR on RDA cube → detections [(r, v, θ)] per frame
  Tracker  : Extended Kalman Filter (EKF) or JPDA
             State: [range, angle, radial_velocity, BR_phase, HR_phase]
             Measurement: [range_bin, doppler_bin, angle_bin]
  Output   : Smooth trajectories per subject + real-time BR/HR per track
```

**Reference [12]** demonstrates 2D localisation + vital sign extraction for a stationary
subject. The RDA cube extension enables:
1. Separation of multiple subjects by Doppler velocity
2. Tracking a subject walking across the FoV
3. Gating vital sign extraction to the subject's current position

### 5.4 Key Parameters for Doppler FFT

| Parameter | Formula | Typical value |
|-----------|---------|---------------|
| Doppler resolution | `Δv = λ / (2·M·T_pri)` | ~5 mm/s |
| Max unambiguous velocity | `v_max = λ / (4·T_pri)` | ~0.5 m/s |
| Doppler FFT size | `N_doppler = M = 512` | 512 bins |
| Vital sign Doppler | `v_resp ≈ ±15 mm/s` | bins ≈ ±3 |

---

## 6. Summary Table

| Stage | Status | Description |
|-------|--------|-------------|
| Raw data reading (DCA1000) | ✅ Done | Seniors' `readDCA1000` |
| CBF Beamforming | ✅ Done | Seniors' `performBeamforming` |
| Stage 1a: DC Offset (mean subtraction) | ✅ Done | [14] |
| Stage 1b: DC Offset (circle-fit at range bin) | ✅ Done | [12] |
| Stage 2: SVD Clutter Removal | ✅ Done | [12] |
| Range FFT + Phase Demodulation | ✅ Done | [12, 14] |
| Vital Sign Extraction (BR + HR) | ✅ Done | [12, 14] |
| **Stage 3: Automated Localisation** | 🔲 Planned | CFAR / CNN (preferred) |
| **Stage 4: RDA Cube + Tracking** | 🔲 Planned | Doppler FFT + EKF |

---

## 7. References

| # | Full Citation |
|---|---------------|
| [12] | M. Mercuri, I. R. Lorato, Y.-H. Liu, F. Wieringa, C. Van Hoof, and T. Torfs, "Automatic radar-based 2-D localization exploiting vital signs signatures," *Scientific Reports*, vol. 12, p. 7651, May 2022. doi: 10.1038/s41598-022-11671-1 |
| [14] | M. Alizadeh, G. Shaker, J. C. M. De Almeida, P. P. Morita, and S. Safavi-Naeini, "Remote Monitoring of Human Vital Signs Using mm-Wave FMCW Radar," *IEEE Access*, vol. 7, pp. 54958–54968, 2019. doi: 10.1109/ACCESS.2019.2912420 |
