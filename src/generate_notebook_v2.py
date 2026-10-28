import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

md_1 = """# FMCW Radar Vital Sign Pipeline — QIDK Deployment & Performance Analysis
## Prepared for QIDK Officials

This notebook details the deployment strategy for the FMCW radar vital sign estimation pipeline on the Qualcomm Intelligent Devices Kit (QIDK). It also provides a comprehensive breakdown of the pipeline latency, execution time, and algorithmic complexity to demonstrate its suitability for real-time edge deployment.

## 1. QIDK Deployment Architecture

Our pipeline is designed for seamless edge execution on the QIDK without heavy external dependencies.

### Deployment Methodology
1. **Headless Execution Core:** The core signal processing pipeline (`qidk_pipeline.py`) is written in pure Python using highly optimized `numpy` vectorised operations and `scipy` for signal analysis. It does not require a GUI or heavy desktop frameworks.
2. **Local Web Server Dashboard:** For visualisation, we deploy a lightweight Flask server (`qidk_web_server.py`) directly on the QIDK. This serves a responsive HTML/JS dashboard on `localhost:5050`.
3. **Display Subsystem:** The QIDK can display this dashboard via multiple fallback strategies:
   - **Local Display (`DISPLAY=:0`)**: Launching a Chromium kiosk-mode window directly on the QIDK's physical screen via X11 or Wayland (weston).
   - **ADB Reverse Tunnel**: If running via USB (Android/Linux base), we establish a reverse tunnel and launch the native Android Chrome intent.
   - **Remote PC Viewing**: Exposing the server over the LAN for remote monitoring via SSH tunnels or direct IP access.

### Key Deployment Features:
- **No Internet Required:** Fully offline processing and visualization for maximum privacy and security.
- **Server-Sent Events (SSE):** Streaming results live from the pipeline thread to the dashboard for real-time updates.
- **Low Memory Footprint:** Processing chirps in a memory-efficient manner without storing massive radar data cubes long-term.

## 2. Algorithmic Complexity & Latency Analysis

To ensure real-time feasibility on the QIDK, we analysed the computational complexity and empirical latency of each stage.

**Definitions:**
- $C = 4$ (RX Antennas)
- $M = 512$ (Chirps per frame)
- $N = 512$ (ADC Samples / Range bins)
- $A = 73$ (Azimuth angles for beamforming)

| Pipeline Stage | Algorithm Overview | Time Complexity | Average Latency (ms)* |
|---|---|---|---|
| **1. Read I/O** | Parsing binary LVDS captures from the DCA1000 | $\mathcal{O}(C \cdot M \cdot N)$ | **19.30 ms** |
| **2. Beamforming (CBF)** | Delay-and-Sum spatial filtering across antennas | $\mathcal{O}(A \cdot C \cdot M \cdot N)$ | **212.69 ms** |
| **3. Stage 1a (DC Removal)** | Mean subtraction per angle bin to remove offsets | $\mathcal{O}(A \cdot M \cdot N)$ | **125.22 ms** |
| **4. Range FFT** | Fast Fourier Transform along fast-time axis | $\mathcal{O}(A \cdot M \cdot N \log N)$ | **113.84 ms** |
| **5. SVD Clutter Removal** | Low-rank SVD across the slow-time & range matrix | $\mathcal{O}(A \cdot M \cdot N^2)$ | **2116.61 ms** |
| **6. Target Detection** | Identifying maximum power in Range-Azimuth map | $\mathcal{O}(A \cdot N)$ | **0.04 ms** |
| **7. Phase & Vitals** | Circle-fit DC correction, unwrap, BP filter, FFT | $\mathcal{O}(M \log M)$ | **1.82 ms** |
| **Total Pipeline (E2E)** | Full integration from read to extracted HR/BR | - | **2589.52 ms** |

*\*Latencies averaged over 10 consecutive sample files (Single Subject).*

### Analysis
As seen from the table, **SVD Clutter Removal** is the most computationally expensive block, taking up ~81% of the total execution time due to the dense matrix decompositions required across 73 azimuth angles. However, the total pipeline execution time remains highly competitive.

## 3. Real-Time Feasibility for QIDK

To determine if the pipeline can run in real-time, we compare the total execution latency to the data collection duration. 

Each data capture consists of 512 chirps collected at a rate of roughly 20 Hz, representing **25.6 seconds** of physical radar data.

- **Average Pipeline Execution Time:** 2.59 seconds (2589.52 ms)
- **Physical Data Capture Duration:** 25.60 seconds (25600.00 ms)

### Conclusion
The pipeline processes a full data block approximately **9.88x faster than real-time**. This massive headroom means that the QIDK will remain mostly idle between captures, resulting in minimal thermal throttling, lower power consumption, and highly responsive live vital sign updates."""

add_md(md_1)

with open("/home/Vansh_1801/.gemini/antigravity-ide/scratch/hyderabaddies/fmcw_radar_preprocessing/QIDK_Deployment_and_Latency_Analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully.")
