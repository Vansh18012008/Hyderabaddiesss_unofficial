import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
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

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

md_1 = """# FMCW Radar Vital Sign Pipeline — QIDK Deployment & Performance Analysis
## Prepared for QIDK Officials

This notebook details the deployment strategy for the FMCW radar vital sign estimation pipeline on the Qualcomm Intelligent Devices Kit (QIDK). It also provides a comprehensive breakdown of the pipeline latency and execution time to demonstrate its suitability for real-time edge deployment.

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

## 2. Execution Time & Latency Analysis

The vital sign pipeline consists of the following key stages:
1. **I/O Read:** Parsing the raw 16-bit binary LVDS capture.
2. **Beamforming (CBF):** Spatial filtering over the 4 RX antennas across 73 azimuth angles.
3. **Stage 1a (DC Removal):** Subtracting temporal mean per angle bin to remove static offsets.
4. **Range FFT:** Fast Fourier Transform along the fast-time axis to resolve ranges.
5. **Stage 2 (SVD Clutter Removal):** Low-rank SVD decomposition to remove static zero-Doppler clutter (like walls/furniture).
6. **Target Detection:** Identifying the primary subject in the Range-Azimuth heatmap.
7. **Phase Demodulation & Vitals:** Circle-fit DC correction, phase unwrapping, bandpass filtering, and FFT/Auto-correlation for Heart Rate (HR) and Breathing Rate (BR).

The code cell below processes a batch of files and visualises the latency of each stage."""

code_1 = """import os, sys, glob, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the pipeline directory to path so we can import modules
sys.path.insert(0, os.path.abspath('.'))

from qidk_pipeline import process_single_file

data_folder = './Single_Subject/'
# Select the first 10 files for this latency test
files = sorted(glob.glob(os.path.join(data_folder, '*.bin')))[:10]

print(f"Analyzing {len(files)} sample files for latency...")

latency_records = []

for f in files:
    # We use CBF method for this benchmark as it is the standard deployment mode
    start = time.perf_counter()
    # run quietly so we don't spam the notebook with terminal output
    res = process_single_file(f, method='cbf', quiet=True)
    total_e2e = (time.perf_counter() - start) * 1000
    
    if 'cbf' in res and 'timing' in res['cbf']:
        timing = res['cbf']['timing']
        latency_records.append({
            'File': os.path.basename(f),
            'Read I/O (ms)': timing.get('read_ms', 0),
            'Beamforming (ms)': timing.get('cbf_ms', 0),
            'DC Correction 1a (ms)': timing.get('dc1a_ms', 0),
            'Range FFT (ms)': timing.get('rfft_ms', 0),
            'SVD Clutter Removal (ms)': timing.get('svd_ms', 0),
            'Target Detection (ms)': timing.get('detect_ms', 0),
            'Vital Signs & Phase (ms)': timing.get('vitals_ms', 0),
            'Pipeline Total (ms)': timing.get('total_ms', 0),
            'E2E with Meta (ms)': total_e2e
        })

df_latency = pd.DataFrame(latency_records)
# Display the average latency for each stage
display(df_latency.drop(columns=['File']).mean().to_frame("Average Latency (ms)"))"""

md_2 = """### Visualization of Latency Breakdown
The following plots illustrate the proportion of time spent in each pipeline stage and the consistency of the pipeline execution time across multiple samples."""

code_2 = """# Calculate mean values for the pie chart (excluding totals)
stages = ['Read I/O (ms)', 'Beamforming (ms)', 'DC Correction 1a (ms)', 
          'Range FFT (ms)', 'SVD Clutter Removal (ms)', 'Target Detection (ms)', 
          'Vital Signs & Phase (ms)']
means = df_latency[stages].mean()

plt.figure(figsize=(10, 6))
plt.pie(means, labels=stages, autopct='%1.1f%%', startangle=140, 
        colors=plt.cm.tab20.colors, textprops={'fontsize': 10})
plt.title('Average Latency Breakdown per Pipeline Stage', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()

# Bar chart of the total pipeline execution times
plt.figure(figsize=(10, 5))
plt.bar(df_latency.index, df_latency['Pipeline Total (ms)'], color='teal', alpha=0.7)
plt.axhline(df_latency['Pipeline Total (ms)'].mean(), color='red', linestyle='--', label=f"Average: {df_latency['Pipeline Total (ms)'].mean():.1f} ms")
plt.xlabel('Sample Index')
plt.ylabel('Total Execution Time (ms)')
plt.title('Pipeline Execution Time Across Samples')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()"""

md_3 = """## 3. Real-Time Feasibility for QIDK

To determine if the pipeline can run in real-time, we compare the total execution latency to the data collection duration.
Each data capture consists of 512 chirps collected at a rate of roughly 20 Hz (based on 512 chirps over 25.6s), representing **25.6 seconds** of physical radar data."""

code_3 = """avg_pipeline_ms = df_latency['Pipeline Total (ms)'].mean()
physical_data_time_ms = 25.6 * 1000

print(f"Average Pipeline Execution Time : {avg_pipeline_ms:.2f} ms")
print(f"Physical Data Capture Duration  : {physical_data_time_ms:.2f} ms")

speedup = physical_data_time_ms / avg_pipeline_ms
print(f"\\nConclusion: The pipeline processes data {speedup:.1f}x faster than real-time.")
print("This makes it highly suitable for real-time edge deployment on the QIDK.")"""

add_md(md_1)
add_code(code_1)
add_md(md_2)
add_code(code_2)
add_md(md_3)
add_code(code_3)

with open("/home/Vansh_1801/.gemini/antigravity-ide/scratch/hyderabaddies/fmcw_radar_preprocessing/QIDK_Deployment_and_Latency_Analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully.")
