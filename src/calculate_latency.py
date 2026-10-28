import os, sys, glob, time
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from qidk_pipeline import process_single_file
from heterogeneous import HeterogeneousPipeline

data_folder = './Single_Subject/'
files = sorted(glob.glob(os.path.join(data_folder, '*.bin')))[:10]

print(f"Benchmarking Latency on {len(files)} files...\n")

# 1. Baseline CPU Pipeline
cbf_metrics = {
    'read_ms': [], 'cbf_ms': [], 'dc1a_ms': [], 
    'rfft_ms': [], 'svd_ms': [], 'detect_ms': [], 
    'vitals_ms': [], 'total_ms': []
}

for f in files:
    res = process_single_file(f, method='cbf', quiet=True)
    if 'cbf' in res and 'timing' in res['cbf']:
        t = res['cbf']['timing']
        for k in cbf_metrics.keys():
            if k in t:
                cbf_metrics[k].append(t[k])

# 2. Tri-Engine Heterogeneous Pipeline (Synchronous Frame Latency)
tri_metrics = {
    'cpu_read_ms': [], 'gpu_cbf_ms': [], 'gpu_dc1a_ms': [],
    'gpu_rfft_ms': [], 'gpu_svd_ms': [], 'npu_detect_ms': [],
    'cpu_vitals_total_ms': [], 'total_e2e_ms': []
}

pipe = HeterogeneousPipeline()
for f in files:
    res = pipe.process_single_frame(f)
    t = res['timing']
    for k in tri_metrics.keys():
        if k in t:
            tri_metrics[k].append(t[k])

# 3. Asynchronous Pipelined Batch (Concurrency Across CPU, GPU, NPU)
_, async_total_s = pipe.process_batch_asynchronous(files)
async_eff_latency = (async_total_s / len(files)) * 1000.0
async_fps = len(files) / async_total_s

print("=" * 65)
print("  LATENCY & THROUGHPUT BENCHMARK: CPU vs. HETEROGENEOUS")
print("=" * 65)

print("\n1. BASELINE SINGLE-CORE CPU (ms):")
for k, v in cbf_metrics.items():
    print(f"   {k:<15}: {np.mean(v):>8.2f} ms")
cbf_total = np.mean(cbf_metrics['total_ms'])
print(f"   --> Frame Rate : {1000.0 / cbf_total:.2f} fps")

print("\n2. TRI-ENGINE HETEROGENEOUS (CPU + GPU + NPU) (ms):")
for k, v in tri_metrics.items():
    print(f"   {k:<20}: {np.mean(v):>8.2f} ms")
tri_total = np.mean(tri_metrics['total_e2e_ms'])
print(f"   --> Frame Rate       : {1000.0 / tri_total:.2f} fps")
print(f"   --> Latency Speedup  : {cbf_total / tri_total:.2f}x")

print("\n3. ASYNCHRONOUS PIPELINED CONCURRENCY (Triple Buffering):")
print(f"   --> Total Time for 10 captures : {async_total_s:.2f} s")
print(f"   --> Effective Frame Latency    : {async_eff_latency:.2f} ms")
print(f"   --> Pipelined Throughput       : {async_fps:.2f} fps")
print(f"   --> Throughput Speedup         : {async_fps / (1000.0 / cbf_total) if cbf_total > 0 else 0:.2f}x (from {1000.0 / cbf_total:.2f} to {async_fps:.2f} fps)")
print("=" * 65)
