"""
heterogeneous/pipeline_manager.py
=================================
Tri-Engine Heterogeneous Pipeline Manager (CPU + GPU + NPU).

Coordinates the concurrent execution of FMCW radar signal processing across:
  - CPU Ingest Worker:  File I/O & LVDS Deserialization
  - GPU Compute Worker: CBF Beamforming, Range-FFT, and Batched SVD Clutter Removal
  - NPU Detect Worker:  Target Localisation & Feature Extraction
  - CPU Vitals Worker:  Taubin Circle Fit, Demodulation & Rate Estimation

Operating Modes:
  1. Synchronous Frame Mode: End-to-end execution of a single capture with detailed profiling.
  2. Asynchronous Pipelined Mode: Double-buffered concurrent streaming across all three hardware engines.
"""

import time
import queue
import threading
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from .cpu_io_engine import CPUIOEngine
from .gpu_engine import GPURadarEngine
from .npu_engine import NPURadarEngine
from .cpu_vitals_engine import CPUPostProcessEngine

class HeterogeneousPipeline:
    """
    Manager for heterogeneous execution across CPU, GPU, and NPU.
    """
    def __init__(
        self,
        range_axis: Optional[np.ndarray] = None,
        azimuth_grid: Optional[np.ndarray] = None,
        fs_vital: float = 20.0,
        c: float = 3e8,
        bw: float = 2.982e9,
        tm: float = 100e-6,
        f_adc: float = 600e3,
        f_min: float = 77e9,
        f_max: float = 79.982e9,
        num_rx: int = 4,
        m_chirps: int = 512,
        n_chirps: int = 512,
        n_fft_range: int = 512,
    ):
        self.c = c
        self.bw = bw
        self.tm = tm
        self.f_adc = f_adc
        self.f_min = f_min
        self.f_max = f_max
        self.num_rx = num_rx
        self.m_chirps = m_chirps
        self.n_chirps = n_chirps
        self.n_fft_range = n_fft_range
        self.fs_vital = fs_vital

        self.r_max = (c * tm / 2.0) * (f_adc / bw)
        self.lambda_c = c / ((f_min + f_max) / 2.0)

        if range_axis is None:
            self.range_axis = np.linspace(0, self.r_max, n_fft_range // 2)
        else:
            self.range_axis = range_axis

        if azimuth_grid is None:
            self.azimuth_grid = np.arange(90, -90 - 2.5, -2.5)
        else:
            self.azimuth_grid = azimuth_grid

        # Initialize hardware engines
        self.io_engine = CPUIOEngine(
            num_rx=num_rx,
            m_chirps=m_chirps,
            n_chirps=n_chirps
        )
        self.gpu_engine = GPURadarEngine(
            num_rx=num_rx,
            m_chirps=m_chirps,
            n_chirps=n_chirps,
            n_fft_range=n_fft_range,
            d_ant=(c / f_min) / 2.0,
            lambda_c=self.lambda_c,
            azimuth_grid=self.azimuth_grid
        )
        self.npu_engine = NPURadarEngine(
            range_axis=self.range_axis,
            azimuth_grid=self.azimuth_grid,
            min_range_m=0.3
        )
        self.vitals_engine = CPUPostProcessEngine(
            fs_vital=fs_vital,
            lambda_c=self.lambda_c
        )

    def process_single_frame(self, bin_path: str) -> Dict[str, Any]:
        """
        Executes a single frame synchronously through the heterogeneous pipeline stages
        while recording the latency contribution of each hardware engine.
        """
        timing = {}
        t_start = time.perf_counter()

        # Stage 1: CPU I/O & Ingestion
        if_data, io_timing = self.io_engine.read_frame(bin_path)
        timing.update(io_timing)

        # Stage 2: GPU Beamforming, Batched FFT, and Clutter Suppression
        cleaned_cube, ra_heatmap, gpu_timing = self.gpu_engine.process_frame(if_data)
        timing.update(gpu_timing)

        # Stage 3: NPU Target Detection & Slow-Time IQ Extraction
        targets, slow_iq, npu_timing = self.npu_engine.process_heatmap(ra_heatmap, cleaned_cube)
        timing.update(npu_timing)

        # Stage 4: CPU Demodulation, Circle-fit, and Vital Sign Estimation
        vitals, vitals_timing = self.vitals_engine.extract_vitals(slow_iq)
        timing.update(vitals_timing)

        timing['total_e2e_ms'] = (time.perf_counter() - t_start) * 1e3
        timing['total_ms'] = timing['total_e2e_ms']

        primary = targets[0] if len(targets) > 0 else {'range_m': 0.0, 'angle_deg': 0.0}

        return {
            'method':         'TRI_ENGINE_HETEROGENEOUS',
            'hr_fft':         vitals['hr_fft'],
            'br_fft':         vitals['br_fft'],
            'hr_ac':          vitals['hr_ac'],
            'br_ac':          vitals['br_ac'],
            'range_m':        primary['range_m'],
            'angle_deg':      primary['angle_deg'],
            'ra_heatmap':     ra_heatmap,
            'slow_iq':        slow_iq,
            'disp_mm':        vitals['disp_mm'],
            'resp_signal':    vitals['resp_signal'],
            'heart_signal':   vitals['heart_signal'],
            'num_targets':    len(targets),
            'all_targets':    targets,
            'timing':         timing,
        }

    def process_batch_asynchronous(
        self,
        bin_paths: List[str],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a batch of frames concurrently across CPU, GPU, and NPU
        using double-buffered asynchronous queues.
        """
        # Intermediate decoupling queues
        raw_queue   = queue.Queue(maxsize=4)  # CPU Ingest -> GPU Engine
        gpu_queue   = queue.Queue(maxsize=4)  # GPU Engine -> NPU Engine
        npu_queue   = queue.Queue(maxsize=4)  # NPU Engine -> CPU Vitals
        results     = []
        results_lock = threading.Lock()

        # Thread 1: CPU Data Ingestion
        def _ingest_worker():
            for p in bin_paths:
                if_data, t_io = self.io_engine.read_frame(p)
                raw_queue.put({'path': p, 'data': if_data, 'timing': t_io})
            raw_queue.put(None)  # Sentinel

        # Thread 2: GPU Acceleration (CBF, Range-FFT, SVD Clutter)
        def _gpu_worker():
            while True:
                item = raw_queue.get()
                if item is None:
                    gpu_queue.put(None)
                    break
                cleaned_cube, ra_heatmap, t_gpu = self.gpu_engine.process_frame(item['data'])
                item['timing'].update(t_gpu)
                gpu_queue.put({
                    'path': item['path'],
                    'cleaned_cube': cleaned_cube,
                    'ra_heatmap': ra_heatmap,
                    'timing': item['timing']
                })

        # Thread 3: NPU Detection (Target Localisation & Feature Extraction)
        def _npu_worker():
            while True:
                item = gpu_queue.get()
                if item is None:
                    npu_queue.put(None)
                    break
                targets, slow_iq, t_npu = self.npu_engine.process_heatmap(
                    item['ra_heatmap'],
                    item['cleaned_cube']
                )
                item['timing'].update(t_npu)
                npu_queue.put({
                    'path': item['path'],
                    'targets': targets,
                    'slow_iq': slow_iq,
                    'ra_heatmap': item['ra_heatmap'],
                    'timing': item['timing']
                })

        # Thread 4: CPU Post-Processing (Circle-fit & Vital Signs)
        def _vitals_worker():
            while True:
                item = npu_queue.get()
                if item is None:
                    break
                vitals, t_vitals = self.vitals_engine.extract_vitals(item['slow_iq'])
                item['timing'].update(t_vitals)

                primary = item['targets'][0] if len(item['targets']) > 0 else {'range_m': 0.0, 'angle_deg': 0.0}
                res = {
                    'file':           item['path'],
                    'method':         'TRI_ENGINE_PIPELINED',
                    'hr_fft':         vitals['hr_fft'],
                    'br_fft':         vitals['br_fft'],
                    'hr_ac':          vitals['hr_ac'],
                    'br_ac':          vitals['br_ac'],
                    'range_m':        primary['range_m'],
                    'angle_deg':      primary['angle_deg'],
                    'ra_heatmap':     item['ra_heatmap'],
                    'disp_mm':        vitals['disp_mm'],
                    'resp_signal':    vitals['resp_signal'],
                    'heart_signal':   vitals['heart_signal'],
                    'num_targets':    len(item['targets']),
                    'all_targets':    item['targets'],
                    'timing':         item['timing'],
                }
                with results_lock:
                    results.append(res)
                if callback:
                    callback(res)

        threads = [
            threading.Thread(target=_ingest_worker, name='CPUIOThread'),
            threading.Thread(target=_gpu_worker, name='GPUComputeThread'),
            threading.Thread(target=_npu_worker, name='NPUDetectThread'),
            threading.Thread(target=_vitals_worker, name='CPUVitalsThread'),
        ]

        t_pipeline_start = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        total_time_s = time.perf_counter() - t_pipeline_start

        return results, total_time_s
