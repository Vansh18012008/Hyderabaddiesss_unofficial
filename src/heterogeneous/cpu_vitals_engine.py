"""
heterogeneous/cpu_vitals_engine.py
==================================
CPU Post-Processing & Vital Signs Extraction Engine.

Workloads:
  1. Stage-1b Taubin Algebraic Circle-Fit DC Offset Correction on slow-time IQ [12].
  2. Phase Demodulation: Wrapped phase extraction, phase unwrapping, displacement [14].
  3. Butterworth Bandpass Filtering: Respiration (0.1–0.5 Hz) & Cardiac (0.8–2.5 Hz).
  4. Dual Rate Estimators: High-resolution FFT peak + Autocorrelation lag estimation [12, 14].
"""

import time
import numpy as np
from typing import Dict, Any, Optional

from dc_offset_correction import circle_fit_correction
from phase_demodulation import demodulate
from vital_sign_extraction import extract_vital_signs
from scipy.signal import find_peaks

class CPUPostProcessEngine:
    """
    CPU-based demodulation and biometric rate estimation engine.
    """
    def __init__(
        self,
        fs_vital: float = 20.0,
        lambda_c: float = 0.003822,
        resp_band: tuple = (0.1, 0.5),
        heart_band: tuple = (0.8, 2.5)
    ):
        self.fs_vital = fs_vital
        self.lambda_c = lambda_c
        self.resp_band = resp_band
        self.heart_band = heart_band

    def _autocorr_rate(self, signal: np.ndarray) -> Optional[float]:
        """Estimates rate in bpm via autocorrelation first peak."""
        x = signal - np.mean(signal)
        r = np.correlate(x, x, mode='full')
        r = r[r.size // 2:]
        r /= (r[0] + 1e-30)
        lags_s = np.arange(len(r)) / self.fs_vital
        peaks, _ = find_peaks(r, distance=int(self.fs_vital * 0.4))
        if len(peaks) == 0:
            return None
        T = lags_s[peaks][0]
        return (1.0 / T) * 60.0 if T > 0 else None

    def extract_vitals(self, slow_iq: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Executes end-to-end demodulation and vital sign extraction on a slow-time IQ signal.

        Args:
            slow_iq: Complex slow-time signal array [M_CHIRPS]

        Returns:
            vitals: Dictionary containing HR, BR, displacement, and filtered signals
            timing: Execution latency in milliseconds
        """
        t0 = time.perf_counter()
        timing = {}

        # 1. Stage-1b Circle-fit DC correction (Taubin fit on IQ constellation)
        t_sub = time.perf_counter()
        slow_iq_corr, circle_info = circle_fit_correction(slow_iq)
        timing['cpu_circle_fit_ms'] = (time.perf_counter() - t_sub) * 1e3

        # 2. Phase demodulation & continuous chest displacement (mm)
        t_sub = time.perf_counter()
        demod = demodulate(slow_iq_corr, wavelength_m=self.lambda_c, remove_dc_in_phase=True)
        disp_mm = demod['displacement_mm']
        timing['cpu_demod_ms'] = (time.perf_counter() - t_sub) * 1e3

        # 3. Filtering & FFT rate estimation
        t_sub = time.perf_counter()
        vs = extract_vital_signs(
            disp_mm,
            fs=self.fs_vital,
            resp_band=self.resp_band,
            heart_band=self.heart_band,
            estimator='fft'
        )
        timing['cpu_filter_fft_ms'] = (time.perf_counter() - t_sub) * 1e3

        # 4. Autocorrelation-based rate estimation
        t_sub = time.perf_counter()
        hr_ac = self._autocorr_rate(vs['heart_signal_mm'])
        br_ac = self._autocorr_rate(vs['resp_signal_mm'])
        timing['cpu_autocorr_ms'] = (time.perf_counter() - t_sub) * 1e3

        timing['cpu_vitals_total_ms'] = (time.perf_counter() - t0) * 1e3

        vitals = {
            'hr_fft':        vs['heart_rate_bpm'],
            'br_fft':        vs['resp_rate_bpm'],
            'hr_ac':         hr_ac,
            'br_ac':         br_ac,
            'disp_mm':       disp_mm,
            'resp_signal':   vs['resp_signal_mm'],
            'heart_signal':  vs['heart_signal_mm'],
            'circle_info':   circle_info,
        }
        return vitals, timing
