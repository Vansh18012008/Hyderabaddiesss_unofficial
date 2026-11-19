"""
heterogeneous/cpu_io_engine.py
==============================
CPU Asynchronous Ingestion & DCA1000 Binary Parser.

Workloads:
  1. DCA1000 raw 16-bit IQ unpacking (4-lane LVDS de-interleaving).
  2. Reshaping into complex64 IF signal tensor [NUM_RX, M_CHIRPS, N_CHIRPS].
  3. Pre-buffering into memory to feed the GPU compute stage without pipeline stalling.
"""

import time
import numpy as np
from typing import Tuple, Dict

class CPUIOEngine:
    """
    High-throughput DCA1000 binary reader and deserializer.
    """
    def __init__(self, num_rx: int = 4, m_chirps: int = 512, n_chirps: int = 512):
        self.num_rx = num_rx
        self.m_chirps = m_chirps
        self.n_chirps = n_chirps

    def read_frame(self, bin_path: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Reads and unpacks a DCA1000 capture file into an IF tensor.

        Args:
            bin_path: Path to .bin file

        Returns:
            if_data: Complex array [NUM_RX, M_CHIRPS, N_CHIRPS]
            timing: Execution latency in ms
        """
        t0 = time.perf_counter()
        with open(bin_path, 'rb') as f:
            raw = np.fromfile(f, dtype=np.int16)

        I = raw[0::4] + 1j * raw[2::4]
        Q = raw[1::4] + 1j * raw[3::4]
        LVDS = np.empty(I.size + Q.size, dtype=np.complex64)
        LVDS[0::2] = I
        LVDS[1::2] = Q

        num_chirps = raw.shape[0] // (2 * self.n_chirps * self.num_rx)
        LVDS = LVDS.reshape((self.n_chirps * self.num_rx, num_chirps), order='F').T

        out = np.zeros((self.num_rx, num_chirps * self.n_chirps), dtype=np.complex64)
        for rx in range(self.num_rx):
            for chp in range(num_chirps):
                out[rx, chp * self.n_chirps:(chp + 1) * self.n_chirps] = \
                    LVDS[chp, rx * self.n_chirps:(rx + 1) * self.n_chirps]

        if_data = out.reshape(self.num_rx, self.m_chirps, self.n_chirps).astype(np.complex64)
        timing = {'cpu_read_ms': (time.perf_counter() - t0) * 1e3}
        return if_data, timing
