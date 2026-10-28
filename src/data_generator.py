"""
data_generator.py
=================
Synthetic FMCW radar data cube generator for testing preprocessing algorithms.

Signal Model (based on [12] Mercuri 2022 and [14] Alizadeh 2019):
-----------------------------------------------------------------
The FMCW beat signal for a target at range R with chest displacement d(t) is:

    s_beat(t_fast, t_slow) = A * exp(j * (4*pi*S*R / c * t_fast + 4*pi*R/lambda + phi_d(t_slow)))

where:
    - S      = chirp slope (B/Tc) [Hz/s]
    - R      = target range [m]
    - c      = speed of light [m/s]
    - lambda = wavelength [m]
    - phi_d  = phase from chest displacement

Static clutter is modeled as constant-amplitude reflectors at fixed ranges.
DC offset is modeled as a constant complex phasor shift in the IQ plane.

References:
    [12] Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14] Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Radar Configuration
# ---------------------------------------------------------------------------

@dataclass
class RadarConfig:
    """
    FMCW Radar Parameters.

    Defaults match a typical 77 GHz mmWave radar configured for vital signs.

    Key distinction:
        chirp_duration  : Active sweep time per chirp [s] (e.g., 40 µs)
        chirp_interval  : Time between consecutive chirp starts [s] (e.g., 50 ms)
                          This sets the slow-time (inter-chirp) sampling rate:
                          slow_time_fs = 1 / chirp_interval

        For vital signs monitoring we need slow_time_fs >> 2 * max_vital_sign_freq.
        Max vital sign: ~2 Hz (heartbeat).
        So slow_time_fs >= 10 Hz → chirp_interval <= 100 ms.
        Default: chirp_interval = 50 ms → slow_time_fs = 20 Hz.
    """
    fc: float = 77e9          # Center frequency [Hz]
    bandwidth: float = 4e9    # Sweep bandwidth [Hz]
    chirp_duration: float = 40e-6    # Active chirp duration Tc [s]
    chirp_interval: float = 50e-3   # Chirp repetition interval (PRI) [s]
    n_chirps: int = 128       # Number of chirps per frame (slow-time samples)
    n_samples: int = 256      # ADC samples per chirp (fast-time samples)
    n_rx: int = 1             # Number of RX antennas (1 = SISO)
    noise_power_db: float = -40.0  # AWGN noise power [dBm]

    # Derived parameters (computed in __post_init__)
    c: float = field(init=False, default=3e8)   # Speed of light [m/s]
    slope: float = field(init=False)             # Chirp slope [Hz/s]
    wavelength: float = field(init=False)        # Wavelength [m]
    range_resolution: float = field(init=False)  # Range resolution [m]
    max_range: float = field(init=False)          # Maximum unambiguous range [m]
    slow_time_fs: float = field(init=False)       # Slow-time sampling rate [Hz]
    fast_time_fs: float = field(init=False)       # Fast-time (ADC) sampling rate [Hz]

    def __post_init__(self):
        self.slope = self.bandwidth / self.chirp_duration
        self.wavelength = self.c / self.fc
        self.range_resolution = self.c / (2 * self.bandwidth)
        # Fast-time ADC rate (Nyquist for max beat frequency at max range)
        self.fast_time_fs = self.n_samples / self.chirp_duration
        self.max_range = (self.fast_time_fs * self.c) / (2.0 * self.slope)
        # Slow-time rate = 1 / chirp_interval (NOT 1/chirp_duration)
        self.slow_time_fs = 1.0 / self.chirp_interval


# ---------------------------------------------------------------------------
# Target Definition
# ---------------------------------------------------------------------------

@dataclass
class Target:
    """
    A single radar target (person or static reflector).

    Args:
        range_m:          Range from radar [m]
        rcs_db:           Radar cross-section [dBsm]
        is_human:         If True, adds cardiopulmonary chest motion
        resp_rate_hz:     Respiration rate [Hz]  (default: ~0.25 Hz = 15 breaths/min)
        heart_rate_hz:    Heart rate [Hz]         (default: ~1.17 Hz = 70 bpm)
        resp_amplitude_m: Respiration displacement amplitude [m]  (typical: 5-20 mm)
        heart_amplitude_m:Heartbeat displacement amplitude [m]  (typical: 0.2-0.5 mm)
        initial_phase_rad:Initial phase of chest oscillation [rad]
    """
    range_m: float = 1.5
    rcs_db: float = 0.0
    is_human: bool = True
    resp_rate_hz: float = 0.25          # 15 breaths/min
    heart_rate_hz: float = 1.17         # 70 bpm
    resp_amplitude_m: float = 8e-3      # 8 mm
    heart_amplitude_m: float = 0.3e-3   # 0.3 mm
    initial_phase_rad: float = 0.0


@dataclass
class ClutterReflector:
    """
    A static clutter reflector (wall, furniture, etc.)

    Args:
        range_m:  Range [m]
        rcs_db:   Radar cross-section [dBsm] — typically higher than human
        phase_rad: Fixed phase of clutter reflection [rad]
    """
    range_m: float = 2.0
    rcs_db: float = 10.0      # Clutter is typically stronger than human target
    phase_rad: float = 0.0


# ---------------------------------------------------------------------------
# Data Generator
# ---------------------------------------------------------------------------

class FMCWDataGenerator:
    """
    Generates synthetic FMCW radar data cubes for testing.

    The data cube has shape [n_chirps, n_samples] (slow-time x fast-time),
    containing complex IQ samples.

    Usage:
        cfg = RadarConfig()
        gen = FMCWDataGenerator(cfg)
        data_cube, ground_truth = gen.generate()
    """

    def __init__(self, config: RadarConfig, seed: int = 42):
        """
        Args:
            config: RadarConfig instance with radar parameters.
            seed:   Random seed for reproducibility.
        """
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def _fast_time_axis(self) -> np.ndarray:
        """Fast-time (intra-chirp) sample time axis [s]."""
        return np.arange(self.cfg.n_samples) / self.cfg.fast_time_fs

    def _slow_time_axis(self) -> np.ndarray:
        """Slow-time (inter-chirp) axis in seconds."""
        return np.arange(self.cfg.n_chirps) * self.cfg.chirp_interval

    def _beat_signal_for_target(
        self,
        t_fast: np.ndarray,       # shape [n_samples]
        t_slow: np.ndarray,       # shape [n_chirps]
        target: Target,
    ) -> np.ndarray:
        """
        Generate beat signal for a human target at a given range.

        The phase of the beat signal is:
            phi(t_slow) = 4*pi*R/lambda + 4*pi*d(t_slow)/lambda

        where d(t_slow) = resp_amp * sin(2*pi*f_resp*t_slow + phi0)
                        + heart_amp * sin(2*pi*f_hr*t_slow + phi0)

        The beat frequency (range information) is:
            f_beat = 2 * S * R / c

        Returns:
            Complex beat signal, shape [n_chirps, n_samples]
        """
        R = target.range_m
        rcs_lin = 10 ** (target.rcs_db / 10.0)
        amplitude = np.sqrt(rcs_lin)

        # --- Fast-time: beat frequency encodes range ---
        f_beat = 2.0 * self.cfg.slope * R / self.cfg.c
        fast_signal = np.exp(1j * 2 * np.pi * f_beat * t_fast)  # [n_samples]

        # --- Slow-time: phase encodes chest displacement ---
        phi_static = 4.0 * np.pi * R / self.cfg.wavelength  # static range phase

        if target.is_human:
            # Chest displacement due to breathing and heartbeat
            d_resp = target.resp_amplitude_m * np.sin(
                2 * np.pi * target.resp_rate_hz * t_slow + target.initial_phase_rad
            )
            d_heart = target.heart_amplitude_m * np.sin(
                2 * np.pi * target.heart_rate_hz * t_slow + target.initial_phase_rad + np.pi / 4
            )
            d_total = d_resp + d_heart
            phi_motion = 4.0 * np.pi * d_total / self.cfg.wavelength
        else:
            phi_motion = np.zeros_like(t_slow)

        slow_signal = np.exp(1j * (phi_static + phi_motion))  # [n_chirps]

        # Outer product: [n_chirps, n_samples]
        beat = amplitude * np.outer(slow_signal, fast_signal)
        return beat

    def _beat_signal_for_clutter(
        self,
        t_fast: np.ndarray,
        reflector: ClutterReflector,
    ) -> np.ndarray:
        """
        Generate beat signal for a static clutter reflector.

        Static clutter has constant phase → same across all chirps (zero Doppler).

        Returns:
            Complex beat signal, shape [n_chirps, n_samples]
        """
        R = reflector.range_m
        rcs_lin = 10 ** (reflector.rcs_db / 10.0)
        amplitude = np.sqrt(rcs_lin)

        f_beat = 2.0 * self.cfg.slope * R / self.cfg.c
        phi_static = 4.0 * np.pi * R / self.cfg.wavelength + reflector.phase_rad

        fast_signal = amplitude * np.exp(
            1j * (2 * np.pi * f_beat * t_fast + phi_static)
        )  # [n_samples], constant across chirps

        # Broadcast to [n_chirps, n_samples] — same signal for every chirp
        clutter_signal = np.tile(fast_signal[np.newaxis, :], (self.cfg.n_chirps, 1))
        return clutter_signal

    def _add_dc_offset(self, data: np.ndarray, dc_i: float, dc_q: float) -> np.ndarray:
        """
        Add DC offset to the data cube.

        In hardware, DC offset arises from LO self-mixing and IQ mixer imbalances.
        Modeled as a constant complex offset (dc_i + j*dc_q) added to every sample.

        Args:
            data:  Complex data cube, shape [n_chirps, n_samples]
            dc_i:  DC offset on I channel [linear units]
            dc_q:  DC offset on Q channel [linear units]

        Returns:
            Data cube with DC offset applied.
        """
        return data + (dc_i + 1j * dc_q)

    def _add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add complex AWGN to the data cube."""
        noise_std = 10 ** (self.cfg.noise_power_db / 20.0)
        noise = noise_std * (
            self.rng.standard_normal(data.shape)
            + 1j * self.rng.standard_normal(data.shape)
        ) / np.sqrt(2)
        return data + noise

    def generate(
        self,
        targets: List[Target] = None,
        clutter: List[ClutterReflector] = None,
        dc_offset: Tuple[float, float] = (0.5, -0.3),
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate a full FMCW data cube.

        Args:
            targets:   List of Target objects. Defaults to one human at 1.5 m.
            clutter:   List of ClutterReflector objects. Defaults to two static objects.
            dc_offset: Tuple (dc_i, dc_q) — DC offset in linear amplitude units.

        Returns:
            data_cube:    Complex array, shape [n_chirps, n_samples]
            ground_truth: Dictionary with all known parameters for evaluation.
        """
        if targets is None:
            targets = [Target(range_m=1.5, resp_rate_hz=0.25, heart_rate_hz=1.17)]
        if clutter is None:
            clutter = [
                ClutterReflector(range_m=2.5, rcs_db=15.0, phase_rad=0.3),
                ClutterReflector(range_m=3.8, rcs_db=12.0, phase_rad=-0.8),
            ]

        t_fast = self._fast_time_axis()   # [n_samples]
        t_slow = self._slow_time_axis()   # [n_chirps]

        # Start with zeros
        data_cube = np.zeros((self.cfg.n_chirps, self.cfg.n_samples), dtype=complex)

        # Add human targets
        for tgt in targets:
            data_cube += self._beat_signal_for_target(t_fast, t_slow, tgt)

        # Add static clutter
        for refl in clutter:
            data_cube += self._beat_signal_for_clutter(t_fast, refl)

        # Add DC offset (hardware artifact)
        data_cube = self._add_dc_offset(data_cube, dc_offset[0], dc_offset[1])

        # Add AWGN noise
        data_cube = self._add_noise(data_cube)

        # Ground truth for evaluation
        ground_truth = {
            "targets": targets,
            "clutter": clutter,
            "dc_offset": dc_offset,       # True (dc_i, dc_q) values
            "t_slow": t_slow,             # Slow-time axis [s]
            "t_fast": t_fast,             # Fast-time axis [s]
            "radar_config": self.cfg,
        }

        return data_cube, ground_truth


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_single_target_scene(
    target_range: float = 1.5,
    resp_rate_hz: float = 0.25,
    heart_rate_hz: float = 1.17,
    dc_i: float = 0.5,
    dc_q: float = -0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, dict, RadarConfig]:
    """
    Create a simple single-human scene with two static clutter reflectors.

    Returns:
        data_cube, ground_truth, radar_config
    """
    cfg = RadarConfig()
    gen = FMCWDataGenerator(cfg, seed=seed)
    targets = [Target(range_m=target_range, resp_rate_hz=resp_rate_hz, heart_rate_hz=heart_rate_hz)]
    clutter = [
        ClutterReflector(range_m=2.5, rcs_db=15.0, phase_rad=0.3),
        ClutterReflector(range_m=3.8, rcs_db=12.0, phase_rad=-0.8),
    ]
    data_cube, gt = gen.generate(targets=targets, clutter=clutter, dc_offset=(dc_i, dc_q))
    return data_cube, gt, cfg


def make_multi_target_scene(seed: int = 42) -> Tuple[np.ndarray, dict, RadarConfig]:
    """
    Create a two-human scene (for multi-target experiments matching [12] and [15]).

    Returns:
        data_cube, ground_truth, radar_config
    """
    cfg = RadarConfig()
    gen = FMCWDataGenerator(cfg, seed=seed)
    targets = [
        Target(range_m=1.2, resp_rate_hz=0.22, heart_rate_hz=1.10, initial_phase_rad=0.0),
        Target(range_m=2.1, resp_rate_hz=0.30, heart_rate_hz=1.30, initial_phase_rad=np.pi / 3),
    ]
    clutter = [
        ClutterReflector(range_m=3.0, rcs_db=15.0, phase_rad=0.5),
        ClutterReflector(range_m=4.5, rcs_db=13.0, phase_rad=-1.2),
    ]
    data_cube, gt = gen.generate(targets=targets, clutter=clutter, dc_offset=(0.6, -0.4))
    return data_cube, gt, cfg


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== data_generator.py self-test ===")
    cube, gt, cfg = make_single_target_scene()
    print(f"  Radar config    : fc={cfg.fc/1e9:.1f} GHz, BW={cfg.bandwidth/1e9:.1f} GHz")
    print(f"  Range res       : {cfg.range_resolution*100:.2f} cm")
    print(f"  Max range       : {cfg.max_range:.2f} m")
    print(f"  Data cube shape : {cube.shape}  (n_chirps x n_samples)")
    print(f"  Data cube dtype : {cube.dtype}")
    print(f"  DC offset (true): I={gt['dc_offset'][0]}, Q={gt['dc_offset'][1]}")
    print(f"  Target range    : {gt['targets'][0].range_m} m")
    print(f"  Resp rate       : {gt['targets'][0].resp_rate_hz} Hz")
    print(f"  Heart rate      : {gt['targets'][0].heart_rate_hz} Hz")
    print("  PASS")
