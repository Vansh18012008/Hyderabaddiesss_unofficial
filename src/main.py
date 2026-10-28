"""
main.py
=======
End-to-end FMCW radar preprocessing pipeline runner (synthetic-data mode).

This module exercises the full preprocessing stack on synthetic data,
validating each processing stage in isolation. For real QIDK deployment
on actual .bin data, use qidk_pipeline.py instead.

Modes:
  --mode=demo      : Run full pipeline on synthetic data, print results.
  --mode=evaluate  : Run all methods, print comparison tables.

Usage:
  python main.py --mode=demo
  python main.py --mode=evaluate
  python main.py --mode=demo --fc 77e9 --bandwidth 4e9 --n_chirps 256

For QIDK real-data processing (CBF + 2D MUSIC):
  python qidk_pipeline.py --file <path>.bin --method both
  python qidk_pipeline.py --folder ./Single_Subject/ --method music

Module layout:
  data_generator.py       -- Synthetic FMCW scene generator
  range_fft.py            -- Range FFT and target bin selection
  dc_offset_correction.py -- IQ DC offset correction methods [12][14]
  clutter_suppression.py  -- Static clutter removal methods [12][14]
  phase_demodulation.py   -- Arctangent phase demodulation [14]
  vital_sign_extraction.py-- Respiration + heart rate extraction [12][14]
  music_2d.py             -- 2D MUSIC algorithm for multi-object detection [NEW]
  qidk_pipeline.py        -- QIDK integrated pipeline with on-screen HR/BR display
  run_fixed_pipeline.py   -- Fixed AWR1843 pipeline (single-object CBF)
  run_folder.py           -- Folder-level batch processing script

References:
    [12]  Mercuri et al., Scientific Reports, 2022. doi:10.1038/s41598-022-11671-1
    [14]  Alizadeh et al., IEEE Access, 2019. doi:10.1109/ACCESS.2019.2912420
    [NEW] Van Rossum & Hoogeboom, "Application of 2D MUSIC Algorithm to
          Range-Azimuth FMCW Radar Data," EuMA 2012. (TNO/Delft University)
"""

import argparse
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Imports — all modules in this package
# ---------------------------------------------------------------------------
from data_generator import (
    RadarConfig, FMCWDataGenerator,
    make_single_target_scene, make_multi_target_scene,
)
from range_fft import (
    range_fft, find_target_range_bin,
    extract_slow_time_signal, compute_range_power_profile,
)
from dc_offset_correction import (
    apply_all_dc_methods,
    mean_subtraction, ema_subtraction,
    highpass_filter, circle_fit_correction,
)
from clutter_suppression import (
    apply_all_clutter_methods,
    mean_clutter_removal, svd_clutter_removal, frame_differencing,
    notch_filter_removal,
)
from phase_demodulation import demodulate
from vital_sign_extraction import extract_vital_signs


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "FMCW Radar Preprocessing -- Synthetic-Data Verification Tool.\n"
            "For real QIDK .bin data with on-screen HR/BR display, use qidk_pipeline.py."
        )
    )
    p.add_argument("--mode", choices=["demo", "evaluate"], default="demo",
                   help="Run mode (default: demo)")
    p.add_argument("--fc", type=float, default=77e9,
                   help="Carrier frequency [Hz] (default: 77e9)")
    p.add_argument("--bandwidth", type=float, default=4e9,
                   help="Bandwidth [Hz] (default: 4e9)")
    p.add_argument("--n_chirps", type=int, default=128,
                   help="Number of chirps per frame (default: 128)")
    p.add_argument("--n_samples", type=int, default=256,
                   help="ADC samples per chirp (default: 256)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--target_range", type=float, default=1.5,
                   help="Target range [m] (default: 1.5)")
    p.add_argument("--resp_hz", type=float, default=0.25,
                   help="Respiration rate [Hz] (default: 0.25 = 15 bpm)")
    p.add_argument("--hr_hz", type=float, default=1.17,
                   help="Heart rate [Hz] (default: 1.17 = 70 bpm)")
    p.add_argument("--dc_i", type=float, default=0.5,
                   help="True DC offset I channel (default: 0.5)")
    p.add_argument("--dc_q", type=float, default=-0.3,
                   help="True DC offset Q channel (default: -0.3)")
    return p


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> dict:
    """
    Execute the full preprocessing pipeline.

    Returns:
        results: Dictionary with all intermediate and final results.
    """
    print("\n" + "=" * 70)
    print("FMCW RADAR PREPROCESSING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    print("\n[Step 1] Generating synthetic FMCW data cube...")

    cfg = RadarConfig(
        fc=args.fc,
        bandwidth=args.bandwidth,
        n_chirps=args.n_chirps,
        n_samples=args.n_samples,
    )
    data_cube, gt, _ = make_single_target_scene(
        target_range=args.target_range,
        resp_rate_hz=args.resp_hz,
        heart_rate_hz=args.hr_hz,
        dc_i=args.dc_i,
        dc_q=args.dc_q,
        seed=args.seed,
    )

    print(f"  Data cube     : {data_cube.shape}  (n_chirps x n_samples)")
    print(f"  fc            : {cfg.fc/1e9:.1f} GHz")
    print(f"  Bandwidth     : {cfg.bandwidth/1e9:.1f} GHz")
    print(f"  Range res     : {cfg.range_resolution*100:.2f} cm")
    print(f"  Max range     : {cfg.max_range:.2f} m")
    print(f"  Wavelength    : {cfg.wavelength*1000:.2f} mm")
    print(f"  Slow-time fs  : {cfg.slow_time_fs:.0f} Hz")
    print(f"  Target range  : {args.target_range} m")
    print(f"  Resp rate     : {args.resp_hz} Hz ({args.resp_hz*60:.1f} bpm)")
    print(f"  Heart rate    : {args.hr_hz} Hz ({args.hr_hz*60:.1f} bpm)")
    print(f"  True DC (I,Q) : ({args.dc_i}, {args.dc_q})")

    # ------------------------------------------------------------------
    # Step 2: Range FFT
    # ------------------------------------------------------------------
    print("\n[Step 2] Computing Range FFT...")
    range_profile, range_axis = range_fft(data_cube, cfg, window_type="hanning")
    print(f"  Range profile shape: {range_profile.shape}")

    # --- Apply mean clutter suppression FIRST so target bin detection is accurate ---
    # Static clutter at 2.5 m / 3.8 m is much stronger than the human target.
    # Without suppression, max_power selects the clutter bin, not the human.
    range_profile_cleaned_for_detection, _ = mean_clutter_removal(range_profile)

    # Identify target range bin on the CLUTTER-REMOVED profile
    tgt_bin, tgt_range = find_target_range_bin(
        range_profile_cleaned_for_detection, range_axis, method="max_power"
    )
    print(f"  Target bin: {tgt_bin}  |  Estimated range: {tgt_range:.3f} m")
    print(f"  Range error: {abs(tgt_range - args.target_range)*100:.2f} cm")

    # Clutter range bins from ground truth
    clutter_bins = [
        int(np.argmin(np.abs(range_axis - c.range_m))) for c in gt["clutter"]
    ]
    clutter_ranges = [c.range_m for c in gt["clutter"]]
    print(f"  Clutter bins: {clutter_bins}  ranges: {[f'{r:.2f}m' for r in clutter_ranges]}")

    # Raw slow-time signal from ORIGINAL (uncluttered) profile at correct target bin
    # (used only for the "before" IQ constellation visualization)
    sts_raw = extract_slow_time_signal(range_profile, tgt_bin)

    # ------------------------------------------------------------------
    # Step 3: Static Clutter Suppression
    # ------------------------------------------------------------------
    print("\n[Step 3] Applying clutter suppression methods...")
    clutter_results = apply_all_clutter_methods(range_profile, svd_rank=1)
    for name in clutter_results:
        print(f"  [{name}] Done.")

    # ------------------------------------------------------------------
    # Step 4: DC Offset Correction
    # ------------------------------------------------------------------
    # Extract slow-time signal from CLUTTER-SUPPRESSED profile.
    # Using the raw profile here would give a DC-estimate dominated by clutter
    # (not the hardware DC offset) and a displacement signal too small for
    # vital sign extraction.
    print("\n[Step 4] Applying DC offset correction methods...")
    # Use mean-clutter-suppressed profile for all DC correction methods
    rp_mean_clean, _ = mean_clutter_removal(range_profile)
    sts_clean = extract_slow_time_signal(rp_mean_clean, tgt_bin)
    dc_results = apply_all_dc_methods(sts_clean, cfg.slow_time_fs)
    for name, (corrected, info) in dc_results.items():
        print(f"  [{name}] Est DC = ({info['dc_i']:.4f}, {info['dc_q']:.4f})")

    # ------------------------------------------------------------------
    # Step 5: Phase Demodulation (using circle-fit corrected signal)
    # ------------------------------------------------------------------
    print("\n[Step 5] Phase demodulation (mean-subtraction corrected signal)...")
    sts_corrected, _ = mean_subtraction(sts_clean)
    demod = demodulate(sts_corrected, cfg.wavelength, remove_dc_in_phase=True)
    disp_mm = demod["displacement_mm"]
    print(f"  Phase range : [{demod['phase_clean_rad'].min():.3f}, "
          f"{demod['phase_clean_rad'].max():.3f}] rad")
    print(f"  Displacement: [{disp_mm.min():.3f}, {disp_mm.max():.3f}] mm")

    # ------------------------------------------------------------------
    # Step 6: Vital Sign Extraction
    # ------------------------------------------------------------------
    print("\n[Step 6] Extracting vital signs (FFT estimator)...")
    vs_result = extract_vital_signs(disp_mm, cfg.slow_time_fs, estimator="fft")
    print(f"  Estimated respiration : {vs_result['resp_rate_hz']:.4f} Hz  "
          f"({vs_result['resp_rate_bpm']:.2f} bpm)  "
          f"[True: {args.resp_hz:.4f} Hz = {args.resp_hz*60:.2f} bpm]")
    print(f"  Estimated heart rate  : {vs_result['heart_rate_hz']:.4f} Hz  "
          f"({vs_result['heart_rate_bpm']:.2f} bpm)  "
          f"[True: {args.hr_hz:.4f} Hz = {args.hr_hz*60:.2f} bpm]")

    resp_err_bpm = abs(vs_result['resp_rate_hz'] - args.resp_hz) * 60
    hr_err_bpm   = abs(vs_result['heart_rate_hz'] - args.hr_hz) * 60
    print(f"  Resp MAE  : {resp_err_bpm:.2f} bpm")
    print(f"  HR MAE    : {hr_err_bpm:.2f} bpm")

    # ------------------------------------------------------------------
    # Step 7: Evaluation (inline metrics -- no external eval module)
    # ------------------------------------------------------------------
    print("\n[Step 7] Quantitative evaluation...")

    # DC correction errors
    true_dc = gt["dc_offset"]
    print("  DC correction method errors:")
    for name, (_, info) in dc_results.items():
        err_i = abs(info['dc_i'] - true_dc[0])
        err_q = abs(info['dc_q'] - true_dc[1])
        print(f"    [{name}] est=({info['dc_i']:.4f}, {info['dc_q']:.4f})  "
              f"err=({err_i:.4f}, {err_q:.4f})")

    # Clutter suppression: power reduction at clutter bins
    print("  Clutter suppression (power change at clutter bins):")
    for name, (cleaned, _) in clutter_results.items():
        n_c = cleaned.shape[0]
        pwr_in  = np.mean(np.abs(range_profile[:n_c, clutter_bins]) ** 2)
        pwr_out = np.mean(np.abs(cleaned[:, clutter_bins]) ** 2)
        db_change = 10 * np.log10((pwr_out + 1e-30) / (pwr_in + 1e-30))
        print(f"    [{name}] clutter power change: {db_change:+.1f} dB")

    # Vital sign errors
    resp_err_bpm = abs(vs_result['resp_rate_hz'] - args.resp_hz) * 60
    hr_err_bpm   = abs(vs_result['heart_rate_hz'] - args.hr_hz) * 60
    print(f"  Vital sign errors (FFT estimator):")
    print(f"    Resp MAE  : {resp_err_bpm:.2f} bpm")
    print(f"    HR MAE    : {hr_err_bpm:.2f} bpm")

    dc_scores = {
        name: {
            'err_i': abs(info['dc_i'] - true_dc[0]),
            'err_q': abs(info['dc_q'] - true_dc[1]),
        }
        for name, (_, info) in dc_results.items()
    }
    vs_scores = {
        'resp_err_bpm': resp_err_bpm,
        'hr_err_bpm':   hr_err_bpm,
    }

    print("\n[Step 8] Visualization (plotting removed -- use qidk_pipeline.py for dashboards)")
    print("  Tip: For full on-screen HR/BR QIDK display, run:")
    print("       python qidk_pipeline.py --file <path>.bin --method both")
    print()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    return {
        "data_cube":       data_cube,
        "range_profile":   range_profile,
        "range_axis":      range_axis,
        "target_bin":      tgt_bin,
        "sts_raw":         sts_raw,
        "dc_results":      dc_results,
        "clutter_results": clutter_results,
        "demod":           demod,
        "vs_result":       vs_result,
        "dc_scores":       dc_scores,
        "vs_scores":       vs_scores,
        "gt":              gt,
        "cfg":             cfg,
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)
