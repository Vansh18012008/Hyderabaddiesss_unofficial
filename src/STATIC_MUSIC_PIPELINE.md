# Stationary multi-person candidate pipeline

`static_music_pipeline.py` implements the proposed path using the existing range-FFT + per-range-bin angle MUSIC. It is callable from Python and executable from the command line. Each candidate receives its own complex slow-time signal and existing circle-fit, phase demodulation, filtering and vital-rate estimation. Candidate IDs are local to one capture, not persistent person identities.

The localization path preserves stationary returns: no slow-time mean subtraction or rank-1 removal. Existing spatial smoothing and forward/backward averaging inside MUSIC are reused. No new background subtraction, calibration, tracking, or human classifier is added.

## Usage

From this `src` directory:

```bash
python static_music_pipeline.py --input capture.npz --config radar.json --method music --output results/music
python static_music_pipeline.py --input capture.npz --config radar.json --method old --output results/old
python static_music_pipeline.py --input capture.npz --config radar.json --method cbf_static --output results/cbf_static
```

NPZ must contain complex `cube` in `(RX, observations, ADC samples)` order and optionally `timestamps_s` with one timestamp per observation. For an existing DCA1000 BIN, provide `--observations N`; the file length is checked against the explicit configuration. Arbitrary BIN layouts are not inferred.

JSON fields: sample_rate_hz, slope_hz_s, carrier_hz, antenna_spacing_m, n_samples. Optional fields: num_rx (4), nfft (512), angular_sign (-1), min_range_m (.4), max_range_m (6.9), angle_min_deg (-60), angle_max_deg (60), angle_step_deg (2.5), slow_fs_hz (null), min_vital_duration_s (20).

Example values must come from the actual capture configuration. The benchmark's synthetic_config.json is supplied for the generated synthetic NPZ files, not for unrelated recordings. The half-FFT range interval is retained to match the legacy engine; the requested range region must fit that interval.

## Outputs and limitations

- result.json: candidate ranges/angles, scores, vital estimates or a reason estimates were skipped, processing duration.
- signals.npz: RA map, axes, separate complex candidate signals, displacement and filtered respiratory/cardiac traces where timing permits.
- The common comparison detector uses local maxima, -12 dB relative threshold, maximum six candidates and fixed spatial suppression. It is not a validated human detector or calibrated CFAR.
- `old` reuses the original mean subtraction, FFT, CBF and rank-1 algorithm; candidate detection is shared with the new path for comparison. Old vital extraction is extended to every candidate rather than its former primary-only output.
- Vitals require known regular sampling >5 Hz and at least 20 seconds. This is an execution guard, not physiological validation. Missing vital evidence never classifies a reflector as non-human.
- MUSIC localization does not improve CBF's extraction aperture. Two peaks can still produce contaminated vital signals when conventional beams overlap.
- Signal-count estimation can be unstable in noiseless or poorly conditioned cases. MUSIC can produce false peaks and confuse multipath with direct targets.
- Existing algorithm modules are loaded from this src directory. The vitals module is evaluated with deferred annotations to tolerate its existing missing Tuple type import on older Python versions; its numeric processing is unchanged.

The web dashboard is not switched automatically. Import `RadarConfig` and `StaticMusicPipeline`, then call `process_cube` or `process_bin` to use this implementation from an application.
