# Joint range–angle MUSIC pipeline

This is the selected RaDICaL-style joint MUSIC method, packaged with the walking recording and joint-only output. It is independent of the other project pipelines and uses relative paths, so this whole folder can be moved to another machine.

## Run

```bash
cd joint_music_pipeline
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/joint-music-mpl python3 run_pipeline.py
```

Open `output/replay.html` in a brot* args=(t*)malloc(T*sizeof(t));
wser. The replay includes Play/Pause, Previous/Next, and a slider. `output/joint_music.gif` is the animated version. Only joint MUSIC is displayed; comparison methods are not included in this package.

To process another compatible recording:

```bash
python3 run_pipeline.py --input /path/to/capture.bin --config config.json --output output_new
```

To regenerate the visuals without recomputing MUSIC:

```bash
python3 run_pipeline.py --render-only
```

## Included files

- `data/RATHAN_WALKING_VIDEO.bin`: exact copy of the supplied 8 MiB recording.
- `run_pipeline.py`: full decoding → preprocessing → joint MUSIC → detection → tracking → output runner.
- `joint_music.py`: the joint range/angle implementation used in the RaDICaL experiment.
- `adc_io.py`: existing int16 complex-IQ decoder.
- `walking_confidence_tracker.py`: peak detection, strong-start/weaker-continuation association, and behavior checks.
- `config.json`: radar and processing settings used for this result.
- `requirements.txt`: NumPy, SciPy, Matplotlib, and Pillow requirements.
- `output/`: joint-only maps, tracks, reports, all 63 frame images, GIF, and interactive replay.
- `validation.json`: reproduction and data-integrity checks.

## Pipeline

**BIN → complex cube → overlapping windows → within-chirp DC removal → slow-time mean subtraction → joint spatial/fast-time smoothing → forward–backward covariance averaging → joint range–angle MUSIC → candidate confidence → tracking → maps/replay.**

The range FFT is used only for the detector's auxiliary range-energy check, not to generate the joint MUSIC spectrum. The 512-sample ADC input becomes a 3-receiver × 64-fast-time-sample smoothing aperture; covariance dimension is 192. Every observation is used, fast-time starts advance by 8 samples, and the signal rank is 6. Rank is a subspace assumption, not a person count.

For this recording: 4 receivers × 1,024 observations × 512 complex samples; 32-observation windows with hop 16 give 63 windows. The scan covers equivalent range bins 0–255 and nominal angles −60° to +60° in 1° steps. The displayed spectrum is ungated, with per-window −30 to 0 dB normalization.

Tracking starts from strong detections and needs three consecutive strong matches for confirmation. Weaker nearby measurements can continue confirmed tracks, with at most five successive weak updates. Missing measurements are not fabricated; a lost track is retired after three misses. Green markers mean strong confirmed support, amber means weaker continuation, and gray means tentative. IDs are reflector tracks, not human identities.

## Current result and limits

The selected run contains 186 candidate pool entries, 34/63 windows with confirmed tracks, and 19 confirmed track segments. Only 1 of the last 20 windows has a confirmed track. This package reproduces the selected result; it does not establish person-localization accuracy. Bright flat late-frame maps do not imply strong localized targets.

Physical distances and time calibration remain unresolved: the original code's 600 ksample/s ADC rate, 512 samples, and 100 μs ramp are inconsistent. Distances therefore remain in frequency/range bins. The original assumed 20 observations/s is not a verified capture timestamp rate. The half-FFT search interval is unchanged. No BR or HR is reported, and no moving-target phase compensation is implemented.

Existing historical data and experiments elsewhere in the parent project are left intact. This package's `output/` is the selected joint-only presentation.
