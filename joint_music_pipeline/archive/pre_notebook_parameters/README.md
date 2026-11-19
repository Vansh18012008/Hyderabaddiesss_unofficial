# Joint range–angle MUSIC pipeline

This is the selected RaDICaL-style joint MUSIC method, packaged with the walking recording and joint-only output. It is independent of the other project pipelines and uses relative paths, so this whole folder can be moved to another machine.

## Run

```bash
cd joint_music_pipeline
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/joint-music-mpl python3 run_pipeline.py
```

Open `output/replay.html` in a browser. The replay includes Play/Pause, Previous/Next, and a slider. `output/joint_music.gif` is the animated version. Only joint MUSIC is displayed; comparison methods are not included in this package.

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

## Underlying reflection-level result and limits

The selected run contains 186 candidate pool entries, 34/63 windows with confirmed tracks, and 19 confirmed track segments. Only 1 of the last 20 windows has a confirmed track. This package reproduces the selected result; it does not establish person-localization accuracy. Bright flat late-frame maps do not imply strong localized targets.

Physical distances and time calibration remain unresolved: the original code's 600 ksample/s ADC rate, 512 samples, and 100 μs ramp are inconsistent. Distances therefore remain in frequency/range bins. The original assumed 20 observations/s is not a verified capture timestamp rate. The half-FFT search interval is unchanged. No BR or HR is reported, and no moving-target phase compensation is implemented.

Existing historical data and experiments elsewhere in the parent project are left intact. This package's `output/` is the selected joint-only presentation.

## Grouped candidate-person output (current default)

The main replay now shows **one selected candidate-person circle at most**, using your confirmed single-person recording setting (`expected_people: 1` in `config.json`). It does not label every reflection as a person. Click **Show all reflection peaks and groups** to open `output/debug_replay.html`; white crosses are raw peaks and outlined circles are associated groups.

Grouping is implemented in `peak_grouping.py` and applied by `group_output.py`. The sequence is:

**Joint MUSIC → raw peaks → separation/valley/motion/identity grouping → range-energy-weighted group centers → group tracking → one frame candidate → rolling phase-based BR/HR analysis.**

Four evidence types are used:

1. **Separation:** peaks must be within 18 range bins and 12 degrees. Complete-link grouping requires every pair in a group to pass, so a chain of neighbors cannot merge an arbitrarily wide scene.
2. **Spectrum between peaks:** a dip of at least 4 dB, relative to the weaker endpoint, on two consecutive windows keeps peaks separate. A dip of at least 8 dB separates them immediately. Values are measured along the line connecting peaks in the joint spectrum.
3. **Motion/history:** peak histories are associated between windows. After at least three observations, velocity differences greater than 4 bins/window or 4 degrees/window keep peaks separate. Unknown history is not counted as evidence of co-motion. Group tracking requires strong confirmation and permits weaker continuation. A strong measurement within the displacement bound can support a turn when constant-velocity prediction is wrong; distant jumps are still rejected.
4. **Existing identities:** peaks assigned to different established group tracks are not merged. Ownership also constrains association. A single unresolved peak compatible with two established tracks is marked ambiguous and withheld from positional updates, rather than copied into two invented detections or declared one person.

Group centers use underlying measured FFT range power as weights, not MUSIC pseudospectrum height. For this single-person sequence, the primary group is acquired only after confirmation and at least eight bins of observed range span. It remains selected while active; missing detections do not cause an immediate switch to another group. Once retired, a separately qualified group can be reacquired under a new ID. This heuristic uses the known scene count; it is not a person classifier. It may miss people walking sideways or standing still.

**Current grouped result:** 186 raw peaks → 133 groups; 40 groups contain multiple peaks; four unresolved group instances are flagged ambiguous. There are nine confirmed group-track segments. The tracker confirms a group in 17/63 windows. The main view displays at most one circle and also applies the separate 15 dB presence gate. Three IDs occur across acquisition/reacquisition; they are not three people.

For a future multi-person experiment, set `expected_people` to `null` to expose all confirmed groups rather than apply the known single-person selection. Group IDs still are not verified person identities. Thresholds remain provisional in bins/degrees until physical calibration is resolved, and testing on this single-person recording does not validate close-person separation. Crossing/overlapping people may remain ambiguous. Do not report a person count from these groups alone.

`output/frame_states.json` contains every group, its members, ambiguity flags, pairwise decisions, and the selected candidate. `output/reflection_states.json` preserves the original reflection-level results. `output/tracks.csv` contains selected candidate positions. `output/grouping_settings.json` records thresholds. Run the nine grouping behavior checks with `python3 test_peak_grouping.py`.

## Contrast gated candidate presence

The main replay shows a candidate only when that frame’s joint MUSIC peak exceeds the configured map-background contrast in `config.json` (`candidate_presence_min_contrast_db`, currently 15 dB). Contrast is the map maximum minus the median over the scanned range-angle map. The absolute map maximum is normalized to 0 dB, so its displayed value alone cannot be used as an absolute-strength threshold. If the contrast gate fails, `candidate_available` is false and no circle is drawn; the strongest pixel is not passed off as a person location.

Each frame is scored on its own map using map peak-to-median contrast, the selected peak level, its prominence above a surrounding map ring, and selected-bin range energy above the frame's median range energy. The output labels HIGH, MEDIUM, LOW, or AMBIGUOUS using documented fixed cutoffs; they are support categories, not probabilities. The per-frame metrics and source are saved in the CSV and `frame_states.json`. We did not lower the MUSIC rank or normalize the output coordinate to force a target location.

For this recording, frames 0–47 have at least 17.75 dB map contrast; frames 48–62 have at most 14.52 dB. The 15 dB setting therefore leaves candidate markers in **48/63 windows** and marks the final **15/63 as no candidate**. This threshold was chosen from the observed separation in this clip, not validated on other recordings. A strong static reflector or a localized artifact can still produce high MUSIC contrast, so this is not a human classifier. The candidate CSV retains one row per frame and records the contrast and whether a candidate passed the gate.

The debug replay continues to show raw reflection peaks and groups; it overlays the frame candidate with a magenta × only when the presence gate passes. `output/candidate_locations.csv` includes the frame-wise contrast and gate result.

## BR/HR outputs

The pipeline now extracts a beamformed complex slow-time signal at the per-frame candidate, unwraps its phase, and evaluates respiration (0.1–0.5 Hz) and heart-rate (0.8–2.5 Hz) bands on rolling **20-second / 400-observation** windows. The rate table advances with the localization frames. A frame itself is only a 1.6-second overlapping aperture and cannot provide a reliable breathing/heart estimate by itself.

The quality-gated rates require at least 80% medium/high support from one continuous candidate group, limited range/angle motion, and a clear in-band peak. This recording passes none of those checks: **24 rows lack 20 seconds of history; the other 39 fail tracked-signal continuity.** The strict BR/HR columns remain blank.

To provide rough per-frame output, the pipeline now also shows **exploratory, unvalidated** spectral peaks after at least 10 seconds of candidate phase, using up to 20 seconds of history and bypassing the continuity and prominence gates. It produced both rates in 36 frames (12–47): BR median **11.2 bpm** (range **6.0–24.8**) and HR median **64.7 bpm** (range **58.0–66.2**). These values are especially unstable for breathing and may reflect walking motion or candidate-cell changes rather than physiology. They are labeled `exploratory_*` in the CSV and in orange on the replay; they are not quality-gated measurements.

The CSV retains `diagnostic_*` FFT peaks in rejected windows for debugging; those values must not be treated as rates. Frames before 12 do not have the minimum 10 seconds of phase history. The final frames have no candidate after the contrast gate and therefore show no exploratory rates.

Slow-time sampling is set to the legacy **20 Hz assumption**, but this recording has no verified observation timestamps. Walking-motion compensation is not implemented. No synchronized reference BR/HR is available, so these displayed exploratory values have not been validated.

Outputs: `output/vital_estimates.csv` (one row per frame), `output/vital_summary.json`, and `output/vital_phase.npz`. The full pipeline computes these automatically. To recompute from saved maps and raw data without rerunning MUSIC: `python3 run_vital_analysis.py`, then `python3 run_pipeline.py --render-only`.
