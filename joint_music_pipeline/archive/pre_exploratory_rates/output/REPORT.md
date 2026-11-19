# Joint MUSIC and contrast-gated candidate presence

The selected joint MUSIC spectrum is unchanged. There are 63 overlapping windows. Candidate markers now appear only when the per-frame MUSIC map has enough peak-to-background contrast. Empty/flat frames are left without a candidate rather than assigning their strongest pixel as a person location.

## Frame-wise selection

1. Prefer the known single-person candidate group when the group tracker supports it.
2. If the frame has unresolved competing group identities, report its best group as AMBIGUOUS.
3. If there is no supported group, inspect the strongest MUSIC maximum inside the configured in-range, interior-angle scan.
4. Suppress any candidate whose map peak-to-median contrast is below the configurable 15 dB cutoff.

Frames below the cutoff have no candidate circle. The gate is a visual presence heuristic; it does **not** classify a high-contrast return as a human.

For selected locations, the output reports map peak-to-median contrast, selected MUSIC level, local prominence above its neighboring ring, range-energy excess over that frame's median, and whether its range bin is near a scan edge. HIGH, MEDIUM, LOW, and AMBIGUOUS are support labels, not calibrated probabilities. The presence cutoff is the map maximum minus the median background; the map maximum itself is always normalized to 0 dB.

## Result

- 63 frame rows; candidate markers in **48/63** and no candidate in **15/63**.
- Frames 0–47 have 17.75–32.87 dB contrast; the final 15, visually empty frames, have 0.13–14.52 dB. The clip-calibrated 15 dB cutoff separates these sections in this recording.
- 17/63 frames have a confirmed grouped person candidate. A marker still indicates radar-map support, not proof that a human is present.
- 186 raw peaks became 133 groups; 40 groups combine multiple peaks.

The cutoff is specific to this recording and should be recalibrated on labeled occupied and empty-scene data before reuse. MUSIC contrast alone cannot distinguish a person from a strong static wall or other reflector; an empty-room reference, clutter model, and persistence/micro-motion checks would improve presence decisions.

## Files

- `replay.html`: candidate marker and confidence when the presence gate passes; no marker otherwise.
- `debug_replay.html`: raw reflections/groups plus a frame candidate when the gate passes.
- `candidate_locations.csv`: 63 frame-indexed rows, with coordinates blank when no candidate passes the gate.
- `frame_states.json`: selected candidate and complete group decisions per frame.
- `reflection_states.json`: reflection-level detections before grouping.
- `maps.npz`: unchanged joint MUSIC maps and supporting arrays.
- `joint_music.gif`, `debug_groups.gif`, and the `frames/` / `debug_frames/` directories: rendered output.

The thresholds and grouping distances remain provisional because physical timing/range calibration is unresolved. Candidate circles are not verified identities. No BR/HR is produced.


## BR/HR integration and result

The per-frame candidate IQ is beamformed and phase-unwrapped, then evaluated on rolling 20-second windows (400 samples at the unverified assumed 20 Hz). The window is advanced with the candidate frames. The DSP uses 0.1–0.5 Hz for breathing and 0.8–2.5 Hz for heart rate. A single 1.6-second localization aperture is not used as an independent vital-rate estimate.

This capture produced **zero reportable BR/HR values**: 24 output rows lack 20 seconds of history; the other 39 fail the tracked-signal coverage/continuity gate. This is because the walking candidate is not reliably associated with one stable range/angle group for a full rolling window, not because the estimator omitted all rows.

For diagnostics only, FFT peaks from the rejected candidate-phase windows had a respiration-band median of 10.84 (range 7.18–24.76) and a heart-band median of 61.96 (range 53.91–67.24). They are stored in `vital_estimates.csv` under `diagnostic_*`; they are not presented as breathing/heart rates because bulk walking motion, cell changes, and track switches can generate those frequencies. The frame replay labels BR/HR unavailable.

Slow-time frequency remains an assumed 20 Hz and is explicitly unverified. No motion compensation or reference physiology labels are available. The result is therefore that the pipeline evaluates BR/HR but does not claim a valid rate for this recording. See README for methods and output files.
