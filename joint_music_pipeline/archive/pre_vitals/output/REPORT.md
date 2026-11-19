# Joint MUSIC and one candidate location per frame

The selected joint MUSIC spectrum is unchanged. There are 63 overlapping windows. The main output now supplies **exactly one measured candidate location in every frame**, including weak frames. Candidate support is labeled per frame; no position is propagated as though it were a current measurement.

## Frame-wise selection

1. Prefer the known single-person candidate group when the group tracker supports it.
2. If the frame has unresolved competing group identities, report its best group as AMBIGUOUS.
3. If there is no supported group, use the strongest MUSIC maximum inside the configured in-range, interior-angle scan. Mark it `no_supported_peak` and LOW confidence.

The last rule makes the output complete for frame review. It does **not** mean the walking person was detected in a low-support frame.

For every selected location, the output reports map peak-to-median contrast, selected MUSIC level, local prominence above its neighboring ring, range-energy excess over that frame's median, and whether its range bin is near a scan edge. These per-frame measurements assign HIGH, MEDIUM, LOW, or AMBIGUOUS using fixed thresholds. The classes are not calibrated probabilities. Thresholding a flat map cannot create target evidence, so fallback coordinates must be interpreted cautiously.

## Result

- 63 candidate rows and 63 visible main-view markers.
- Confidence categories: **23 HIGH, 18 MEDIUM, 20 LOW, 2 AMBIGUOUS**.
- 18 frames use the explicit spectrum fallback.
- 17/63 frames have a confirmed grouped person candidate; other markers show the best available group or in-range MUSIC maximum, not a confirmed person.
- 186 raw peaks became 133 groups; 40 groups combine multiple peaks.

Frames 48–62 have generally flat or weak MUSIC maps. The method still places a LOW marker at the best scanned point; it cannot establish that this is the person. Such a point may sit near the scan boundary and be unstable between frames. Use `candidate_locations.csv` to see the evidence and range-edge flag. No accuracy claim follows from having a row in every frame.

## Files

- `replay.html`: one candidate marker and confidence per frame.
- `debug_replay.html`: all raw reflections/groups plus the frame candidate.
- `candidate_locations.csv`: exactly 63 frame-indexed coordinates and evidence metrics.
- `frame_states.json`: selected candidate and complete group decisions per frame.
- `reflection_states.json`: reflection-level detections before grouping.
- `maps.npz`: unchanged joint MUSIC maps and supporting arrays.
- `joint_music.gif`, `debug_groups.gif`, and the `frames/` / `debug_frames/` directories: rendered output.

The thresholds and grouping distances remain provisional because physical timing/range calibration is unresolved. Candidate circles are not verified identities. No BR/HR is produced.
