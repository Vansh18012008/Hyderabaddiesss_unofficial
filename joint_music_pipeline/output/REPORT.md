# Joint MUSIC with grouped candidate-person display

The underlying joint MUSIC maps are unchanged from the selected RaDICaL-style method. Grouping changes detections, association, and display only.

## Result

- 186 raw reflection peaks across 63 windows.
- 133 groups, including 40 groups formed from multiple peaks.
- Four group instances flagged ambiguous due to competing established identities.
- Nine confirmed group-track segments.
- Selected candidate shown in **17/63 windows**, with **at most one circle** in the main view.
- Three primary IDs across acquisition/reacquisition; these do not mean three people.

The single-person display uses the known scene count and a confirmed moving-group association. It is not achieved by merging all peaks in the scene or selecting a fresh global maximum each frame. It remains uncertain/lost when no qualified association exists. This conservative grouping does not solve full-recording person localization.

## Views

Open `replay.html` or `joint_music.gif` for the selected candidate-person view. Open `debug_replay.html` or `debug_groups.gif` to inspect all raw reflections and associated groups. The joint spectrum is visible in both views without hard masking. Bright flat maps do not establish a reliable location.

## Four grouping checks

Close separation (18 bins / 12 degrees), persistent spectral dips (4 dB twice, or 8 dB immediately), historical motion disagreement, and separate established track identities determine grouping. Complete-link grouping prevents chains of peaks spanning an overly large region. Group centers use measured range-energy weights. Unresolved shared peaks produce ambiguity rather than forced merging. Strong-turn handling permits changes of direction within the displacement bound. Thresholds are provisional and not calibrated body dimensions or false-alarm probabilities.

The nine behavior tests include connected grouping, weighted centers, chain prevention, persistent valleys, different motion, protected identities, unresolved shared peaks, turn handling, distant-jump rejection, and a single-person view with no fabricated lost position. They validate implementation behavior on constructed cases, not real multi-person accuracy.

See the package README for full pipeline details, reproduction commands, configuration, and limits. Physical range/time calibration and BR/HR remain unresolved. This single-person recording cannot establish reliable discrimination between closely spaced people.
