# Selected joint MUSIC result

Input: `../data/RATHAN_WALKING_VIDEO.bin`.

63 windows of 32 observations, hop 16. Joint range–angle MUSIC with three-receiver/64-fast-time smoothing, forward–backward covariance averaging, and signal rank 6.

- Candidate pool entries: **186**.
- Windows with confirmed tracks: **34/63**.
- Confirmed track segments: **19**.
- Last 20 windows with a confirmed track: **1/20**.

`replay.html` and `joint_music.gif` show only this method. `maps.npz` contains only the joint spectrum and supporting range-energy/eigenvalue arrays. `tracks.csv` and `frame_states.json` contain measured candidates and tracking states.

This output matches the selected RaDICaL-method-on-walking run. Candidate tracks are not identified people; later maps can be flat or low-contrast. Physical timing and distance calibration remain unverified. No BR/HR is reported. See the package README for the full pipeline, thresholds, reproduction command, and limitations.
