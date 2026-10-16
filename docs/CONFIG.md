# Configuration

Generated from `src/arucotrack/config.py`. Do not edit by hand.

One configuration for a run: the schema, its defaults, loading, overrides and checks.

Run with `arucotrack run config.yaml [key=value ...]`. Overrides use dotted paths, for example `gate.mode=drop camera.frames=120`. The resolved configuration is written next to the output files (`<stem>.config.yaml`).

## Top level

| key | default | meaning |
|---|---|---|
| `video` | required | Video file(s), relative to `data_root`. |
| `data_root` | null | Base of relative input paths (default: the config file's directory, `--data-root` overrides). |

## `marker`

| key | default | meaning |
|---|---|---|
| `marker.dictionary` | `DICT_ARUCO_MIP_36h12` | OpenCV predefined ArUco dictionary, by its `cv2.aruco` name. The refinement models a 6x6 bit grid inside a one-cell border, so only 6x6 dictionaries are accepted. |
| `marker.side_m` | required | Side of the printed black square, metres (required: measure your print). Sets the scale of every position. |
| `marker.border_cells` | `0.869` | Printed border width in data-cell widths, nominally 1. The default is for an unmeasured print: set the value for your print and camera against a marker at a known position (docs/CALIBRATION.md, "Border constant"). |
| `marker.ids` | null | Optional roster: the marker ids in use. The dictionary is restricted to them, so other codes cannot decode and more bit errors can be corrected without ambiguity. |

## `camera`

| key | default | meaning |
|---|---|---|
| `camera.calibration` | required | Calibration JSON (`docs/CALIBRATION_FILE.md`): intrinsics and an optional camera pose. Without a camera pose the positions are in the camera frame, and `floor.model: auto` switches the floor and both height gates off. |
| `camera.name` | empty | Camera id recorded in the sidecar (default: the calibration file's `camera_id`). |
| `camera.decoder` | `auto` | Video decoder: `auto` (NVDEC when a GPU and ffmpeg's CUVID decoder are present, else software), `nvdec`, `sw`. |
| `camera.frames` | null | Process only the first N frames (a quick check), null = the whole video. |

## `floor`

| key | default | meaning |
|---|---|---|
| `floor.model` | `auto` | `auto`: `plane` with a camera pose, else `none` with both height gates off. `plane` (the plane z = 0), `surfaces` (the walking surfaces of `floor.surfaces`) and `none` (no floor columns) can be set, and `plane` and `surfaces` need a camera pose. |
| `floor.surfaces` | null | Walking-surfaces JSON, for `floor.model: surfaces` only (`docs/FORMATS.md`, "Walking surfaces"). |

## `detect`

| key | default | meaning |
|---|---|---|
| `detect.constants` | `[3, 6, 9]` | Adaptive-threshold constants of the multipass detector. Every pass runs and the passes are merged. |
| `detect.scales` | `[1.0, 2.0]` | Image scales of the multipass detector (2 = the frame upscaled 2x, bicubic). |
| `detect.win_size_max` | `73` | Adaptive-threshold window maximum, px (`adaptiveThreshWinSizeMax`). |
| `detect.win_size_step` | `4` | Adaptive-threshold window step, px (`adaptiveThreshWinSizeStep`). |
| `detect.pixels_per_cell` | `8` | Perspective removal pixels per cell (`perspectiveRemovePixelPerCell`). |
| `detect.corner_refinement` | `subpix` | OpenCV corner refinement of each pass: `subpix`, `none`, `contour` or `apriltag`. |
| `detect.iou` | `0.5` | Intersection over union above which detections of two passes are one marker. |
| `detect.periphery_rule` | `any_corner` | *fixed*. Not used by `arucotrack run` (pinhole cameras only). |
| `detect.theta_max_deg` | `65.0` | *fixed*. Not used by `arucotrack run` (pinhole cameras only). |
| `detect.threads` | `0` | OpenCV threads during detection, 0 = all usable cores, the cores this process may run on (on a shared machine set a number). Detections are byte-identical for any value. |
| `detect.patch` | `0` | Marker image patches stored in the detection cache, px (0 = none). |

## `refine`

| key | default | meaning |
|---|---|---|
| `refine.samples_per_edge` | `3` | Samples per interior edge. |
| `refine.inset_cells` | `0.22` | *fixed*. Inset of the samples from the edge ends, cells. |
| `refine.half_window` | `0.42 cell, clipped to 1-4 px` | *fixed*. Half-width of each edge profile: 0.42 of a cell, clipped to 1-4 px. |
| `refine.profile_step_px` | `0.25` | *fixed*. Edge profile sample spacing, px. |
| `refine.min_contrast` | `12.0` | *fixed*. Minimum edge contrast, grey levels. |
| `refine.cauchy_c` | `1.5` | Cauchy constant of the robust line fit, in units of the robust scale. |
| `refine.gauss_newton_steps` | `6` | *fixed*. Gauss-Newton steps per outer iteration. |
| `refine.iterations` | `3` | Outer iterations (re-sample the edges, re-fit the homography). |
| `refine.min_samples` | `12` | *fixed*. Minimum usable edge samples. With fewer, the refinement fails. |
| `refine.max_move_cells` | `1.0` | *fixed*. A refinement that moves the corners by more than this (cells) has diverged. |
| `refine.crop_px` | `224` | Square crop around each marker in which the refinement works, px. |
| `refine.blur_window` | `False` | Blur-adaptive window: widen each edge profile by the motion blur `exposure_fraction x \|v . n\|` of a moving marker. Off by default (the values used in the paper: `docs/METHOD.md`). |
| `refine.exposure_fraction` | null | Effective exposure fraction for the blur window (null = fitted from the video's fast markers). |
| `refine.sigma_optics` | null | At-rest optics blur, px (required with `blur_window`). |
| `refine.velocity_window_frames` | `2.5` | *fixed*. Neighbouring detections within this many frame intervals give a marker's image velocity. |

## `gate`

| key | default | meaning |
|---|---|---|
| `gate.mode` | `resolve` | What happens to a refused refinement. `resolve`: the detector corners or the refused refinement are used when they agree with accepted neighbours on both sides, else the row is left out of the trajectory files. `drop`: the row is removed. |
| `gate.max_edge_rms_px` | `0.8` | Largest edge root-mean-square residual of an accepted refinement, px. |
| `gate.min_shift_px` | `1.5` | Lower bound of the corner shift limit: an accepted refinement moves the corners by at most max(min_shift_px, max_shift_frac x diagonal), px. |
| `gate.max_shift_frac` | `0.07` | Corner shift limit as a fraction of the detector quad's diagonal. |
| `gate.max_downweighted_frac` | `0.3` | Largest share of unusable or far edge samples. |
| `gate.agree_px` | `3.0` | `resolve`: largest corner distance from the position predicted by the accepted neighbours, px. |
| `gate.window_frames` | `5.0` | `resolve`: the accepted neighbours must lie within this many frames on both sides. |

## `track`

| key | default | meaning |
|---|---|---|
| `track.z_range_m` | `[-3.0, 4.5]` | World height gate on the marker centre [min, max], metres, applied before the per-frame de-duplication (null = off). Needs a camera pose. |
| `track.height_band_m` | `[0.3, 2.8]` | Marker height above the floor surface under it [min, max], metres, on rows that have a surface (null = off). Needs a camera pose and a floor. |
| `track.max_gap_frames` | `30` | Largest gap inside a segment, frames. A longer gap splits the track. |
| `track.min_segment_rows` | `5` | Shortest segment kept in the smoothed files, measured rows. |
| `track.sigma_acc_m` | `0.00045` | Acceleration prior of the smoother, metres per frame squared (each axis). The default is the RMS head acceleration of walking participants below 6 Hz (the paper, Section 3.6). |
| `track.sigma_s` | `1.4826 MAD of second differences / sqrt(6), pooled; 2 mm below 20 triples; floor 0.1 mm` | Measurement noise of the smoother, metres (each axis), for example `0.0023`, one value for every measured row of the clip. The default text estimates it per clip as it says. |
| `track.interp_cap_frames` | `30` | Largest gap filled by linear interpolation, frames. |

## `output`

| key | default | meaning |
|---|---|---|
| `output.dir` | `out` | Trajectory files, relative to the current directory (`--out` overrides it). |
| `output.work` | `work` | Caches of detection, refined corners and poses, relative to the current directory (`--work` overrides it). |
| `output.dataset` | `dataset` | Dataset label, kept in the resolved configuration (`<stem>.config.yaml`). |
| `output.variants` | `['fused', 'fused_interp', 'fused_smooth', 'petrack']` | *fixed*. Files written per video: measured rows, plus linear fill, smoothed, PeTrack six-column. |
