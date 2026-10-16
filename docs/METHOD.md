# Method

`arucotrack run` runs the pipeline below in eight steps. The paper describes the method in Section 3 and lists the
parameters of both datasets in Appendix A. This page names the code of each step and the configuration field that
sets each parameter ([CONFIG.md](CONFIG.md)). Defaults are given in brackets.

The steps are internal modules of the package `arucotrack`, grouped in five subpackages:

- `io`: video decoding, the trajectory files and the PeTrack layout;
- `detection`: the multipass detector and the detection cache;
- `refinement`: the corner refinement, its quality gate and the refined-corner cache;
- `geometry`: the marker pose, the marker orientation, the walking surfaces and the lens distortion;
- `calibration`: the calibration file and the camera model.

`run` (the command), `config` (the configuration), `pipeline` (the trajectory step) and `provenance` sit at the top
level.

25-Representation also used a second camera and a two-camera stage. That stage is not part of this repository.
Everything below is the single-camera method.

## 1. Decode and time

`io.video`. ffmpeg decodes the luma plane of every frame, in software or with NVDEC (`camera.decoder`
[auto], [ENVIRONMENT.md](ENVIRONMENT.md)). The frame number is the packet ordinal, and the time of a frame is its
packet's presentation timestamp (PTS). A dropped frame therefore cannot shift later times. The frame rate in the
headers is (number of packets - 1) / (PTS span). `camera.frames` limits a run to the first N frames. Frames are
decoded as stored: a video with rotation metadata is refused, and the calibration must describe the stored
orientation.

## 2. Detect

`detection.detectors.MultiPass`, driven by `detection.detect.run`. The OpenCV ArUco detector runs with its
sub-pixel corner refinement (`detect.corner_refinement` [subpix]) on the dictionary `marker.dictionary`
[DICT_ARUCO_MIP_36h12]. No single adaptive-threshold constant finds every marker. The detector therefore runs once
per constant (`detect.constants` [3, 6, 9]) and image scale (`detect.scales` [1, 2], where scale 2 is the frame
upscaled bicubically), with threshold windows up to `detect.win_size_max` [73] px in steps of `detect.win_size_step`
[4] px. Detections whose quadrilaterals overlap with an intersection over union above `detect.iou` [0.5] are one
marker. The corners of the first pass that found it are kept (scale 1 before 2, constants in order). A group with two
different ids is dropped. With `marker.ids` the dictionary is restricted to the listed ids, so other codes cannot
decode.

The detection cache is `<output.work>/<clip>.nvdec_y.<tag>.npz`. The name `nvdec_y` means "luma plane", whichever
decoder made it.

## 3. Refine the corners

`refinement.refine.refine`, driven by `refinement.refine_cache.run`. The detector's corners lie on the outer edge of
the black square. That is a single dark-to-light edge, and its bias shrinks the quadrilateral, which reads as a height
error. The corners are therefore re-estimated from the interior edges (typically 34-50) where the marker's cells
change colour. These come in equal numbers of each polarity, so a constant edge bias cancels.

- Each interior edge is located at `refine.samples_per_edge` [3] samples, inset 0.22 cell from its ends. Each sample
  is the centroid of the intensity gradient across a profile of half-width 0.42 cell (clipped to 1-4 px), sampled
  every 0.25 px. An edge with less than 12 grey levels of contrast is not used.
- A homography from cell coordinates to the undistorted image is refined by Gauss-Newton with point-to-line
  residuals and Cauchy weights (`refine.cauchy_c` [1.5]), in `refine.iterations` [3] outer iterations of 6 steps.
  The intensities are sampled in the image, and the geometry is solved in the ideal plane.
- The outer corners are the homography's image of the black square's corners, placed at a border of
  `marker.border_cells` data cells. The side of the square is M = (6 + 2b) c, with M = `marker.side_m` and c the
  width of one data cell.
- A refinement with fewer than 12 usable samples, or whose corners move by more than one cell, has not converged, and
  its detection is dropped. Each marker is refined in a crop of `refine.crop_px` [224] px.

The border constant b is an effective marker size, not a printed dimension. It combines the print with the edge
response of the optics and differs between cameras. Distances from the camera scale with 1 / (6 + 2b). How to set
it and what an error in it costs: [CALIBRATION.md](CALIBRATION.md), "Border constant".

The refinement needs cells wider than the optics blur. A cell is an eighth of the marker's apparent side, so a
marker 38 px across has cells under 5 px, against an edge width of about 3 px on the cameras of the paper. Below
that size neighbouring edges smear into one another, and the refinement is not expected to help.

Optional: `refine.blur_window` widens each edge profile by the motion blur `exposure_fraction x |v . n|` of a moving
marker, where `refine.sigma_optics` is the camera's blur at rest ("Values used in the paper" below).

## 4. Quality gate

`refinement.gate` (`gate`, `resolve_gated`). A refinement is accepted when three conditions hold. Its edge
root-mean-square residual is at most `gate.max_edge_rms_px` [0.8 px]. Its mean corner shift is at most
max(`gate.min_shift_px` [1.5 px], `gate.max_shift_frac` [0.07] x the diagonal of the detected quadrilateral). At most
`gate.max_downweighted_frac` [0.3] of its samples are unusable. `gate.mode` decides what happens to a refused
refinement:

- `resolve` [default]: the row is compared with a linear prediction from accepted neighbours within
  `gate.window_frames` [5] frames on both sides. Of the detector corners and the refused refinement, the one
  closer to the prediction is kept if it lies within `gate.agree_px` [3 px]. Otherwise the detection is dropped. A row
  without accepted neighbours on both sides is kept as unverified with the refused corners. Unverified rows stay in
  the caches and are left out of the trajectory files.
- `drop`: the detection is removed, so only accepted refinements remain.

The refined cache is `<output.work>/<clip>.nvdec_y.<tag>.<gate.mode>.refined.npz`.

## 5. Marker pose

`geometry.pose.derive`. The marker pose of each kept detection comes from OpenCV's IPPE solver for squares.
The marker centre of the solution with the smaller reprojection error is kept. The two solutions of the planar
ambiguity differ mainly in rotation, and their centres by a few millimetres. With a camera pose in the calibration
file the marker centre is in the world frame (z up). The floor position of a head is then the (x, y) of the marker
centre whatever the shape of the floor, and z is the marker height. Without a camera pose the marker centre is in
the camera frame ([CALIBRATION.md](CALIBRATION.md), "What the calibration must provide").

## 6. Floor and height gates

`pipeline` (`measured_rows`, `attach_floor`, `height_gate`). Measured rows are first limited to world
heights in `track.z_range_m` [-3.0, 4.5] m, before one row per frame and marker is chosen. The head point is then
dropped vertically onto the floor. The floor is the plane z = 0 (`floor.model: plane`) or a set of walking surfaces
(`floor.model: surfaces` with `floor.surfaces`, format in [FORMATS.md](FORMATS.md)), and fills the floor columns. A
row above a surface is kept only when its marker height above that surface lies in `track.height_band_m`
[0.3, 2.8] m. The same gate is
applied again to the smoothed rows. Without a camera pose there is no floor, and both gates are off.

## 7. Tracks: segments, smoother, interpolation

`pipeline` (`segments`, `smooth_cols`, `interp_cols`). Each marker id is one track, split at gaps longer
than `track.max_gap_frames` [30] frames. The smoothed file fits every segment with at least
`track.min_segment_rows` [5] measured rows, including the frames without a detection, by minimising

    E = sum over measured frames |p_t - y_t|^2 / sigma_s^2  +  sum over frames |p_{t-1} - 2 p_t + p_{t+1}|^2 / sigma_a^2

with a sparse linear solve. sigma_a is `track.sigma_acc_m` [0.45 mm per frame squared, each axis]. sigma_s is
1.4826 x the median absolute deviation of the measured centres' second differences / sqrt(6), pooled over the clip.
It is 2 mm when there are fewer than 20 triples of consecutive frames, and at least 0.1 mm. `track.sigma_s` [metres]
fixes it instead, for example to a noise measured once for a camera. A fixed value is written into the
`track:` header line, and the value used is in the sidecar. The interpolated file adds a linear fill of gaps of up to `track.interp_cap_frames` [30]
frames. Shorter segments stay in the raw file only.

## 8. Files and provenance

`io.trajwrite`, `io.petrack_export`, `pipeline.metric_step`. Per clip the run writes the
full layout (raw, interpolated, smoothed), the PeTrack layout of the smoothed rows with a sidecar of its `observed`
values, the marker orientations, a JSON sidecar and the resolved configuration ([FORMATS.md](FORMATS.md)). The
header of each full-layout file names the clip, the units and frame, the frame rate, the calibration file, the
marker, the track parameters and the code version. Each stage is skipped when its inputs are unchanged.

## Values used in the paper

The defaults of this package are the values of the paper (Appendix A, "Parameters of the tracker") except these,
which the paper sets per dataset:

| Field | Default | 26-Stairwell | 25-Representation |
|---|---|---|---|
| `marker.border_cells` | 0.869 (for an unmeasured print) | 0.855 | 0.855 (overhead camera), 0.892 (second camera) |
| `track.sigma_s` | estimated per clip | 2.3 mm | 1.0 mm (overhead), 1.4 mm (second camera), 0.2 mm (two-view) |
| `track.min_segment_rows` | 5 | 5 | 3 rows, or a track that reaches the door |
| `refine.blur_window` | off | off | on: `exposure_fraction` 1.00-1.11 measured per run and `sigma_optics` 1.84 px (overhead), 0.978 from the exposure metadata and 0.48 px (second camera) |

The 0.855 was set on the 25-Representation overhead camera against a reference of known scale. The 26-Stairwell
configurations in the dataset carry their values.
