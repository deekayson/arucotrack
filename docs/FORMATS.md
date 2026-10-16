# Output formats

`arucotrack run` writes, per video `<clip>`, into `output.dir`:

| File | Content |
|---|---|
| `<clip>.fused.txt` | measured rows (one per frame and marker id), full layout |
| `<clip>.fused_interp.txt` | measured rows plus linear fill of gaps of at most `track.interp_cap_frames` frames |
| `<clip>.fused_smooth.txt` | the point smoother's output per segment, every frame of the segment |
| `<clip>.petrack.txt` | the smoothed rows in the PeTrack layout (what PedPy reads) |
| `<clip>.petrack.observed.json` | the `observed` value of every row of the PeTrack file |
| `<clip>.orient.npz` | marker orientation per measured row (for viewers) |
| `<clip>.sidecar.json` | parameters, counts, what each gate removed, dropped segments, sigma_s, frame rate |
| `<clip>.config.yaml` | the fully resolved configuration of the run (schema defaults, the file, every override) |

and caches into `output.work` (detection, refined corners, per-detection marker poses) with a `.run.json` stamp per
stage, which later runs reuse: a stage whose inputs, parameters and code are unchanged is skipped. Running again with
`<clip>.config.yaml` and the same `--data-root` repeats the run.

The file names and the columns follow the published trajectory files of the paper's datasets. One of them was made
with two cameras, which is why the names say "fused" and some columns name a camera. Here every row comes from one
camera.

## Full layout (`.fused*.txt`)

Plain text, one row per frame and marker id, values separated by single spaces, missing numbers `nan`, missing
strings `-`. Header lines start with `#` ("Header lines" below). The first line without `#` names the columns:

    id frame t_s x/m y/m z/m observed source miss_mm ray_angle_deg view_basler_deg corner_basler r_basler reproj_px flags markerID x_floor/m y_floor/m z_floor/m surface_id floor_src

| Column | Content |
|---|---|
| `id`, `markerID` | ArUco marker id (repeated in the `markerID` column) |
| `frame` | packet ordinal of the video (frames are counted, not computed from time) |
| `t_s` | the packet's presentation timestamp, seconds. Use it for time, not frame / frame rate |
| `x/m`, `y/m`, `z/m` | centre of the ArUco marker worn on the head, metres, in the world frame of the calibration (z up), or in the camera frame when the calibration has no camera pose (the header says which) |
| `observed` | 1 a measured frame, 0 a frame filled inside a gap by the linear fill or the smoother (not a measurement) |
| `source` | the same values as `observed` in a single-camera run (the column is kept from the published layout) |
| `miss_mm`, `ray_angle_deg` | two-camera quantities, always `nan` here |
| `view_basler_deg` | off-nadir angle: between the camera's ray to the marker centre and the downward vertical (world -z), degrees. Without a camera pose (camera-frame output) it is the angle between that ray and the optical axis (+z) |
| `corner_basler` | corners the row was posed from: `refined`, or, for a row the quality gate refused but both accepted neighbours confirmed, `detector` (the detector's corners) or `refused_refinement` (the refinement the gate refused). `-` on rows that are not measurements |
| `r_basler` | image radius of the marker centre as a fraction of the image corner radius |
| `reproj_px` | reprojection error of the per-detection marker pose, px |
| `flags` | `interp` on filled rows, otherwise `-` |
| `x_floor/m`, `y_floor/m`, `z_floor/m` | the point of the floor or walking surface under the head (`nan` without a floor) |
| `surface_id` | id of that surface (the plane is 0), -1 when there is none under the head |
| `floor_src` | 0 no surface, 1 a measured surface, 2 an extrapolated surface |

The columns `view_basler_deg`, `corner_basler` and `r_basler` refer to the one camera of the run, whatever its make.

The marker height above the floor is `z - z_floor`. The position is the centre of the marker, which sits on top of
the head: it is not the top of the head and not the centre of the body.

## PeTrack layout (`.petrack.txt`)

Columns `id frame x y z markerID`, metres. The header is five lines:

    # framerate: 60.0000 fps
    # metres, world frame of the calibration's camera pose (z up); x y z = the marker centre
    # id: the ArUco marker id (markerID repeats it), not PeTrack's sequential person index
    # frame: the frame index of the video, from 0
    # id frame x/m y/m z/m markerID

PedPy reads it with `pedpy.load_trajectory_from_txt`. Without a camera pose the second line names the camera frame.
The layout has no time column, and `frame / framerate` is off by the dropped frames when the camera drops any: use
`t_s` of the full layout for time. The layout carries no `observed` column either, so `.petrack.observed.json` lists
it per id, row by row (1 measured, 0 interpolated across a gap).

## Header lines

The `#` lines at the top of each `.fused*.txt` file, one fact per line:

    # arucotrack 0.1.0 trajectories | clip tilt | fused: measured rows, one per frame and marker
    # units: metres, in the world frame of the calibration's camera pose (z up, origin and axes as the pose defines them); x y z = the marker centre
    # frame rate: 60.0000 fps
    # frame: the frame index of the video, from 0; t_s: that frame's timestamp, seconds
    # calibration: calibration.json
    # camera centre: (1.1768, -0.2559, 3.1617) m
    # marker: side 0.1845 m, border constant 0.8553
    # track: z gate -3.0..4.5 m | height above the floor 0.3..2.8 m | max gap 30 frames | min segment 5 rows | sigma_acc 0.45 mm/frame^2 | sigma_s 2.3 mm | linear fill up to 30 frames
    # floor: the walking surfaces of ground_surfaces.json under the marker centre
    # source: 0 interpolated, 1 measured
    # observed: 1 measured, 0 interpolated (not a measurement)
    # floor_src: 0 no surface under the head (floor columns nan, surface_id -1), 1 measured surface, 2 extrapolated surface (a surface continued beyond the measured ones)
    # corner_basler: refined | detector | refused_refinement, the corners a measured row was posed from; flags: interp on rows filled linearly

The frame rate is (number of frames - 1) / (time of the last frame - time of the first). `sigma_s` is listed when
it is fixed (`track.sigma_s`); the value estimated per clip is in the sidecar. Without a camera pose the units line
names the camera frame and the gates are absent. The header lines differ between versions of the code; the data
lines are what a run computes:

    grep -v '^#' a/clip.fused.txt | sha256sum
    grep -v '^#' b/clip.fused.txt | sha256sum

## Orientation (`.orient.npz`)

Arrays `id`, `frame`, `quat_xyzw` (world-from-marker rotation, x y z w, where the marker frame has x right, y up,
z out of the face), `ippe_flip_rejected` (the other IPPE solution was taken because the chosen one flipped against the smoothed
orientation) and `corners_px` (the refined image corners).

## Sidecar (`.sidecar.json`)

The clip and camera, the coordinate frame (`world` or `camera`), the code version (`arucotrack 0.1.0`), the
configuration summary, the calibration (file name, focal length, camera centre, marker side), the parameters, the row
rule, the names of the detection and refined caches in `output.work`, the floor model and file name, the detection
counts, per id the measured and smoothed row counts and first and last frames, what each gate removed, the dropped
short segments, the measurement scale sigma_s and the frame rate. Files are named without their folder.

## Walking surfaces (input, `floor.surfaces`)

The floor under a head can be a set of planar polygons, for example the treads and landings of a stairway. The file
is one JSON object in the world frame of the calibration (metres, z up):

    {
      "frame": "world_m",
      "surfaces": [
        {"id": 0, "kind": "tread", "polygon_xy": [[1.41, -2.79], [2.95, -2.79], [2.95, -2.47], [1.41, -2.47]],
         "plane": [-0.0047, 0.0328, 0.9995, 1.2904]},
        {"id": 1, "kind": "landing", "polygon_xy": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [0.0, 1.5]],
         "plane": [0.0, 0.0, 1.0, 0.0], "source": "extrapolated"}
      ]
    }

| Field | Content |
|---|---|
| `frame` | must be `world_m` |
| `id` | integer written into `surface_id` |
| `kind` | a free label (for example `tread`, `landing`) |
| `polygon_xy` | the outline in the x-y plane, metres, at least three points |
| `plane` | `[a, b, c, d]` of the plane a x + b y + c z + d = 0, c not 0 |
| `source` | optional: `extrapolated` marks a surface that was continued rather than measured (`floor_src` 2) |

`arucotrack run` reads and checks the file before detection. It reads `frame`, and per surface `id`, `polygon_xy`,
`plane`, `source` and the optional `kind`, `z_mean`, `rms_mm` and `n_pts`. Other fields are ignored. A missing file,
another `frame`, a polygon with fewer than three points and a plane with c = 0 are refused, naming the surface id. A head point drops
vertically onto the plane of the polygon that contains its x-y position. Where polygons overlap, the highest plane
that is not above the head is used. A head outside every polygon gets no floor (`surface_id` -1).

How to obtain them: survey at least three points on each tread and landing in the world frame of the
calibration (more points give a better plane), fit a plane a x + b y + c z + d = 0 to them by least squares, and give
the outline of the tread or landing as `polygon_xy`. The points can also come from a photogrammetric model of the
site. `input/ground_surfaces.json` of the 26-Stairwell dataset is a worked example: planes fitted to a mesh of the
stairwell, with the extra fields of its fit.
