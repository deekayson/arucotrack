# Calibration

ArUcoTrack does not calibrate cameras. It reads a calibration file (`camera.calibration`, format in
[CALIBRATION_FILE.md](CALIBRATION_FILE.md)) made with other tools, typically OpenCV. This page says what the file
must contain, how to obtain it, how to set the border constant, and how the 26-Stairwell calibration was made.

## What the calibration must provide

- Intrinsics of the camera at the resolution of the video: focal lengths, principal point, a distortion model and
  the image size. The model is OpenCV's pinhole model, the one `arucotrack run` uses, with 4, 5, 8, 12 or 14
  coefficients (12 and 14 only with the thin-prism and tilt terms zero, [CALIBRATION_FILE.md](CALIBRATION_FILE.md)).
  A fisheye (Kannala-Brandt) calibration is read and refused. The marker pose, and so every position and marker
  height, scales with the focal length. A 1 % focal-length error is about 1 % of the camera-to-marker distance in
  height.
- The distortion model must hold out to the edge of the image. For a camera with a wide field of view (for example
  more than 120 degrees across) choose a model whose distortion is still valid at the image corners. ArUcoTrack
  does not check that a calibration is plausible. It uses the model as given.
- Camera pose (optional) in a world frame with z up, the floor at z = 0, in metres, as `cam_from_world`
  (the convention of OpenCV's `solvePnP`, X_cam = R X_world + t). With it the trajectories are world positions and
  the floor model works. On a flat floor at z = 0, z is the marker height above the floor. Over walking surfaces
  the marker height above the floor is z - z_floor.

Without a camera pose the trajectories are in the camera frame: x right, y down, z along the optical axis, metres,
origin at the camera. x and y then lie in the image plane of a tilted, y-down camera frame (a mirror image of a
floor plan seen from above), not on the floor. The default `floor.model: auto` switches the floor and both height
gates off by itself. With an explicit floor model, `plane` and `surfaces` are refused, and `none` needs the gates
set to null (`track.z_range_m=null track.height_band_m=null`). The headers and the sidecar name the frame.

Two numbers outside the calibration file also set the scale: the printed marker side (`marker.side_m`) and the
border constant (`marker.border_cells`, [METHOD.md](METHOD.md), step 3). The border constant is set for the camera
and print in use ("Border constant" below).

## How to obtain it with OpenCV

1. Intrinsics. Record a planar calibration target (for example a 9 x 6 chessboard pattern of measured pitch on a
   rigid, flat panel) with the camera at the resolution and settings of the recordings. Use many poses that cover the
   whole image, including its corners and edges, tilted up to about 40 degrees. Detect the corners
   (`cv2.findChessboardCorners` and `cv2.cornerSubPix`) and fit with `cv2.calibrateCamera`. Keep the frames. The distortion model is only trustworthy out to the image
   radius the target reached.
2. Choose the calibration by a held-out height. Place the calibration target, or a marker, at several measured
   heights in view of the recording camera, and keep these placements out of every fit. The calibration whose
   heights match best is the one to use. A low reprojection error on frames held out of the fit is a necessary check
   but not enough: it does not reveal a focal-length error, and the marker height scales with the focal length
   (the paper, Appendix B).
3. Camera pose. Survey at least six points on the floor, spread over the area and over the image, with
   coordinates in metres in your world frame (z = 0 on the floor). Click or detect their image positions in a frame
   from the recording camera, which must not have moved since. Solve the pose with `cv2.solvePnP`, using the
   intrinsics and distortion from step 1 (`cv2.SOLVEPNP_SQPNP` or `cv2.SOLVEPNP_ITERATIVE`). Floor points alone fix
   the camera pose but not the focal length, so keep the focal length from step 1.
4. Write the file by hand, as [CALIBRATION_FILE.md](CALIBRATION_FILE.md) describes, or dump the results of
   steps 1 and 3 into it with a few lines of Python. `K` and `dist` come from `cv2.calibrateCamera`, `rvec` and
   `tvec` from `cv2.solvePnP` on the floor points. The per-view `rvecs` and `tvecs` of `calibrateCamera` are poses of
   the planar calibration target, not of the camera, and do not belong in the file:

       import json

       import numpy as np

       width, height = 2592, 1944                       # the image size of the video (frame.shape is height, width)
       doc = {
           "camera_id": "cam1",
           "model": "FULL_OPENCV",
           "image_size": [int(width), int(height)],
           "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                          "dist": np.asarray(dist, float).ravel().tolist()},
           "extrinsics": {"rvec": np.asarray(rvec, float).ravel().tolist(),
                          "tvec": np.asarray(tvec, float).ravel().tolist()},
       }
       with open("calibration.json", "w") as f:
           json.dump(doc, f, indent=2)

   Leave out `extrinsics` for a camera without a pose. `arucotrack run` checks the file when it loads it and names
   every problem.

## Border constant

The refinement fits the grid lines between the marker's data cells. It finds the outer corners by extending the
fitted grid outwards by b cells, where b is `marker.border_cells`. The side of the black square is M = (n + 2b) c,
with n = 6 data cells, c the width of one data cell and M = `marker.side_m`. On paper the border is one cell wide.
The print and the edge response of the optics make the fitted grid end slightly off the visible edge, so b is not 1
and differs between cameras. It is a scale constant of one camera, one marker print and one calibration, and it also
absorbs a small focal-length error. Set it once against a reference of known scale:

1. Place one marker with a known id at a known position X_true (the centre of its black square) in the world frame,
   in view of the camera. It can lie on the floor or on a surveyed stand. Record a short clip in which the marker
   does not move.
2. Run `arucotrack run` on the clip with the current value b0 in `marker.border_cells` (the default 0.869 the first
   time). A marker lying on the floor is below the marker-height gate, so switch the gate off for this run with
   `track.height_band_m=null` on the command line.
3. In `<clip>.fused.txt` ([FORMATS.md](FORMATS.md)), take the rows with the reference marker's id in column `id` and
   `observed` 1. The median of the columns `x/m`, `y/m`, `z/m` over these rows is X_meas. The camera centre C is on
   the header line `# camera centre: (x, y, z) m`. For marker id 7:

       python - out/clip.fused.txt 7 <<'EOF'
       import re, sys, numpy as np
       lines = open(sys.argv[1]).read().splitlines()
       head = next(l for l in lines if l.startswith("# camera centre:"))
       print("C", re.search(r"\(([^)]*)\)", head).group(1))
       rows = [l.split() for l in lines if not l.startswith("#")]
       col = {name: k for k, name in enumerate(rows[0])}
       sel = [r for r in rows[1:] if r[col["id"]] == sys.argv[2] and r[col["observed"]] == "1"]
       xyz = [[float(r[col[c]]) for c in ("x/m", "y/m", "z/m")] for r in sel]
       print("X_meas", np.median(xyz, axis=0), len(sel), "rows")
       EOF

4. Compute d_meas = |C - X_meas| and d_true = |C - X_true|, and set

       b1 = ((6 + 2 b0) * d_meas / d_true - 6) / 2

   in `marker.border_cells`. One step is enough, because the distance from the camera scales as 1 / (6 + 2b).

The sign follows from the geometry. A b that is too large makes the marker look larger in the image, hence closer
to the camera. A marker that reads too close (d_meas < d_true) therefore needs a smaller b. Without a camera pose
the positions are in the camera frame, and the header's camera centre is (0, 0, 0). Then d_meas is the norm of the
position and d_true the measured distance from the camera to the marker centre.

How much it matters. An error db in b scales every distance from the camera by 2 db / (6 + 2b). The 0.13 cells
between the default 0.869 and the nominal 1 are 3.4 %. Measured on one 26-Stairwell clip with the camera about 3 m
above the floor, the default 0.869 instead of the value set for that camera (0.8553) moves the positions by a median
of 3.5 mm horizontally and 4.9 mm vertically. The nominal 1.0 moves them by 36 mm and 50 mm.

Without a reference of known scale, keep the default. On the 26-Stairwell camera it cost 4.9 mm of height (the
median above). On another print or camera the error is unknown, and the formula above gives what an error in b
does to the positions.

## Checklist

- [ ] The camera settings (resolution, zoom, focus, field-of-view mode, stabilisation) are the same in the
      calibration recording and the experiment. A consumer camera can change its field of view without recording it.
- [ ] The calibration target is rigid and flat, and its pitch is measured, not taken as nominal.
- [ ] The frames for the intrinsics cover the whole image, including the corners, at several tilts.
- [ ] The reprojection error on frames held out of the fit is low (a necessary check, not the selection).
- [ ] The camera did not move between the frame used for the camera pose and the experiment, or a camera pose is solved per recording.
- [ ] The floor points are surveyed, spread over the walking area, and at least six.
- [ ] The acceptance test: an object at a known height (the calibration target on a step, a marker on a stand),
      not used in any fit, reads at its height through the calibration. For a marker, run `arucotrack run` on a
      short clip and read its `z`.
- [ ] `marker.side_m` is the measured side of the black square, and `marker.border_cells` was set for this camera,
      print and calibration against a marker at a known position ("Border constant" above).

## How the 26-Stairwell calibration was made

The paper describes it in Appendix B. The distortion comes from a fit with OpenCV's rational model to images of a
planar calibration target (a 9 x 6 chessboard pattern, 59.44 mm pitch) taken in the laboratory with the same
camera. The focal length, the principal point and the camera pose were solved
with COLMAP on one image of the empty stairwell, registered to a photogrammetric model of the stairwell. The model
was scaled by tape-measured distances between printed fiducials on the floor. The candidate calibrations were
compared by the height of the calibration target placed 15 times at known heights on a second visit and held out of
every fit. The same COLMAP registration on an image of that visit read the target 3.9 mm too low on average (SD
3.8 mm). The final calibration itself, solved on an image of the experiment day, was not tested on held-out targets.
The `calibration.json` of the dataset records its own provenance.
