# The calibration file

One JSON object, written by hand or by a few lines of Python from OpenCV results ([CALIBRATION.md](CALIBRATION.md),
"How to obtain it with OpenCV"). `arucotrack run` checks it on load: a file with a problem is refused with every
problem named.

## Example

    {
      "camera_id": "cam1",
      "model": "FULL_OPENCV",
      "image_size": [2592, 1944],
      "intrinsics": {
        "fx": 638.0, "fy": 638.0, "cx": 1296.9, "cy": 1035.4,
        "dist": [-0.278, 0.185, 0.00017, -0.00057, -0.0067]
      },
      "extrinsics": {
        "rvec": [-3.1373, 0.0545, -0.0001],
        "tvec": [0.0013, -1.3312, 3.1069]
      }
    }

## Fields

| Field | Required | Content |
|---|---|---|
| `model` | yes | the pinhole model `arucotrack run` uses: `FULL_OPENCV` (also accepted: `OPENCV`, `pinhole`, `k1k2p1p2k3`, `k1k2`, `k1k2p1p2`, `rational`). A fisheye model (`OPENCV_FISHEYE`, `fisheye`) is read and refused |
| `image_size` | yes | `[width, height]`, integers in pixels, of the video the trajectories come from (`arucotrack run` refuses a video of another size) |
| `intrinsics.fx`, `fy`, `cx`, `cy` | yes, or `K` | focal lengths and principal point in pixels (OpenCV pixel-centre convention) |
| `intrinsics.K` | instead of the four above | the 3 x 3 camera matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` |
| `intrinsics.dist` | yes | pinhole: OpenCV order `k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [tx ty]]]]` (4, 5, 8, 12 or 14 values). The thin-prism terms s1..s4 and the tilt terms tx, ty must be zero: the package does not invert them |
| `extrinsics.rvec`, `extrinsics.tvec` | no | the camera pose, `cam_from_world`: X_cam = R(rvec) X_world + tvec, as returned by `cv2.solvePnP` |
| `extrinsics.R`, `extrinsics.t` | instead of rvec / tvec | the same camera pose as a 3 x 3 rotation matrix and a translation |
| `camera_id` | no | recorded in the sidecar of each clip (the configuration's `camera.name` overrides it) |
| anything else | no | kept, ignored (for example provenance or a note on how the file was made) |

World frame: z points up, the floor is the plane z = 0, units are metres. x and y are free: choose them to suit
the analysis (for example x across a door, y through it). With a floor that is not flat, the walking-surfaces JSON
(`floor.surfaces`) must be in the same frame.

Without extrinsics (`"extrinsics": null` or absent) the trajectories are written in the camera frame
([CALIBRATION.md](CALIBRATION.md), "What the calibration must provide"). The default `floor.model: auto` turns the
floor and both height gates off. With an explicit floor model, `plane` and `surfaces` are refused, and so is a gate
left on with `none`: set `floor.model=none track.z_range_m=null track.height_band_m=null`.

## Checks on load

`image_size` has two integers from 1 to 65536. `fx` and `fy` are positive, and K's last row is `[0, 0, 1]`. `dist`
has a length allowed for the model, and its coefficients 9 to 14 are zero. `R` is a rotation (orthonormal, determinant +1). `rvec` and `tvec` have three finite
values. `model` is given and is one of the names above (a file without it is refused: the model is never guessed).
