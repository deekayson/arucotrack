# Third-party software

ArUcoTrack's own code is under the licence in `LICENSE`. It uses the software below, which is not part of this
repository and keeps its own licence. Versions are those locked in `uv.lock`. The licences were read from the
metadata of the installed packages and, for ffmpeg, from `/usr/share/doc/ffmpeg/copyright` in the Docker image.

## Runtime dependencies (installed by `uv sync`)

| Package | Version | Licence | Used for |
|---|---|---|---|
| opencv-python-headless | 5.0.0.93 | Apache-2.0. The wheel bundles third-party libraries, including an LGPL-2.1 build of FFmpeg (see its `LICENSE-3RD-PARTY.txt`) | ArUco detection, IPPE marker pose, image operations |
| numpy | 2.5.3 | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | arrays |
| scipy | 1.18.1 | BSD-3-Clause | the sparse smoother |
| omegaconf | 2.3.1 | BSD-3-Clause | the configuration schema and overrides |
| antlr4-python3-runtime | 4.9.3 | BSD-3-Clause | required by omegaconf |
| PyYAML | 6.0.3 | MIT | YAML |
| loguru | 0.7.3 | MIT | logging |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | progress bars |
| colorama, win32-setctime | 0.4.6, 1.2.0 | BSD-3-Clause, MIT | Windows only, required by loguru and tqdm |

For development only (the `dev` group, which `uv sync` installs by default): pytest (MIT), iniconfig (MIT), pluggy
(MIT), Pygments (BSD-2-Clause), packaging (Apache-2.0 or BSD-2-Clause).

## External programs

| Program | Licence | Used for |
|---|---|---|
| ffmpeg / ffprobe | LGPL-2.1-or-later, or GPL-2.0-or-later (GPL-3.0-or-later for some builds), depending on the build. Debian's package in the Docker image is GPL | decoding the video (called as a separate program, not linked) |
| NVIDIA driver and NVDEC (optional) | proprietary, from the host | hardware decoding |

## Related software, not included

- OpenCV calibration functions (`cv2.calibrateCamera`, `cv2.solvePnP`): how the
  calibration file is obtained ([CALIBRATION.md](CALIBRATION.md)).
- COLMAP (BSD-3-Clause): solved the camera pose of 26-Stairwell. It is not needed to run anything here.
- PeTrack and PedPy: read the PeTrack-layout output.

## Docker base image

`python:3.12-slim` (Debian), pinned by digest in the `Dockerfile`. Python is under the Python Software Foundation
licence, and the system packages (ffmpeg and its dependencies) under their Debian package licences.
