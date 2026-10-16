# Environment

ArUcoTrack is tested on Linux. macOS should work but is untested.

## Versions

| Component | Version | Where it is fixed |
|---|---|---|
| Python | 3.12 (3.12.13 in the Docker image) | `pyproject.toml` `requires-python`, Docker base image digest |
| opencv-python-headless | 5.0.0.93 (provides `DICT_ARUCO_MIP_36h12`, needs no GUI or OpenGL library) | `pyproject.toml` pin, `uv.lock` |
| numpy, scipy, omegaconf, ... | as locked | `uv.lock` (`uv sync --frozen`) |
| ffmpeg / ffprobe | tested with 6.1.1 (Ubuntu), which made the published trajectories and runs the tests. The Docker image has Debian's 7.1.5 | the host, or the Docker base image |
| NVIDIA driver (optional) | the runs of the published trajectories used 610.57.04 on an A100 | host |

No GPU library is a Python dependency. Nothing in the package imports torch, CUDA or cupy.

## Decoder

ffmpeg decodes the video into the luma plane. `camera.decoder` (or the environment variable `ARUCOTRACK_DECODER`)
chooses how:

- `auto` (default): NVDEC (ffmpeg's CUVID decoder) when all of these hold: `nvidia-smi -L` lists a GPU,
  `CUDA_VISIBLE_DEVICES` is not empty or `-1`, and ffmpeg has the codec's CUVID decoder. Otherwise software.
- `sw`: always software.
- `nvdec`: always NVDEC. It is an error when no GPU can be used.

The probe looks at `nvidia-smi` and the `ffmpeg -decoders` list only. If CUVID then fails while decoding, the run
stops with the decode error, and `camera.decoder=sw` is the remedy. The decoder used is written into every cache.

Software and NVDEC decoding give the same pixels on the clips checked. The luma planes of every frame were
compared by SHA-256 on an h264 4:2:0 2592x1944 clip of 720 frames and an HEVC 4:4:4 clip of 956 frames, with no
differing frame. A test (`tests/test_video_decoder.py`) checks this on synthetic HEVC 4:4:4 and 4:2:0 clips whenever
a GPU is present. On the 26-Stairwell clips checked, the detection caches made with CPU and with GPU decoding are
identical array for array, and so are the trajectory files.

## Speed and memory

Detection takes about 95 % of the run time. Measured on a server CPU (Intel Xeon Gold 6254, shared with other jobs)
with 2592 x 1944 frames and the default settings:

| OpenCV threads (`detect.threads`) | Detection speed |
|---|---|
| 4 | about 0.7 frames/s |
| 6 | about 1.0 frames/s |
| 8 | about 1.2 frames/s |
| 12 | about 1.6 frames/s |
| 18 | about 2.2 frames/s |

The `tilt` clip of 26-Stairwell (480 frames) takes about 7 minutes with 8 threads. All 15 clips (20,400 frames)
take about 4.6 hours at 1.24 frames/s (8 threads). Peak memory of a whole run with 8 threads was 489 MiB on the
`tilt` clip (480 frames) and 518 MiB on the longest clip (3420 frames): it grows only slowly with the length of the
video, because the detection cache keeps only the detections.

`detect.threads` is 0 by default, which uses every core the process may run on. On a shared machine set a number,
for example `detect.threads=8`. The detections are byte-identical for any number of threads. A GPU is used only
for decoding and does not change the speed materially, because the detector, not the decoder, is the cost.

Each stage is cached in `output.work`. Changing a refinement, gate or track parameter reruns only the stages after
detection: on the `tilt` clip the refinement takes about 8 s and the trajectory step under a second. Each stage
stamp names the code version, so after an update of the code every stage is redone. Use a fresh `output.work` then,
which keeps the old caches apart. To run several videos at once, run one video per process and size the number of
processes by the peak memory and the cores each process uses.

## Docker

The image is CPU-only: `python:3.12-slim` and uv, both pinned by digest, ffmpeg from Debian, the locked environment, and the
`arucotrack` command as entry point. Build it from the repository root and run it from the folder of the unzipped
dataset:

    docker build -t arucotrack .
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data -w /data arucotrack \
        run input/configs/tilt.yaml --data-root input --out out --work work

`--user` lets the container write into the mounted folder as you. With rootless Docker use `--user 0:0` instead,
which maps to your own user. Symbolic links inside the mounted folder must point to files that are also mounted.

GPU decoding in the same image: Debian's ffmpeg includes the CUVID decoders, so the image decodes with NVDEC when
the host's GPU is passed in. Use the NVIDIA container toolkit (`--gpus all`) or CDI
(`--device nvidia.com/gpu=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`). As above, this does not make
detection faster. The image is about 1.3 GB.
