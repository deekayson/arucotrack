"""Frames from video files, with drop-aware frame indices.

This module is built around three properties of recorded video:

- Never seek, and never derive time from a frame counter. A camera can drop frames (a few per recording on a
  global-shutter camera recording at 60 Hz), so `seq / fps` desyncs at every drop. A frame's identity is its packet
  sequence number `seq`. Its time is the packet PTS, always. A per-frame integer slot on the nominal frame-rate
  grid cannot be derived either: the PTS can carry about 0.5 s of arrival jitter at the start of a recording
  (deltas of 21-44 ms at 60 Hz that are not frame periods), so rounding elapsed time onto a grid collides
  neighbouring frames.
- The luma plane is taken straight from the decoder: no chroma, no colour conversion. NVDEC and software decoding
  then give the same pixels (the clips compared are listed in docs/ENVIRONMENT.md, "Decoder").
- ffmpeg's rawvideo muxer defaults to constant-frame-rate vsync, which would duplicate a frame at every PTS gap and
  silently undo the first point. Every pipe runs with `-fps_mode passthrough` and the frame count is checked
  against the packet count.

The frames are decoded as stored: `-noautorotate` is set, and `probe` refuses a stream with rotation metadata.

Decoder choice (CPU-only machines run the same pipeline): the `nvdec_y` source names the luma-plane path, not the
hardware. It decodes with NVDEC (the codec's CUVID decoder) when an NVIDIA GPU is visible and ffmpeg has that
decoder, and in software otherwise, with no flag needed. `ARUCOTRACK_DECODER` (`auto` default, `nvdec` to
require the GPU, `sw` to forbid it, set from the configuration's `camera.decoder`) overrides the choice. `sw_y` always decodes in
software. The decoder actually used is what `decoder_label` records in the provenance.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from loguru import logger

Source = Literal["nvdec_y", "sw_y"]
SOURCES: tuple[Source, ...] = ("nvdec_y", "sw_y")


class UnreadableVideo(RuntimeError):
    """The file has no decodable video stream (for example a truncated capture of a few hundred bytes)."""


class FrameCountMismatch(RuntimeError):
    """The decoder emitted a different number of frames than the stream has packets.

    Either vsync duplicated/dropped frames or the stream is damaged. In both
    cases the PTS join is wrong and nothing downstream may be trusted.
    """


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    codec: str
    pix_fmt: str
    pts: np.ndarray  # presentation timestamps, seconds, sorted, one per packet

    @property
    def n_packets(self) -> int:
        return int(len(self.pts))

    def timing(self) -> dict:
        """What the timestamps say, without pretending they are a grid."""
        if self.n_packets < 2:
            return {"n_packets": self.n_packets}
        d_ms = np.diff(self.pts) * 1000.0
        period_ms = 1000.0 / float(self.fps)
        span = float(self.pts[-1] - self.pts[0])
        u, c = np.unique(np.rint(d_ms).astype(int), return_counts=True)
        return {
            "n_packets": self.n_packets,
            "span_s": round(span, 3),
            "packets_per_span_fps": round((self.n_packets - 1) / span, 3),
            "nominal_fps": float(self.fps),
            "nominal_slots_missing": int(round(span * float(self.fps)) + 1 - self.n_packets),
            "delta_ms_median": round(float(np.median(d_ms)), 3),
            "delta_ms_mean_steady": round(float(d_ms[np.abs(d_ms - period_ms) < 2.0].mean()), 3),
            "n_deltas_over_1p5_periods": int((d_ms > 1.5 * period_ms).sum()),
            "delta_ms_histogram": {int(k): int(v) for k, v in zip(u, c)},
        }


@dataclass(frozen=True)
class Frame:
    seq: int          # packet sequence number = frame identity (0-based, presentation order)
    pts: float        # seconds, the only legitimate time of this frame
    gray: np.ndarray  # (H, W) uint8


def _run(cmd: list[str], timeout: float = 900) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}): {p.stderr.strip()[-800:]}")
    return p.stdout


def ffmpeg_version() -> str:
    return _run(["ffmpeg", "-version"]).splitlines()[0].split(" Copyright")[0]


# ffmpeg and ffprobe read local files and write to a pipe, nothing else (no network or device protocols).
PROTOCOLS = ["-protocol_whitelist", "file,pipe"]
MAX_SIDE_PX = 16384
"""Largest accepted frame width or height. A larger value in a file is a damaged or hostile header, and it would size
the frame buffer of every decoded frame."""


def _arg(path: Path) -> str:
    """The path as ffmpeg's input argument: absolute, so a file name that begins with `-` is not read as an option."""
    return str(Path(os.path.abspath(path)))


def probe(path: str | Path) -> VideoInfo:
    """Size, frame rate, codec, pixel format and the PTS of every packet of the first video stream. Refuses a stream
    with rotation metadata: the frames are decoded as stored (`-noautorotate`), and the calibration must describe
    that orientation."""
    path = Path(path)
    j = json.loads(_run(["ffprobe", "-v", "error", *PROTOCOLS, "-select_streams", "v:0",
                         "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt:stream_tags=rotate"
                         ":stream_side_data=rotation", "-of", "json", _arg(path)]))
    streams = j.get("streams") or []
    if not streams:
        raise UnreadableVideo(f"{path.name}: no video stream")
    s = streams[0]
    rot = [float(d["rotation"]) for d in s.get("side_data_list") or [] if d.get("rotation") not in (None, "")]
    rot += [float(s["tags"]["rotate"])] if (s.get("tags") or {}).get("rotate") not in (None, "") else []
    rot = [r for r in rot if r % 360]
    if rot:
        raise ValueError(f"{path.name}: the video stream carries rotation metadata ({rot[0]:g} degrees). ArUcoTrack "
                         f"decodes frames as stored and does not rotate them. Remove the rotation (for example "
                         f"`ffmpeg -display_rotation 0 -i in -c copy out`) and calibrate in the stored orientation.")
    w, h = int(s["width"]), int(s["height"])
    if not (0 < w <= MAX_SIDE_PX and 0 < h <= MAX_SIDE_PX):
        raise ValueError(f"{path.name}: frame size {w}x{h} is outside 1..{MAX_SIDE_PX} px per side")
    # Packet PTS needs no decode. Frames are emitted in presentation order, so the PTS are sorted.
    out = _run(["ffprobe", "-v", "error", *PROTOCOLS, "-select_streams", "v:0",
                "-show_entries", "packet=pts_time", "-of", "csv=p=0", _arg(path)])
    pts = []
    for tok in out.split():
        try:
            pts.append(float(tok))
        except ValueError:
            continue
    if not pts:
        raise UnreadableVideo(f"{path.name}: no video packets (damaged or empty stream, {path.stat().st_size} B)")
    try:
        fps = Fraction(s["r_frame_rate"])
    except (KeyError, ZeroDivisionError, ValueError):
        fps = Fraction(0)
    return VideoInfo(path=path, width=w, height=h,
                     fps=fps, codec=s.get("codec_name", ""), pix_fmt=s.get("pix_fmt", ""),
                     pts=np.sort(np.asarray(pts, dtype=np.float64)))


# The CUVID decoder of each codec that has one. A codec not listed (rawvideo, for example) always decodes in
# software. Whether a listed decoder is usable on this machine is checked separately (`nvdec_available`).
CUVID = {"hevc": "hevc_cuvid", "h264": "h264_cuvid", "av1": "av1_cuvid", "vp9": "vp9_cuvid",
         "vp8": "vp8_cuvid", "mpeg4": "mpeg4_cuvid", "mpeg2video": "mpeg2_cuvid",
         "mpeg1video": "mpeg1_cuvid", "vc1": "vc1_cuvid", "mjpeg": "mjpeg_cuvid"}


def hw_decoder_for(codec: str | None) -> str | None:
    """The CUVID decoder for this codec, or None when it must decode in software."""
    return CUVID.get((codec or "").lower())


DECODER_ENV = "ARUCOTRACK_DECODER"
DECODER_MODES = ("auto", "nvdec", "sw")


class DecoderUnavailable(RuntimeError):
    """`nvdec` was required but no NVIDIA GPU is visible or ffmpeg has no CUVID decoder for the codec."""


def decoder_mode() -> str:
    """`ARUCOTRACK_DECODER` (auto | nvdec | sw), `auto` when unset or empty."""
    m = (os.environ.get(DECODER_ENV) or "auto").strip().lower()
    if m not in DECODER_MODES:
        raise ValueError(f"{DECODER_ENV}={m!r}: one of {DECODER_MODES}")
    return m


def set_decoder(mode: str | None) -> None:
    """Set the decoder mode for this process and its children (`camera.decoder`). None leaves it as it is."""
    if mode is None:
        return
    if mode not in DECODER_MODES:
        raise ValueError(f"decoder {mode!r}: one of {DECODER_MODES}")
    os.environ[DECODER_ENV] = mode


@functools.cache
def _nvdec_probe(cuda_visible: str | None, decoder: str) -> bool:
    if cuda_visible is not None and cuda_visible.strip() in ("", "-1"):
        return False                                    # no GPU visible to this process
    if shutil.which("nvidia-smi") is None or shutil.which("ffmpeg") is None:
        return False
    try:
        gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        if gpu.returncode or "GPU" not in gpu.stdout:
            return False
        dec = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return dec.returncode == 0 and any(line.split()[1:2] == [decoder] for line in dec.stdout.splitlines())


def nvdec_available(codec: str | None) -> bool:
    """An NVIDIA GPU is visible (`nvidia-smi -L`, `CUDA_VISIBLE_DEVICES` not empty or -1) and ffmpeg has the CUVID
    decoder of `codec`. Cached per (CUDA_VISIBLE_DEVICES, decoder)."""
    dec = hw_decoder_for(codec)
    return dec is not None and _nvdec_probe(os.environ.get("CUDA_VISIBLE_DEVICES"), dec)


_announced: set[str] = set()   # decoders whose software fallback has been logged (once per process)


def select_decoder(source: Source, codec: str | None) -> str | None:
    """The ffmpeg decoder `frames` uses for `source`: the CUVID decoder, or None for software.

    `sw_y` is always software. `nvdec_y` follows `decoder_mode()`. `auto` takes NVDEC when `nvdec_available`, else
    software. `nvdec` requires it and raises DecoderUnavailable when it is absent (a codec with no CUVID decoder at
    all still decodes in software). `sw` never uses it."""
    if source != "nvdec_y":
        return None
    hw = hw_decoder_for(codec)
    if hw is None:
        return None
    mode = decoder_mode()
    if mode == "sw":
        return None
    if nvdec_available(codec):
        return hw
    if mode == "nvdec":
        raise DecoderUnavailable(f"{DECODER_ENV}=nvdec but {hw} is not usable here (no visible NVIDIA GPU or no "
                                 f"CUVID decoder in this ffmpeg)")
    if hw not in _announced:
        _announced.add(hw)
        logger.info("decoder auto: no usable GPU ({} unavailable), decoding in software", hw)
    return None


def _check_source(source) -> None:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")


def decoder_label(source: Source, codec: str | None = None) -> str:
    _check_source(source)
    hw = select_decoder(source, codec)
    how = f"{hw} (NVDEC)" if hw else f"software {codec or 'auto'}"
    if source == "nvdec_y" and not hw:
        how += (" [no CUVID decoder for this codec]" if hw_decoder_for(codec) is None
                else f" [NVDEC not used: {DECODER_ENV}={decoder_mode()}, no usable GPU or CUVID decoder, or sw chosen]")
    return f"{ffmpeg_version()} {how}"


def is_planar_yuv(pix_fmt: str | None) -> bool:
    """True when the luma plane can be taken straight from the decoded frame.

    Planar YUV and grey formats have one. A capture decoded to an RGB format such as `bgr0` has no Y plane and is
    converted instead (`grey_filter`).
    """
    p = (pix_fmt or "").lower()
    return p.startswith("yuv") or p.startswith("gray") or p.startswith("nv")


def grey_filter(pix_fmt: str | None) -> str:
    return "extractplanes=y" if is_planar_yuv(pix_fmt) else "format=gray"


def pixel_pipeline_label(source: Source, pix_fmt: str | None = None) -> str:
    _check_source(source)
    if is_planar_yuv(pix_fmt):
        return "luma plane as decoded, limited range (16-235), no colour conversion"
    # A different pipeline, and it matters: BT.601 luma from RGB is full
    # range, so grey-level constants (adaptiveThreshConstant) do not carry
    # over from the limited-range luma path.
    return f"BT.601 luma converted from {pix_fmt} (format=gray), full range - not the luma-plane path"


def frames(path: str | Path, source: Source = "nvdec_y", max_frames: int | None = None,
           info: VideoInfo | None = None) -> Iterator[Frame]:
    """Yield frames in presentation order, each with its packet PTS.

    Raises FrameCountMismatch if the decoder's frame count disagrees with the
    packet count (only checked when the whole stream is read): the guard
    against vsync silently duplicating or dropping frames.
    """
    _check_source(source)
    info = info or probe(path)
    yield from _frames_ffmpeg(info, hw=(source == "nvdec_y"), max_frames=max_frames)


def _read_exact(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _frames_ffmpeg(info: VideoInfo, hw: bool, max_frames: int | None) -> Iterator[Frame]:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *PROTOCOLS, "-noautorotate"]
    dec = select_decoder("nvdec_y" if hw else "sw_y", info.codec)
    if dec:
        cmd += ["-c:v", dec]
    cmd += ["-i", _arg(info.path), "-an", "-sn", "-fps_mode", "passthrough",
            "-vf", grey_filter(info.pix_fmt), "-pix_fmt", "gray"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-"]
    nbytes = info.width * info.height
    # stderr must not be a pipe: some streams make ffmpeg emit one warning per frame
    # (e.g. a rawvideo muxer complaining about non-monotonic dts), and a pipe's small
    # OS buffer fills before it is drained in `finally`. ffmpeg then blocks on the
    # stderr write forever, and this generator with it. A real file has no such
    # backpressure.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
        seq = 0
        try:
            while True:
                buf = _read_exact(proc.stdout, nbytes)
                if len(buf) < nbytes:
                    if buf:
                        raise FrameCountMismatch(f"{info.path.name}: truncated frame after {seq} frames")
                    break
                if seq >= info.n_packets:
                    raise FrameCountMismatch(f"{info.path.name}: more frames than packets ({info.n_packets}); vsync duplicated?")
                gray = np.frombuffer(buf, np.uint8).reshape(info.height, info.width)
                yield Frame(seq=seq, pts=float(info.pts[seq]), gray=gray)
                seq += 1
        finally:
            proc.stdout.close()
            rc = proc.wait()
            errf.seek(0)
            err = errf.read().decode(errors="replace")
    if rc not in (0, -13, 141):  # SIGPIPE codes occur only when the consumer stopped early
        raise RuntimeError(f"ffmpeg exited {rc}: {err.strip()[-800:]}")
    expected = info.n_packets if not max_frames else min(max_frames, info.n_packets)
    if seq != expected:
        raise FrameCountMismatch(f"{info.path.name}: decoded {seq} frames, expected {expected} "
                                 f"(packets {info.n_packets}); vsync or damaged stream")
