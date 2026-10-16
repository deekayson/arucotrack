"""Decoder choice of the luma path (`video.select_decoder`): NVDEC when a GPU and ffmpeg's CUVID decoder are there,
software otherwise, `ARUCOTRACK_DECODER` / `camera.decoder` to override. And the two decoders give the same luma bytes and
frame counts on a short synthetic HEVC clip (libx265, 4:4:4 and 4:2:0)."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from arucotrack.io import video
from arucotrack.io.video import DECODER_ENV, DecoderUnavailable, frames, probe, select_decoder


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(DECODER_ENV, raising=False)
    video._nvdec_probe.cache_clear()
    yield
    video._nvdec_probe.cache_clear()


def test_auto_takes_nvdec_when_available_and_software_otherwise(monkeypatch):
    monkeypatch.setattr(video, "nvdec_available", lambda codec: True)
    assert select_decoder("nvdec_y", "hevc") == "hevc_cuvid"
    assert select_decoder("sw_y", "hevc") is None                     # sw_y is always software
    assert select_decoder("nvdec_y", "rawvideo") is None              # no CUVID decoder for the codec
    monkeypatch.setattr(video, "nvdec_available", lambda codec: False)
    assert select_decoder("nvdec_y", "hevc") is None                  # no GPU: software, no flag needed
    assert "NVDEC not used" in video.decoder_label("nvdec_y", "hevc")


def test_the_override_forces_either_path(monkeypatch):
    monkeypatch.setattr(video, "nvdec_available", lambda codec: True)
    monkeypatch.setenv(DECODER_ENV, "sw")
    assert select_decoder("nvdec_y", "hevc") is None
    monkeypatch.setenv(DECODER_ENV, "nvdec")
    assert select_decoder("nvdec_y", "hevc") == "hevc_cuvid"
    monkeypatch.setattr(video, "nvdec_available", lambda codec: False)
    with pytest.raises(DecoderUnavailable):
        select_decoder("nvdec_y", "hevc")
    monkeypatch.setenv(DECODER_ENV, "bogus")
    with pytest.raises(ValueError):
        select_decoder("nvdec_y", "hevc")


def test_an_empty_cuda_visible_devices_means_no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert not video.nvdec_available("hevc")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert not video.nvdec_available("hevc")


def test_the_configured_decoder_sets_the_mode(monkeypatch):
    monkeypatch.delenv(video.DECODER_ENV, raising=False)
    video.set_decoder(None)
    assert video.decoder_mode() == "auto"
    video.set_decoder("sw")
    assert video.decoder_mode() == "sw"
    with pytest.raises(ValueError):
        video.set_decoder("gpu")


def _has_libx265() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return " libx265 " in out


@pytest.mark.skipif(not _has_libx265(), reason="ffmpeg with libx265 not present")
@pytest.mark.parametrize("pix_fmt", ["yuv444p", "yuv420p"])
def test_software_and_nvdec_give_identical_luma_and_counts(tmp_path, monkeypatch, pix_fmt):
    clip = tmp_path / f"synth_{pix_fmt}.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
                    "-frames:v", "45", "-c:v", "libx265", "-pix_fmt", pix_fmt, "-x265-params", "log-level=error",
                    str(clip)], check=True)
    info = probe(clip)
    assert info.codec == "hevc" and info.n_packets == 45
    monkeypatch.setenv(DECODER_ENV, "sw")
    sw = [f.gray.copy() for f in frames(clip, "nvdec_y", info=info)]
    assert len(sw) == info.n_packets and sw[0].shape == (240, 320)
    assert [f.gray.tobytes() for f in frames(clip, "sw_y", info=info)] == [g.tobytes() for g in sw]
    monkeypatch.setenv(DECODER_ENV, "auto")
    if not video.nvdec_available("hevc"):
        pytest.skip("no visible NVIDIA GPU or hevc_cuvid: the software path was checked, NVDEC was not")
    monkeypatch.setenv(DECODER_ENV, "nvdec")
    hw = [f.gray.copy() for f in frames(clip, "nvdec_y", info=info)]
    assert len(hw) == len(sw) == info.n_packets
    diff = np.abs(np.stack(sw).astype(np.int16) - np.stack(hw).astype(np.int16))
    assert diff.max() == 0, f"NVDEC differs from software decode: max |diff| {diff.max()}"


def test_auto_without_a_gpu_says_so_once(monkeypatch):
    from loguru import logger
    monkeypatch.setattr(video, "nvdec_available", lambda codec: False)
    video._announced.clear()
    msgs: list = []
    sink = logger.add(lambda m: msgs.append(m.record["message"]), level="INFO")
    try:
        assert select_decoder("nvdec_y", "hevc") is None
        assert select_decoder("nvdec_y", "hevc") is None
    finally:
        logger.remove(sink)
    assert len(msgs) == 1 and "software" in msgs[0] and "no usable GPU" in msgs[0]


def test_only_the_luma_plane_sources_exist():
    assert video.SOURCES == ("nvdec_y", "sw_y")
    for f in (video.decoder_label, video.pixel_pipeline_label):
        with pytest.raises(ValueError, match="source must be one of"):
            f("cv2_bgr_gray")
