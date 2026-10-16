"""What `arucotrack.io.video` refuses or protects against before any frame is decoded."""
import shutil
import subprocess

import numpy as np
import pytest

from arucotrack.io import video


def _ffmpeg(*args):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on the PATH")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def test_a_file_name_starting_with_a_dash_is_read_as_a_file(tmp_path, blank_video, monkeypatch):
    """ffprobe and ffmpeg parse a bare `-name` as an option: the path is passed absolute."""
    dash = tmp_path / "-clip.mp4"
    shutil.copy(blank_video, dash)
    monkeypatch.chdir(tmp_path)
    from pathlib import Path
    info = video.probe(Path("-clip.mp4"))
    assert info.n_packets == 10
    assert sum(1 for _ in video.frames(Path("-clip.mp4"), "sw_y", info=info)) == 10


def test_every_ffmpeg_call_allows_local_files_only(blank_video, monkeypatch):
    seen = []
    real_run, real_popen = video._run, subprocess.Popen

    def run(cmd, timeout=900):
        seen.append(list(cmd))
        return real_run(cmd, timeout)

    def popen(cmd, *a, **k):
        seen.append(list(cmd))
        return real_popen(cmd, *a, **k)

    monkeypatch.setattr(video, "_run", run)
    monkeypatch.setattr(video.subprocess, "Popen", popen)
    info = video.probe(blank_video)
    list(video.frames(blank_video, "sw_y", info=info))
    calls = [c for c in seen if c[0] in ("ffprobe", "ffmpeg") and "-version" not in c]
    assert len(calls) >= 3
    for c in calls:
        i = c.index("-protocol_whitelist")
        assert c[i + 1] == "file,pipe"
    dec = [c for c in calls if c[0] == "ffmpeg"]
    assert dec and all("-noautorotate" in c and c.index("-noautorotate") < c.index("-i") for c in dec)


def test_a_video_with_rotation_metadata_is_refused(tmp_path, blank_video):
    rot = tmp_path / "rotated.mp4"
    _ffmpeg("-display_rotation", "90", "-i", str(blank_video), "-c", "copy", str(rot))
    with pytest.raises(ValueError, match="rotated.mp4.*rotation"):
        video.probe(rot)


def test_a_frame_larger_than_the_cap_is_refused(blank_video, monkeypatch):
    real = video._run

    def run(cmd, timeout=900):
        out = real(cmd, timeout)
        return out.replace('"width": 320', '"width": 20000') if "stream=width,height" in " ".join(cmd) else out

    monkeypatch.setattr(video, "_run", run)
    with pytest.raises(ValueError, match="20000x240.*16384"):
        video.probe(blank_video)
