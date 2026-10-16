import datetime as dt
import re
import subprocess
from pathlib import Path

import pytest

import arucotrack
from arucotrack import pipeline, provenance

REL = "take1.mkv"


def test_git_state_outside_checkout_is_unknown(tmp_path):
    st = provenance.code_state(package_dir=tmp_path / "site-packages" / "arucotrack")
    assert st["commit"] is None


def test_git_state_ignores_an_unrelated_enclosing_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    pkg = tmp_path / ".venv" / "lib" / "site-packages" / "arucotrack"
    pkg.mkdir(parents=True)
    assert provenance.code_state(package_dir=pkg)["commit"] is None


def test_git_state_ignores_another_project_with_a_src_layout(tmp_path):
    """A src/arucotrack inside some other project's checkout is not this package's checkout."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    pkg = tmp_path / "src" / "arucotrack"
    pkg.mkdir(parents=True)
    assert provenance.code_state(package_dir=pkg)["commit"] is None


def test_code_state_of_this_checkout_names_its_commit():
    pkg = Path(arucotrack.__file__).resolve().parent
    if pkg.parent.name != "src" or not (pkg.parents[1] / ".git").exists():
        pytest.skip("not running from a git checkout")
    st = provenance.code_state()
    assert re.fullmatch(r"[0-9a-f]{40}", st["commit"]) and isinstance(st["dirty"], bool)


def test_headers_name_the_code_version_and_no_commit_or_hash(monkeypatch):
    monkeypatch.setattr(provenance, "code_state", lambda package_dir=None: {"commit": None, "dirty": None})
    from tests.conftest import reference_calib
    lines = pipeline.header_lines("clip", reference_calib(), pipeline.Params(), 60.0, "none", pipeline.code_state(),
                                  "fused")
    assert lines[0].startswith(f"arucotrack {arucotrack.__version__} trajectories | clip clip | fused")
    assert not any(w in ln for ln in lines for w in ("sha256", "git", "unknown"))


def test_block_has_every_required_field():
    b = provenance.block(relpath=REL, decoder="test", pixel_pipeline="test", params={"x": 1},
                         started=dt.datetime.now(dt.timezone.utc))
    assert b["input"] == {"relpath": REL}
    assert set(b) == {"code", "environment", "finished_utc", "input", "parameters", "runtime_s", "started_utc"}
    assert all("commit" in c and "dirty" in c for c in b["code"])
    for k in ("opencv", "numpy", "ffmpeg", "python", "hostname", "decoder", "pixel_pipeline"):
        assert k in b["environment"]
    assert b["runtime_s"] >= 0 and b["started_utc"] <= b["finished_utc"]


def test_refine_needs_the_video_path_and_names_a_missing_one(tmp_path):
    from arucotrack.refinement import refine_cache
    with pytest.raises(ValueError, match="video"):
        refine_cache._source(None)
    with pytest.raises(FileNotFoundError, match="nowhere.mp4"):
        refine_cache._source(tmp_path / "nowhere.mp4")


def test_git_is_called_without_fsmonitor_or_optional_locks(monkeypatch, tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append((cmd, kw.get("env") or {}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    provenance._git(tmp_path, "rev-parse", "HEAD")
    cmd, env = seen[0]
    assert cmd[:3] == ["git", "-c", "core.fsmonitor=false"] and env.get("GIT_OPTIONAL_LOCKS") == "0"
