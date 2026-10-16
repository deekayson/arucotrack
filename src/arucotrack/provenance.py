"""Provenance: every cache file records what produced it.

Every cache file carries the block built here: which input (its path as given), which code (the commit and dirty
flag of this package's own checkout, or None for an installed package), the full parameter set (never a diff from
defaults, because defaults move between OpenCV versions), the environment including *which decoder and pixel
pipeline*, the date, and the runtime. Nothing in it is typed by hand.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import socket
import subprocess
import tomllib
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """sha256 of a file, read once in chunks (the identity of a cache or an input in the work folder)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def _git(cwd: Path, *args: str) -> str:
    """Run a read-only git command in `cwd`. The file-system monitor and the optional index lock are off, so a
    repository's configuration cannot start a helper program and the call never writes to the repository."""
    import os
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", str(cwd), *args], capture_output=True,
                          text=True, check=True, timeout=20, env=env).stdout


def _is_own_checkout(package_dir: Path) -> bool:
    """True when `package_dir` is `<checkout>/src/arucotrack` of a checkout of this project: `src` layout, the
    checkout's pyproject.toml names project `arucotrack`, and git's top level is that checkout. An installed package
    (site-packages), or one inside some other repository, is not."""
    if package_dir.parent.name != "src":
        return False
    top = package_dir.parents[1]
    try:
        name = tomllib.loads((top / "pyproject.toml").read_text()).get("project", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        return False
    if name != "arucotrack":
        return False
    try:
        shown = Path(_git(top, "rev-parse", "--show-toplevel").strip())
    except (subprocess.SubprocessError, OSError):
        return False
    return shown.resolve() == top.resolve()


def code_state(package_dir: Path | None = None) -> dict[str, Any]:
    """`{"commit", "dirty"}` of this package's own git checkout (`dirty`: a tracked file differs from the commit).
    Both are None when the package is not running from its own checkout (installed, or inside an unrelated
    repository, whose commit would describe other code)."""
    package_dir = Path(__file__).resolve().parent if package_dir is None else Path(package_dir)
    if not _is_own_checkout(package_dir):
        return {"commit": None, "dirty": None}
    top = package_dir.parents[1]
    try:
        commit = _git(top, "rev-parse", "HEAD").strip()
        # Tracked files only: an uncommitted change to them makes the commit an incomplete description of the code,
        # while untracked files (run outputs such as out/ and work/ inside the checkout) do not.
        status = _git(top, "status", "--porcelain", "--untracked-files=no")
    except (subprocess.SubprocessError, OSError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": any(ln.strip() for ln in status.splitlines())}


def environment() -> dict[str, Any]:
    import cv2
    import numpy
    from .io.video import ffmpeg_version
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "ffmpeg": ffmpeg_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    # torch is not a dependency (nothing in the pipeline uses it). The key is part of the provenance schema and is
    # filled only where torch happens to be installed. The GPU, which matters for the NVDEC decode, comes from
    # nvidia-smi instead (None on a CPU-only machine or with CUDA_VISIBLE_DEVICES empty).
    try:
        import torch
        env["torch"] = torch.__version__
    except Exception:
        env["torch"] = None
    env["gpu"] = _gpu_name()
    return env


def _gpu_name() -> str | None:
    import os
    import shutil
    import subprocess
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (cvd is not None and cvd.strip() in ("", "-1")) or shutil.which("nvidia-smi") is None:
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    names = [n.strip() for n in p.stdout.splitlines() if n.strip()] if p.returncode == 0 else []
    return names[0] if names else None


def detector_params_dict(par) -> dict[str, Any]:
    """Every public attribute of a cv2.aruco.DetectorParameters, not a diff."""
    out = {}
    for k in dir(par):
        if k.startswith("_"):
            continue
        v = getattr(par, k)
        if callable(v):
            continue
        out[k] = v.item() if hasattr(v, "item") else v
    return out


def block(*, relpath: str, decoder: str, pixel_pipeline: str, params: dict[str, Any],
          started: _dt.datetime, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    finished = _dt.datetime.now(_dt.timezone.utc)
    b: dict[str, Any] = {
        "input": {"relpath": relpath},
        "code": [{"repo": "arucotrack", **code_state()}],
        "parameters": params,
        "environment": {**environment(), "decoder": decoder, "pixel_pipeline": pixel_pipeline},
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "runtime_s": round((finished - started).total_seconds(), 3),
    }
    if extra:
        b.update(extra)
    return b
