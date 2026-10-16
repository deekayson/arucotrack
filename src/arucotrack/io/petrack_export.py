"""Trajectories in PeTrack's own column layout (`<stem>.petrack.txt`), with an `observed` sidecar.

The layout is that of a PeTrack world-coordinate export:

    # framerate: 59.8598 fps
    # id frame x/m y/m z/m markerID

with three short lines between them: the units and frame, what `id` is, and what `frame` is. PeTrack's `id` is a
sequential person index from 1. Here `id` and `markerID` are both the ArUco marker id. `frame` is the frame index of
the video, so `frame / framerate` is not the time of a frame when the camera dropped one: the full layout
(`<stem>.fused*.txt`) has each frame's timestamp in `t_s`. The layout has no `observed` column, so the
measured/interpolated distinction goes to a sidecar `<stem>.petrack.observed.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .trajwrite import write_atomic


UNITS_WORLD = "# metres, world frame of the calibration's camera pose (z up); x y z = the marker centre"
UNITS_CAMERA = ("# metres, CAMERA frame (no camera pose): x right, y down, z along the optical axis; "
                "x y z = the marker centre")


def write(path: Path, tracks: dict, *, variant: str, fps: float, camera_frame: bool = False) -> dict:
    """`tracks`: per marker id, rows (frame, x, y, z, observed). `camera_frame`: the positions are camera coordinates
    (a calibration without a camera pose), which the units line says."""
    markers = sorted(tracks)
    lines = [
        f"# framerate: {fps:.4f} fps",
        UNITS_CAMERA if camera_frame else UNITS_WORLD,
        "# id: the ArUco marker id (markerID repeats it), not PeTrack's sequential person index",
        "# frame: the frame index of the video, from 0",
        "# id frame x/m y/m z/m markerID",
    ]
    obs: dict[str, list[int]] = {}
    n = 0
    for m in markers:
        a = tracks[m]
        for r in a:
            lines.append(f"{m} {int(r[0])} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} {m}")
            n += 1
        obs[str(m)] = [int(v) for v in a[:, 4]]
    write_atomic(path, ("\n".join(lines) + "\n").encode("utf-8"))
    side = {"file": path.name, "variant": variant, "id_convention": "marker",
            "frame_index": "packet PTS index",
            "observed_note": "1 = measured, 0 = interpolated across a gap (not a measurement)",
            "observed_codes": "1 = measured, 0 = interpolated (not a measurement)",
            "observed": obs}
    write_atomic(path.parent / f"{path.stem}.observed.json", json.dumps(side, indent=1).encode())
    return {"file": str(path), "rows": n, "markers": len(markers),
            "interpolated": int(sum(v.count(0) for v in obs.values()))}
