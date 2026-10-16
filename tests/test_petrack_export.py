"""PeTrack's column layout, and the three things it cannot carry.

The format is six columns and no more, so every distinction the 21-column trajectory
file makes has to be either stated in the header or moved to a sidecar. Dropping
one quietly is the failure these guard against: a consumer cannot tell an
interpolated row from a measured one by looking at it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arucotrack.io import petrack_export as PE


def _tracks():
    """{marker: (frame, x, y, z, observed)}: marker 7 with an interpolated middle row, marker 9 measured."""
    return {7: np.array([[10, 1.0, -2.0, 1.70, 1], [11, 1.1, -1.9, 1.71, 0], [12, 1.2, -1.8, 1.72, 1]]),
            9: np.array([[10, 0.0, -1.0, 1.80, 1], [11, 0.05, -0.9, 1.80, 1]])}


def _read(p: Path):
    hdr = [l for l in p.read_text().splitlines() if l.startswith("#")]
    rows = [l.split() for l in p.read_text().splitlines()
            if l and not l.startswith("#") and not l.startswith("id ")]
    return hdr, rows


def test_id_and_frame_are_the_marker_id_and_the_frame_index_and_say_so(tmp_path):
    """PeTrack's `id` is a sequential person index from 1. This export writes the marker id instead, and `frame` is the
    frame index of the video: both are stated in the header."""
    p = tmp_path / "o.txt"
    PE.write(p, _tracks(), variant="fused_smooth", fps=60.0)
    hdr, rows = _read(p)
    assert rows[0][0] == "7" and rows[0][5] == "7" and rows[0][1] == "10"
    assert any(h.startswith("# id: the ArUco marker id") for h in hdr)
    assert any(h.startswith("# frame: the frame index of the video") for h in hdr)
    assert len(hdr) == 5, "framerate, units, id, frame and the column line"


def test_the_observed_flag_survives_in_a_sidecar_because_the_layout_has_no_column(tmp_path):
    tracks = _tracks()
    p = tmp_path / "o.txt"
    rec = PE.write(p, tracks, variant="fused_smooth", fps=60.0)
    side = json.loads((tmp_path / "o.observed.json").read_text())
    assert side["observed"]["7"] == [1, 0, 1]
    assert rec["interpolated"] == 1
    # the notes describe exactly the codes the file can hold: 1 measured, 0 interpolated
    for k in ("observed_note", "observed_codes"):
        assert "1 = measured" in side[k] and "0 = interpolated" in side[k] and "2" not in side[k]


def test_the_column_line_is_commented_so_petrack_does_not_import_it_as_a_person(tmp_path):
    """PeTrack's `.txt` reader (`Petrack::importTracker`) skips lines starting with
    `#` and parses every other line as `persNr frameNr x y z`. Its `persNr`/`frameNr`
    default to -1 and its `x, y, z` are left uninitialised, so a bare column line is
    not a parse error - it is a phantom person -1 at frame -1 carrying whatever was
    on the stack."""
    tracks = _tracks()
    p = tmp_path / "o.txt"
    PE.write(p, tracks, variant="fused_smooth", fps=60.0)
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        assert len(f) == 6, f"a non-comment line PeTrack would misparse: {line!r}"
        int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4]), int(f[5])
    assert "# id frame x/m y/m z/m markerID" in p.read_text()


def test_the_column_line_is_the_last_comment_because_petrack_reads_the_unit_from_it(tmp_path):
    """PeTrack decides metres-versus-centimetres from the LAST `#` line before the
    first data row: absent a `cm` in it the file is taken as metres and scaled by
    100. The column line has to be the one that lands last."""
    tracks = _tracks()
    p = tmp_path / "o.txt"
    PE.write(p, tracks, variant="fused_smooth", fps=60.0)
    comments = [l for l in p.read_text().splitlines() if l.startswith("#")]
    assert comments[-1] == "# id frame x/m y/m z/m markerID"
    assert "cm" not in comments[-1], "would flip PeTrack's unit autodetect to centimetres"


def test_the_framerate_line_carries_its_unit_like_a_genuine_export(tmp_path):
    """PeTrack's own export writes `# framerate: 59.8598 fps`. The
    unit stays, so that a reader can tell a frame rate from a frame interval.
    """
    tracks = _tracks()
    p = tmp_path / "o.txt"
    PE.write(p, tracks, variant="fused_smooth", fps=59.8598)
    hdr, _ = _read(p)
    assert "# framerate: 59.8598 fps" in hdr
