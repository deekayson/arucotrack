"""Writer of the single-camera trajectory file (`pipeline`, format in docs/FORMATS.md).

The layout has 21 columns: the 20 of `COLUMNS` plus `markerID`, which repeats `id`. `source`, `miss_mm` and
`ray_angle_deg` keep the column layout of the published files (in single-camera runs they are constant or NaN).
The output is deterministic: rows sorted by (id, frame), fixed decimals, `nan` for a missing number (never `-nan` or
`-0.0000`), `-` for an empty string field, and no wall-clock time. Three names of the column line carry a camera name
(`view_basler_deg`, `corner_basler`, `r_basler`). They are fixed by the published file format and describe the one
camera, whatever it is. The code calls them `view_deg`, `corner` and `r_norm` (`FILE_NAMES`)."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

COLUMNS = ("id", "frame", "t_s", "x", "y", "z", "observed", "source", "miss_mm", "ray_angle_deg", "view_deg",
           "corner", "r_norm", "reproj_px", "flags", "x_floor", "y_floor", "z_floor", "surface_id", "floor_src")
HEADER = ("id frame t_s x/m y/m z/m observed source miss_mm ray_angle_deg view_basler_deg corner_basler r_basler "
          "reproj_px flags markerID x_floor/m y_floor/m z_floor/m surface_id floor_src")
INT_COLS = ("id", "frame", "observed", "source", "surface_id", "floor_src")
# the column line's name of each column whose code name differs (x/m is x, ..., markerID repeats id)
FILE_NAMES = {"x": "x/m", "y": "y/m", "z": "z/m", "view_deg": "view_basler_deg", "corner": "corner_basler",
              "r_norm": "r_basler", "x_floor": "x_floor/m", "y_floor": "y_floor/m", "z_floor": "z_floor/m"}
STR_COLS = ("corner", "flags")
DECIMALS = {"t_s": 6, "x": 4, "y": 4, "z": 4, "miss_mm": 3, "ray_angle_deg": 3, "view_deg": 3, "r_norm": 4,
            "reproj_px": 3, "x_floor": 4, "y_floor": 4, "z_floor": 4}
FLOAT_COLS = tuple(c for c in COLUMNS if c not in INT_COLS + STR_COLS)
SOURCE_TABLE = "source: 0 interpolated, 1 measured"
OBSERVED_TABLE = "observed: 1 measured, 0 interpolated (not a measurement)"
CODED_TEXT_TABLE = ("corner_basler: refined | detector | refused_refinement, the corners a measured row was posed from; "
                    "flags: interp on rows filled linearly")
FLOOR_SRC_TABLE = ("floor_src: 0 no surface under the head (floor columns nan, surface_id -1), 1 measured surface, "
                   "2 extrapolated surface (a surface continued beyond the measured ones)")


def num(v, nd: int) -> str:
    v = float(v)
    if not math.isfinite(v):
        return "nan"
    s = f"{v:.{nd}f}"
    return s.lstrip("-") if s.lstrip("-").strip("0.") == "" else s


def write_interface(path, cols: dict, header: list) -> dict:
    missing = [c for c in COLUMNS if c not in cols]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    n = len(cols["id"])
    o = np.lexsort((np.asarray(cols["frame"]), np.asarray(cols["id"]))) if n else []
    lines = [f"# {h}" for h in header] + [HEADER]
    for k in o:
        parts = []
        for c in COLUMNS:
            v = cols[c][k]
            if c in DECIMALS:
                parts.append(num(v, DECIMALS[c]))
            elif c in STR_COLS:
                s = str(v).strip()
                parts.append(s.replace(" ", "_") if s else "-")
            else:
                parts.append(str(int(v)))
            if c == "flags":
                parts.append(str(int(cols["id"][k])))        # markerID, before the floor columns
        lines.append(" ".join(parts))
    path = Path(path)
    write_atomic(path, ("\n".join(lines) + "\n").encode())
    return {"file": str(path), "rows": n}


def write_atomic(path, data: bytes) -> None:
    """Write `data` to `<path>.tmp` and rename it to `path`, so an interrupted write never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
