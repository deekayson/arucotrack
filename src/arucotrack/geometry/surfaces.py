"""Walking surfaces: project a head point straight down onto the floor under it.

A surface is a planar polygon in world metres (z up), `a x + b y + c z + d = 0`.
The surfaces come from a JSON file (`floor.surfaces` of the configuration, read by
`load_surfaces`), for example planes fitted to a photogrammetry mesh of the site.
A point drops vertically onto the plane of the polygon containing its xy. If
several contain it, the highest plane that is not above the point wins (a landing
under a mezzanine is not a floor for a head that is beneath it). Points outside
every polygon, below every candidate plane, or non-finite get `NaN` and surface
id `-1`: refused, not guessed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A head exactly on a plane (or the plane a hair above it through noise) still
# belongs to it.
Z_TOL_M = 1e-6
# Points on a polygon boundary count as inside, within this distance.
EDGE_TOL_M = 1e-9


@dataclass(frozen=True)
class Surface:
    id: int
    kind: str
    polygon_xy: np.ndarray            # (n, 2)
    plane: tuple[float, float, float, float]
    z_mean: float = float("nan")
    rms_mm: float = float("nan")
    n_pts: int = 0

    def z_at(self, x, y):
        a, b, c, d = self.plane
        return -(a * x + b * y + d) / c


def _in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test, boundary points count as inside."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    on_edge = np.zeros(len(xy), bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        dx, dy = xj - xi, yj - yi
        L2 = dx * dx + dy * dy
        if L2 > 0:
            t = np.clip(((x - xi) * dx + (y - yi) * dy) / L2, 0.0, 1.0)
            dist = np.hypot(x - (xi + t * dx), y - (yi + t * dy))
            on_edge |= dist <= EDGE_TOL_M
        cross = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cross & (x < x_cross)
        j = i
    return inside | on_edge


def project_to_surfaces(xyz, surfaces) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    n = len(xyz)
    out = np.full((n, 3), np.nan)
    sid = np.full(n, -1, int)
    best_z = np.full(n, -np.inf)
    finite = np.isfinite(xyz).all(axis=1)
    for s in surfaces:
        poly = np.asarray(s.polygon_xy, float)
        lo, hi = poly.min(0), poly.max(0)
        cand = finite & (xyz[:, 0] >= lo[0] - 1e-9) & (xyz[:, 0] <= hi[0] + 1e-9) \
            & (xyz[:, 1] >= lo[1] - 1e-9) & (xyz[:, 1] <= hi[1] + 1e-9)
        idx = np.flatnonzero(cand)
        if not len(idx):
            continue
        inside = _in_polygon(xyz[idx, :2], poly)
        idx = idx[inside]
        z = s.z_at(xyz[idx, 0], xyz[idx, 1])
        ok = (z <= xyz[idx, 2] + Z_TOL_M) & (z > best_z[idx])
        idx, z = idx[ok], z[ok]
        best_z[idx] = z
        sid[idx] = s.id
        out[idx, 0] = xyz[idx, 0]
        out[idx, 1] = xyz[idx, 1]
        out[idx, 2] = z
    return out, sid


def load_surfaces(path) -> list[Surface]:
    """The surfaces of a walking-surfaces JSON (`docs/FORMATS.md`, "Walking surfaces"), checked. Read per surface:
    `id`, `polygon_xy`, `plane`, and the optional `kind`, `z_mean`, `rms_mm` and `n_pts`. Raises ValueError naming
    the file and the surface id for a missing file, a frame other than `world_m`, no surfaces, a polygon with fewer
    than three points, or a plane that is vertical (c = 0) or not four finite numbers."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: the walking-surfaces file does not exist")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from None
    if not isinstance(doc, dict) or doc.get("frame") != "world_m":
        raise ValueError(f"{path}: frame is {doc.get('frame') if isinstance(doc, dict) else None!r}, expected "
                         f"'world_m'")
    items = doc.get("surfaces")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: `surfaces` must be a non-empty list of surfaces")
    out = []
    for k, s in enumerate(items):
        sid = s.get("id", f"#{k}") if isinstance(s, dict) else f"#{k}"
        where = f"{path}: surface {sid}"
        if not isinstance(s, dict) or not isinstance(s.get("id"), int) or isinstance(s.get("id"), bool):
            raise ValueError(f"{where}: needs an integer `id`")
        try:
            poly = np.asarray(s["polygon_xy"], float)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{where}: needs `polygon_xy`, a list of [x, y] points") from None
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
            raise ValueError(f"{where}: `polygon_xy` needs at least three finite [x, y] points")
        try:
            plane = np.asarray(s["plane"], float).ravel()
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{where}: needs `plane`, [a, b, c, d] of a x + b y + c z + d = 0") from None
        if plane.size != 4 or not np.isfinite(plane).all():
            raise ValueError(f"{where}: `plane` must be four finite numbers [a, b, c, d]")
        if abs(plane[2]) < 1e-9:
            raise ValueError(f"{where}: `plane` has c = 0, a vertical plane, which has no height under a head")
        out.append(Surface(s["id"], str(s.get("kind", "")), poly, tuple(float(v) for v in plane),
                           float(s.get("z_mean", np.nan)), float(s.get("rms_mm", np.nan)), int(s.get("n_pts", 0))))
    return out
