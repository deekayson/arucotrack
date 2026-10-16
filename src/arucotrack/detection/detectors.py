"""ArUco detection: the multipass OpenCV detector and the shared detection type.

`MultiPass` runs `cv2.aruco.ArucoDetector` over several adaptive-threshold constants and image scales and merges the
passes (see its docstring). `parse_spec` builds it from the spec string `config.det_spec` makes of the `detect` group
(e.g. "multipass:subpix ac=3,6,9 scales=1,2"). An optional roster restricts the dictionary to the
ids in use. Its codes lie farther apart than those of the whole dictionary, so more bit errors can be corrected
without a wrong id becoming possible: up to (d - 1) / 2 bits for a roster whose minimum Hamming distance is d
(`roster_hmax`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    id: int
    corners: np.ndarray                 # (4,2) float64, image px, OpenCV order
    score: float = float("nan")         # detector confidence if it has one
    extra: dict[str, Any] = field(default_factory=dict)


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).reshape(4, 2)
    b = np.asarray(b, np.float32).reshape(4, 2)
    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

REFINE = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}
DICT = cv2.aruco.DICT_ARUCO_MIP_36h12
"""The default dictionary (the configuration's `marker.dictionary`). Every function also takes another 6x6 one."""


def dictionary_id(name: str | int | None) -> int:
    """A `cv2.aruco` predefined-dictionary constant from its name (`DICT_ARUCO_MIP_36h12`), or DICT for None."""
    if name is None:
        return DICT
    if isinstance(name, int):
        return name
    if not str(name).startswith("DICT_") or not hasattr(cv2.aruco, str(name)):
        raise ValueError(f"{name!r} is not a cv2.aruco predefined dictionary")
    return int(getattr(cv2.aruco, str(name)))


def dictionary_name(dictionary: int) -> str:
    """The `cv2.aruco` name of a predefined-dictionary constant (`DICT_ARUCO_MIP_36h12` for DICT)."""
    if int(dictionary) == DICT:
        return "DICT_ARUCO_MIP_36h12"
    return next((n for n in sorted(dir(cv2.aruco)) if n.startswith("DICT_") and getattr(cv2.aruco, n) == int(dictionary)),
                str(int(dictionary)))


def rotational_bit_layout():
    """AprilTag's bit ordering: quadrant generators, then their 90/180/270 rotations.

    The library's `rotate90()` rotates a *code* by a cyclic shift of nbits/4, which
    is only a rotation of the *tag* if bit i+9 sits where bit i lands after a 90
    degree turn. Row-major order (the obvious choice) silently decodes nothing.
    Cells are 1..6 inside the 8-wide border frame. A 90 degree turn about the
    centre (3.5, 3.5) maps (x, y) -> (7 - y, x).
    """
    quad = [(x, y) for y in (1, 2, 3) for x in (1, 2, 3)]
    bx, by = [], []
    for k in range(4):
        for (x, y) in quad:
            for _ in range(k):
                x, y = 7 - y, x
            bx.append(x); by.append(y)
    return bx, by


def mip36h12_codes(dictionary: int = DICT):
    """(codes, bit_x, bit_y): every code of a 6x6 dictionary (MIP_36h12 by default) in the rotational bit order."""
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    bx, by = rotational_bit_layout()
    n = d.bytesList.shape[0]
    codes = np.zeros(n, np.uint64)
    for i in range(n):
        bits = d.getBitsFromByteList(d.bytesList[i:i + 1], d.markerSize)   # (6,6) as (row, col), 1 = white
        v = 0
        for x, y in zip(bx, by):
            v = (v << 1) | int(bits[y - 1, x - 1])
        codes[i] = np.uint64(v)
    return codes, bx, by


def rotational_min_distance(codes: np.ndarray) -> int:
    """Minimum Hamming distance between distinct codes over all four rotations.

    In AprilTag's bit order a 90 degree turn is a cyclic shift by nbits/4, so
    rotations are shifts of the 36-bit word.
    """
    c = [int(x) for x in codes]
    mask = (1 << 36) - 1

    def rot(v, k):
        for _ in range(k):
            v = ((v << 9) & mask) | (v >> 27)
        return v

    best = 36
    for i in range(len(c)):
        for j in range(len(c)):
            for k in range(4):
                if i == j and k == 0:
                    continue
                best = min(best, bin(c[i] ^ rot(c[j], k)).count("1"))
    return best


def restricted_dictionary(ids: list[int], max_correction_bits: int | None = None,
                          dictionary: int = DICT) -> cv2.aruco.Dictionary:
    """A sub-dictionary of `dictionary` holding only `ids`. Row k decodes as ids[k].

    Off-roster codes then cannot decode at all, and because the surviving codes
    are farther apart the same `errorCorrectionRate` corrects more bit errors
    without a wrong id becoming possible. `max_correction_bits` defaults to the
    full dictionary's. Pass `roster_hmax(ids)` to use the subset's own bound.
    """
    full = cv2.aruco.getPredefinedDictionary(dictionary)
    mcb = full.maxCorrectionBits if max_correction_bits is None else int(max_correction_bits)
    return cv2.aruco.Dictionary(full.bytesList[np.asarray(ids, int)], full.markerSize, mcb)


def roster_min_distance(ids: list[int], dictionary: int = DICT) -> int:
    """Minimum Hamming distance between the roster's codes over all four rotations (self-rotations included)."""
    codes, _, _ = mip36h12_codes(dictionary)
    return rotational_min_distance(codes[np.asarray(ids, int)])


def roster_hmax(ids: list[int], dictionary: int = DICT) -> int:
    """Bit errors correctable without ambiguity: floor((tau - 1) / 2), Garrido-Jurado 2014 Sect. 3.4."""
    return (roster_min_distance(ids, dictionary) - 1) // 2


@dataclass
class OpenCVAruco:
    """`cv2.aruco.ArucoDetector` with explicit parameters.

    `params` are DetectorParameters attribute overrides. `refine` is one of
    REFINE. `roster` restricts the dictionary to those ids.
    """
    refine: str = "subpix"
    params: dict[str, Any] = field(default_factory=dict)
    roster: list[int] | None = None
    name: str = ""
    max_correction_bits: int | None = None     # roster dictionaries only (None: the full dictionary's bound)
    dictionary: int = DICT

    def __post_init__(self):
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = REFINE[self.refine]
        for k, v in self.params.items():
            if not hasattr(p, k):
                raise KeyError(f"DetectorParameters has no attribute {k!r}")
            setattr(p, k, type(getattr(p, k))(v))
        d = (restricted_dictionary(self.roster, self.max_correction_bits, self.dictionary) if self.roster
             else cv2.aruco.getPredefinedDictionary(self.dictionary))
        self._det = cv2.aruco.ArucoDetector(d, p)
        self._map = list(self.roster) if self.roster else None
        if not self.name:
            bits = [f"opencv:{self.refine}"] + [f"{k}={v}" for k, v in sorted(self.params.items())]
            if self.roster:
                bits.append("roster")
            self.name = " ".join(bits)

    def detect_with_rejected(self, gray: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """One detectMarkers call: (detections, rejected candidate quads (m,4,2))."""
        cs, ids, rej = self._det.detectMarkers(gray)
        out = []
        if ids is not None:
            for c, i in zip(cs, ids.ravel()):
                mid = self._map[int(i)] if self._map else int(i)
                out.append(Detection(mid, np.asarray(c, np.float64).reshape(4, 2)))
        rejected = np.asarray(rej, np.float64).reshape(-1, 4, 2) if rej is not None and len(rej) else np.zeros((0, 4, 2))
        return out, rejected


def _upscale_back(corners: np.ndarray, k: float) -> np.ndarray:
    """Corners found on a k-times upscaled image, in the original's pixel-centre convention.

    `cv2.resize` maps source centre x to destination (x + 0.5) k - 0.5 (the convention
    `refine._bilinear` documents), so the inverse is (x_up + 0.5) / k - 0.5.
    """
    return (np.asarray(corners, np.float64) + 0.5) / k - 0.5


@dataclass
class MultiPass:
    """Union of OpenCV passes over adaptive-threshold constants and image scales.

    The threshold constant that finds a marker depends on the image (a low
    constant does best on synthetic markers with a dark surround and worse on the
    public Flying-ArUco images), so instead of choosing one, every pass runs and
    the detections are merged:

    - Passes run in order of ascending scale, then the order of `constants`. A
      detection is deduplicated against earlier ones by `quad_iou` > `iou`,
      and the earliest (lowest scale, lowest index) is kept. The interior
      refiner does not depend on its seed, so which pass supplied it is irrelevant.
    - Two ids in one cluster is a **conflict**: the cluster is dropped and
      counted (`conflicts`), never voted on. The union multiplies decode
      attempts, so a disagreement is evidence against both.
    - With `roster`, the sub-dictionary gets the roster's own correction bound
      (`roster_hmax`, which grows as the roster shrinks) and `errorCorrectionRate`
      `ecr`, so up to that many bit errors are corrected without ambiguity.

    Upscaled passes map corners back with the half-pixel convention.
    """
    refine: str = "subpix"
    params: dict[str, Any] = field(default_factory=lambda: {"adaptiveThreshWinSizeMax": 73, "adaptiveThreshWinSizeStep": 4,
                                                             "perspectiveRemovePixelPerCell": 8})
    constants: tuple = (3, 6, 9)
    scales: tuple = (1, 2)
    roster: list[int] | None = None
    ecr: float = 1.0
    iou: float = 0.5
    interpolation: int = cv2.INTER_CUBIC
    name: str = ""
    conflicts: int = 0
    dictionary: int = DICT

    def __post_init__(self):
        self._passes: list[tuple[float, OpenCVAruco]] = []
        hmax = roster_hmax(self.roster, self.dictionary) if self.roster else None
        for k in sorted(self.scales):
            for c in self.constants:
                p = dict(self.params, adaptiveThreshConstant=c)
                if self.roster:
                    p["errorCorrectionRate"] = self.ecr
                self._passes.append((float(k), OpenCVAruco(self.refine, p, self.roster, max_correction_bits=hmax,
                                                           dictionary=self.dictionary)))
        self.hmax = hmax
        if not self.name:
            bits = [f"multipass:{self.refine}", "ac=" + ",".join(str(c) for c in self.constants),
                    "scales=" + ",".join(f"{s:g}" for s in sorted(self.scales))]
            bits += [f"{k}={v}" for k, v in sorted(self.params.items())]
            if self.roster:
                bits.append(f"roster(h<={hmax}) ecr={self.ecr:g}")
            self.name = " ".join(bits)

    def _scaled(self, gray: np.ndarray, k: float) -> np.ndarray:
        if k == 1.0:
            return gray
        return cv2.resize(gray, None, fx=k, fy=k, interpolation=self.interpolation)

    def detect_with_rejected(self, gray: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """(merged detections, union of every pass's rejected quads in original px, not deduplicated)."""
        clusters: list[list[Detection]] = []
        cache: dict[float, np.ndarray] = {}
        rejs = []
        for k, d in self._passes:
            img = cache.setdefault(k, self._scaled(gray, k))
            dets, rej = d.detect_with_rejected(img)
            if len(rej):
                rejs.append(_upscale_back(rej, k) if k != 1.0 else rej)
            for det in dets:
                if k != 1.0:
                    det.corners = _upscale_back(det.corners, k)
                det.extra["pass"] = (k, d.params["adaptiveThreshConstant"])
                for cl in clusters:
                    if quad_iou(cl[0].corners, det.corners) > self.iou:
                        cl.append(det)
                        break
                else:
                    clusters.append([det])
        out = []
        for cl in clusters:
            if len({x.id for x in cl}) > 1:
                self.conflicts += 1
                continue
            cl[0].extra["n_passes"] = len(cl)
            out.append(cl[0])
        return out, (np.concatenate(rejs) if rejs else np.zeros((0, 4, 2)))


def parse_spec(spec: str, roster: list[int] | None = None, dictionary: int = DICT) -> "MultiPass":
    """'multipass:subpix ac=3,6,9 scales=1,2 iou=0.5 roster adaptiveThreshWinSizeMax=73 ...' -> MultiPass (the spec
    `config.det_spec` writes). Values with a '.' or exponent become float, otherwise int."""
    toks = spec.split()
    kind, _, refine = toks[0].partition(":")
    if kind != "multipass":
        raise ValueError(f"unknown detector kind {kind!r}")
    params: dict[str, Any] = {}
    use_roster = False
    mp: dict[str, Any] = {}
    for t in toks[1:]:
        if t == "roster":
            use_roster = True
            continue
        k, _, v = t.partition("=")
        if k == "ac":
            mp["constants"] = tuple(int(x) for x in v.split(","))
        elif k == "scales":
            mp["scales"] = tuple(float(x) for x in v.split(","))
        elif k == "iou":
            mp["iou"] = float(v)
        else:
            params[k] = float(v) if any(ch in v for ch in ".eE") else int(v)
    if use_roster and not roster:
        raise ValueError("multipass spec asks for the roster but none was given")
    base = {"adaptiveThreshWinSizeMax": 73, "adaptiveThreshWinSizeStep": 4, "perspectiveRemovePixelPerCell": 8}
    base.update(params)
    return MultiPass(refine or "subpix", base, roster=(roster if use_roster else None), dictionary=dictionary, **mp)
