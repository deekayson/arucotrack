"""The resolution of gate-failed rows (`refine_cache.resolve_gated`): a gated row is used only if its corners agree
with accepted neighbours on both sides of the same id. If both candidates disagree it is inadmissible (5). With a
neighbour on one side only, or none, it cannot be checked and is kept as unverified (4). Synthetic tracks, no video."""
from __future__ import annotations

import numpy as np

from arucotrack.refinement.gate import GATE_AGREE_PX, INADMISSIBLE, UNVERIFIED, resolve_gated

SQ = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])


def _track(n=9, speed=2.0):
    """One id moving +x at `speed` px per frame: truth corners per frame."""
    return np.stack([SQ + [speed * k, 0.0] for k in range(n)])


def _resolve(truth, gated_at, det_off, ref_off, **kw):
    """Rows accepted except `gated_at`. The detector is `det_off` px off truth there, the refusal `ref_off`."""
    n = len(truth)
    gated = np.zeros(n, bool)
    gated[gated_at] = True
    det, ref = truth.copy(), truth.copy()
    det[gated_at] += np.asarray(det_off, float)
    ref[gated_at] += np.asarray(ref_off, float)
    return resolve_gated(np.arange(n, dtype=float), np.full(n, 7), det, ref, ~gated, gated, **kw)


def test_detector_corners_that_agree_with_the_neighbours_are_used():
    code, agree = _resolve(_track(), 4, det_off=[0.4, 0.0], ref_off=[9.0, 0.0])
    assert code[4] == 2 and agree[4] < 1.0
    assert (np.delete(code, 4) == 1).all()


def test_a_refusal_that_agrees_better_is_used():
    code, agree = _resolve(_track(), 4, det_off=[14.0, 3.0], ref_off=[0.5, 0.0])      # a wrong cell vs a good fit
    assert code[4] == 3 and agree[4] < GATE_AGREE_PX


def test_both_far_from_the_prediction_is_inadmissible():
    code, agree = _resolve(_track(), 4, det_off=[12.0, 0.0], ref_off=[0.0, 8.0])
    assert code[4] == INADMISSIBLE and agree[4] > GATE_AGREE_PX            # the nearer of the two is still reported


def test_the_tolerance_is_a_parameter():
    off = [0.0, 4.0]
    assert _resolve(_track(), 4, off, [30.0, 0.0])[0][4] == INADMISSIBLE
    assert _resolve(_track(), 4, off, [30.0, 0.0], agree_px=5.0)[0][4] == 2


def test_no_accepted_neighbour_in_the_window_is_unverified_not_dropped():
    truth = _track(12, speed=0.0)
    n = len(truth)
    gated = np.zeros(n, bool)
    gated[3:9] = True                                           # a run of 6 gated rows between accepted rows 2 and 9
    t, ids = np.arange(n, dtype=float), np.full(n, 7)
    code, agree = resolve_gated(t, ids, truth, truth, ~gated, gated, window=2.0)
    # with window 2 no row has both sides: rows 3, 4 see row 2 only, rows 7, 8 see row 9 only, 5 and 6 none
    assert (code[3:9] == UNVERIFIED).all() and np.isnan(agree[3:9]).all()
    both = resolve_gated(t, ids, truth, truth, ~gated, gated, window=10.0)[0]
    assert (both[3:9] == 2).all()                               # a wide window gives every row both sides
    # no accepted row of the id at all
    code2, agree2 = resolve_gated(t, ids, truth, truth, np.zeros(n, bool), np.ones(n, bool))
    assert (code2 == UNVERIFIED).all() and np.isnan(agree2).all()


def test_one_sided_rows_are_unverified_whatever_they_look_like_and_ids_do_not_mix():
    truth = _track(6, speed=3.0)
    n = len(truth)
    gated = np.zeros(n, bool)
    gated[5] = True                                             # track end: only an earlier neighbour
    t = np.arange(n, dtype=float)
    ids = np.full(n, 7)
    # a good moving row (truth corners), a wrong one: no hold-still prediction is made, both are only marked
    for off in (0.0, 40.0):
        det = truth.copy()
        det[5] += off
        code, agree = resolve_gated(t, ids, det, det, ~gated, gated)
        assert code[5] == UNVERIFIED and np.isnan(agree[5]) and (code[:5] == 1).all()
    other = ids.copy()
    other[:5] = 9                                               # the neighbours belong to another marker
    assert resolve_gated(t, other, truth, truth, ~gated, gated)[0][5] == UNVERIFIED
    first = np.zeros(n, bool)
    first[0] = True                                             # track start: only a later neighbour
    assert resolve_gated(t, ids, truth, truth, ~first, first)[0][0] == UNVERIFIED
    assert resolve_gated(np.where(gated, np.nan, t), ids, truth, truth, ~gated, gated)[0][5] == UNVERIFIED


def test_rows_that_are_not_gated_are_untouched_and_rows_with_neither_flag_stay_zero():
    truth = _track(5)
    acc = np.array([True, True, False, True, True])
    gated = np.zeros(5, bool)
    code, agree = resolve_gated(np.arange(5.0), np.full(5, 7), truth, truth, acc, gated)
    assert code.tolist() == [1, 1, 0, 1, 1] and np.isnan(agree).all()
