"""Span arithmetic behind dead-air removal.

Pure functions, no ffmpeg and no models - these are the parts that decide what
gets cut, and getting them wrong shows up as clips that end mid-word.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import silence_cutter as sc


def test_merge_spans_joins_neighbours_within_the_gap():
    assert sc._merge_spans([(0, 1), (1.2, 2)], join_below=0.5) == [(0, 2)]
    assert sc._merge_spans([(0, 1), (1.2, 2)], join_below=0.1) == [(0, 1), (1.2, 2)]


def test_merge_spans_handles_overlap_and_disorder():
    assert sc._merge_spans([(2, 3), (0, 2.5)]) == [(0, 3)]


def test_invert_returns_the_gaps_between_keeps():
    assert sc._invert([(1, 2), (4, 5)], duration=6) == [(0, 1), (2, 4), (5, 6)]


def test_invert_without_keeps_is_the_whole_clip():
    assert sc._invert([], duration=5) == [(0, 5)]


def test_gaps_are_cut_longest_first_and_respect_the_floor():
    gaps = [(0, 2), (5, 11), (20, 23)]
    # 30s clip, floor 25s: only 5s may go, so the 6s gap is skipped and the
    # next longest that still fits is taken.
    chosen = sc._select_gaps_to_cut(gaps, duration=30, min_gap=1.0, floor=25.0)
    assert (5, 11) not in chosen
    assert (20, 23) in chosen


def test_short_gaps_are_left_alone():
    assert sc._select_gaps_to_cut([(0, 0.5)], duration=30, min_gap=1.0, floor=0) == []


def test_edges_keep_some_air():
    """Leading and trailing silence is shortened, not removed."""
    trimmed = sc._keep_edges([(0, 3), (10, 12), (25, 30)], duration=30, padding=0.8)
    assert trimmed[0] == (0, 2.2)          # leading: keep the last 0.8s
    assert trimmed[1] == (10, 12)          # middle: untouched
    assert trimmed[-1] == (25.8, 30)       # trailing: keep the first 0.8s


def test_motion_spans_are_clip_relative():
    times = np.array([100.0, 100.5, 101.0, 101.5])
    scores = np.array([0.0, 2.0, 2.0, 0.0])
    spans = sc._motion_spans(times, scores, offset=100.0, duration=2.0,
                             threshold=1.0, padding=0.0)
    assert spans and spans[0][0] == 0.5
    assert spans[0][1] <= 2.0


def test_motion_spans_without_a_profile():
    assert sc._motion_spans(None, None, 0.0, 10.0, 1.0, 0.0) == []


# --- the stretch before the highlight -------------------------------------
# Measured on Haku Challenge #16, clip 2 (source 301.37s + 31.5s): what the
# voice activity detector and the motion profile kept, in source seconds.
_OFFSET, _DURATION = 301.37, 31.5
_SPEECH = [(301.4, 304.6), (313.1, 315.3), (326.3, 332.7)]
_MOTION = [(310.2, 323.1)]


def _cuts(peak, lead=5.0):
    rel = lambda spans: [(a - _OFFSET, b - _OFFSET) for a, b in spans]
    build_up = sc._build_up_span(peak, _OFFSET, _DURATION, lead)
    keep = sc._merge_spans(rel(_SPEECH) + rel(_MOTION) + build_up, join_below=0.5)
    to_cut = sc._select_gaps_to_cut(sc._invert(keep, _DURATION), _DURATION, 1.0, 8.0)
    return [(round(a + _OFFSET, 1), round(b + _OFFSET, 1))
            for a, b in sc._keep_edges(to_cut, _DURATION, 0.8)]


def test_without_a_peak_the_successful_attempt_is_cut():
    """The defect: 323.1-326.3 is the jump that worked, and it went."""
    assert (323.1, 326.3) in _cuts(peak=None)


def test_the_stretch_before_the_peak_is_kept():
    cuts = _cuts(peak=326.5)
    assert not any(a < 326.3 and b > 323.1 for a, b in cuts)
    # The wait between two failed attempts is still dead air.
    assert (304.6, 310.2) in cuts


def test_the_lead_can_be_switched_off():
    assert _cuts(peak=326.5, lead=0.0) == _cuts(peak=None)


def test_the_build_up_span_stays_inside_the_clip():
    assert sc._build_up_span(2.0, 0.0, 30.0, 4.0) == [(0.0, 2.0)]
    assert sc._build_up_span(50.0, 0.0, 30.0, 4.0) == [(26.0, 30.0)]
    assert sc._build_up_span(-1.0, 0.0, 30.0, 4.0) == []
