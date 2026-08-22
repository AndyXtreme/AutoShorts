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
