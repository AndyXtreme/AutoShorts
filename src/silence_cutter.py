"""Remove dead air from a rendered clip.

The pipeline cuts one contiguous window out of the source video, so any pause
inside that window survives into the short. This module cuts those pauses out
and stitches the remainder back together.

What counts as "dead" is deliberately not just silence: in gameplay the best
visual moment is often the one nobody comments on. A stretch is only removed
when *neither* speech nor noticeable motion happens in it, using the same
motion profile the clip selection already computed.

Subtitles are generated afterwards, from the already-cut video, so no timing
has to be remapped - the transcript simply describes the new timeline.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

Span = Tuple[float, float]


def is_enabled() -> bool:
    return os.getenv("REMOVE_SILENCE", "true").lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logging.warning("Env var %s=%r is not a valid float. Using %s.", name, raw, default)
        return default


def _merge_spans(spans: Sequence[Span], join_below: float = 0.0) -> List[Span]:
    """Merge overlapping spans, and spans separated by less than join_below."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= join_below:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _speech_spans(words_path: Path, padding: float) -> List[Span]:
    """Spans covering spoken words, padded so words are not clipped at the edges."""
    try:
        with open(words_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.warning(f"Could not read word timings for silence cutting: {e}")
        return []

    spans: List[Span] = []
    for segment in data.get("segments", []):
        for word in segment.get("words", []):
            try:
                start = float(word["start"]) - padding
                end = float(word["end"]) + padding
            except (KeyError, TypeError, ValueError):
                continue
            spans.append((max(0.0, start), end))
    return _merge_spans(spans, join_below=padding)


def _motion_spans(
    video_times: Optional[np.ndarray],
    video_scores: Optional[np.ndarray],
    offset: float,
    duration: float,
    threshold: float,
    padding: float,
) -> List[Span]:
    """Spans where the clip is visually busy, in clip-relative seconds.

    video_times/scores are absolute source timestamps from the analysis pass;
    the scores are z-scored over the whole video, so the threshold reads as
    "this many standard deviations above the video's average motion".
    """
    if video_times is None or video_scores is None:
        return []
    if len(video_times) == 0 or len(video_scores) == 0:
        return []

    times = np.asarray(video_times, dtype=float)
    scores = np.asarray(video_scores, dtype=float)
    if times.shape != scores.shape:
        size = min(times.size, scores.size)
        times, scores = times[:size], scores[:size]

    mask = (times >= offset) & (times < offset + duration)
    if not np.any(mask):
        return []

    local_times = times[mask] - offset
    local_scores = scores[mask]

    busy = local_scores >= threshold
    if not np.any(busy):
        return []

    # Turn the boolean series into spans, using the sample spacing as width.
    step = float(np.median(np.diff(local_times))) if local_times.size > 1 else 0.1
    step = max(step, 1e-3)

    spans = [
        (max(0.0, float(t) - padding), min(duration, float(t) + step + padding))
        for t, keep in zip(local_times, busy) if keep
    ]
    return _merge_spans(spans, join_below=step)


def _invert(keep: Sequence[Span], duration: float) -> List[Span]:
    """The gaps between keep-spans, i.e. the candidates for removal."""
    gaps: List[Span] = []
    cursor = 0.0
    for start, end in keep:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _select_gaps_to_cut(gaps: Sequence[Span], duration: float, min_gap: float, floor: float) -> List[Span]:
    """Pick which gaps to remove, longest first, without going below `floor`.

    Cutting every qualifying gap can leave a clip too short to post. Removing
    the longest ones first buys the most silence per second of runtime lost.
    """
    candidates = sorted(
        ((end - start, start, end) for start, end in gaps if end - start >= min_gap),
        reverse=True,
    )
    chosen: List[Span] = []
    remaining = duration
    for length, start, end in candidates:
        if floor > 0 and remaining - length < floor:
            continue
        chosen.append((start, end))
        remaining -= length
    return sorted(chosen)


def _keep_edges(to_cut: Sequence[Span], duration: float, padding: float) -> List[Span]:
    """Leave a little air at the very start and end of the clip.

    Cutting the leading and trailing silence flush makes the clip begin and end
    on the exact frame a word starts or stops, which reads as an abrupt cut.
    Those two gaps are shortened rather than removed.
    """
    if padding <= 0:
        return list(to_cut)

    trimmed: List[Span] = []
    for start, end in to_cut:
        if start <= 0.001:                 # leading silence: keep the tail of it
            start_new, end_new = start, max(start, end - padding)
        elif end >= duration - 0.001:      # trailing silence: keep the head of it
            start_new, end_new = min(end, start + padding), end
        else:
            start_new, end_new = start, end

        if end_new - start_new > 0.05:
            trimmed.append((start_new, end_new))
    return trimmed


def _cut_with_ffmpeg(source: Path, keep: Sequence[Span], destination: Path) -> bool:
    parts = []
    for index, (start, end) in enumerate(keep):
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}];"
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]"
        )
    streams = "".join(f"[v{i}][a{i}]" for i in range(len(keep)))
    filter_complex = ";".join(parts) + f";{streams}concat=n={len(keep)}:v=1:a=1[v][a]"

    ffmpeg = os.getenv("FFMPEG_BINARY", "ffmpeg")
    for encoder in ("hevc_nvenc", "libx264"):
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", encoder, "-b:v", "8M",
            "-c:a", "aac", "-b:a", "192k",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
            return True
        logging.warning(f"Silence cut with {encoder} failed: {result.stderr.strip()[:400]}")
        destination.unlink(missing_ok=True)
    return False


def remove_dead_air(
    clip_path: Path,
    source_offset: float,
    duration: float,
    video_times: Optional[np.ndarray] = None,
    video_scores: Optional[np.ndarray] = None,
    min_clip_length: float = 0.0,
) -> bool:
    """Cut silent, motionless stretches out of `clip_path` in place.

    Returns True if the clip was modified.
    """
    if not is_enabled():
        return False

    min_gap = _env_float("SILENCE_MIN_GAP", 1.0)
    padding = _env_float("SILENCE_PADDING", 0.15)
    motion_threshold = _env_float("SILENCE_MOTION_KEEP", 0.5)

    from subtitle_generator import transcribe_audio, word_timings_path

    with tempfile.TemporaryDirectory() as tmpdir:
        probe_srt = Path(tmpdir) / "probe.srt"
        try:
            transcribe_audio(clip_path, probe_srt)
        except Exception as e:
            logging.warning(f"Silence cutting skipped, transcription failed: {e}")
            return False

        speech = _speech_spans(word_timings_path(probe_srt), padding)
        motion = _motion_spans(
            video_times, video_scores, source_offset, duration, motion_threshold, padding
        )

        keep = _merge_spans(list(speech) + list(motion), join_below=min_gap * 0.5)
        keep = [(max(0.0, s), min(duration, e)) for s, e in keep if e > 0 and s < duration]

        if not keep:
            logging.info("Silence cutting skipped: neither speech nor motion detected.")
            return False

        gaps = _invert(keep, duration)
        to_cut = _select_gaps_to_cut(gaps, duration, min_gap, min_clip_length)
        to_cut = _keep_edges(to_cut, duration, _env_float("SILENCE_EDGE_PADDING", 0.8))
        if not to_cut:
            logging.info("Silence cutting: no gap long enough to remove.")
            return False

        final_keep = _invert(to_cut, duration)
        if not final_keep:
            return False

        removed = sum(end - start for start, end in to_cut)

        # Deliberately beside the clip, not in tmpdir: the output folder is a
        # bind mount while /tmp is the container's own filesystem, and
        # os.replace() across devices raises EXDEV. Same directory also keeps
        # the swap atomic instead of copying a freshly encoded file twice.
        cut_path = clip_path.with_name(f"{clip_path.stem}.cutting{clip_path.suffix}")
        try:
            if not _cut_with_ffmpeg(clip_path, final_keep, cut_path):
                logging.warning("Silence cutting failed; keeping the uncut clip.")
                return False
            os.replace(cut_path, clip_path)
        finally:
            cut_path.unlink(missing_ok=True)

    logging.info(
        f"Silence cut: removed {removed:.1f}s of dead air in {len(to_cut)} place(s), "
        f"{duration:.1f}s -> {duration - removed:.1f}s."
    )
    return True
