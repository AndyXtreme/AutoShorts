"""Subtitle generation using Whisper and PyCaps.

This module provides functionality to:
1. Transcribe video audio using OpenAI Whisper
2. Apply animated subtitles using PyCaps with custom styling
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class SubtitleConfig:
    """Configuration for subtitle styling."""
    
    font_family: str = "Bangers"
    font_size: int = 48
    text_color: str = "#FFFFFF"
    highlight_color: str = "#00ff88"  # Neon green for active word
    shadow_color: str = "#000000"
    shadow_offset: int = 2
    position: str = "bottom"  # "top", "center", "bottom"
    margin_bottom: int = 50
    
    @classmethod
    def from_env(cls) -> "SubtitleConfig":
        """Create config from environment variables."""
        # _env_int rather than int(os.getenv(...)): the dashboard writes an
        # empty string for any value it has no entry for, and int("") raises.
        # SUBTITLE_FONT_SIZE is also allowed to be 0 ("use the template size")
        # in the PyCaps path, which would be a 0px font here.
        font_size = _env_int("SUBTITLE_FONT_SIZE", 48)
        return cls(
            font_family=os.getenv("SUBTITLE_FONT") or "Bangers",
            font_size=font_size if font_size > 0 else 48,
            text_color=os.getenv("SUBTITLE_TEXT_COLOR") or "#FFFFFF",
            highlight_color=os.getenv("SUBTITLE_HIGHLIGHT_COLOR") or "#00ff88",
            shadow_color=os.getenv("SUBTITLE_SHADOW_COLOR") or "#000000",
            shadow_offset=_env_int("SUBTITLE_SHADOW_OFFSET", 2),
            position=os.getenv("SUBTITLE_POSITION") or "bottom",
            margin_bottom=_env_int("SUBTITLE_MARGIN_BOTTOM", 50),
        )


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to the default on empty/invalid input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logging.warning("Env var %s=%r is not a valid int. Using %s.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to the default on empty/invalid input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logging.warning("Env var %s=%r is not a valid float. Using %s.", name, raw, default)
        return default


def is_subtitles_enabled() -> bool:
    """Check if subtitle generation is enabled via env."""
    return os.getenv("ENABLE_SUBTITLES", "true").lower() in ("true", "1", "yes")


def get_whisper_model() -> str:
    """Get the configured Whisper model name."""
    return os.getenv("WHISPER_MODEL", "medium")


def transcribe_audio(video_path: Path, output_srt: Optional[Path] = None) -> Path:
    """Transcribe video audio using Whisper.
    
    Args:
        video_path: Path to video file
        output_srt: Optional path for SRT output. If None, creates alongside video.
        
    Returns:
        Path to generated SRT file
    """
    if output_srt is None:
        output_srt = video_path.with_suffix(".srt")
    
    model_name = get_whisper_model()
    
    logging.info(f"Transcribing audio with Whisper ({model_name})...")
    
    try:
        import whisper

        model = _load_whisper_model(model_name)

        result = None
        if _use_vad():
            result = _transcribe_speech_regions(model, video_path)

        if result is None:
            # condition_on_previous_text=False: with loud game audio the model
            # otherwise lets its own earlier output steer the decoding and drops
            # the tail of a clip. Measured on gameplay footage: 48 words ending
            # at 36.6s with it on, 51 words ending at 41.2s with it off.
            result = model.transcribe(
                str(video_path),
                task="transcribe",
                verbose=False,
                word_timestamps=True,  # Enable word-level timestamps for PyCaps
                condition_on_previous_text=False,
            )

        result = _drop_stock_phrases(result)

        # Generate SRT content
        srt_content = _generate_srt(result)

        # Write SRT file
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logging.info(f"Transcription saved to: {output_srt}")

        # SRT only carries segment boundaries, so the caption renderer would have
        # to spread words evenly across each segment - which drifts audibly out of
        # sync. Keep Whisper's per-word timings in a sidecar for PyCaps to use.
        _write_word_timings(result, word_timings_path(output_srt))
        

        
        return output_srt
        
    except ImportError:
        logging.error("whisper not installed. Run: pip install openai-whisper")
        raise
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise


_WHISPER_MODELS: Dict[str, Any] = {}
_VAD_MODEL: List[Any] = []


def _load_whisper_model(model_name: str):
    """Load a Whisper model once per process.

    A run transcribes every clip, and `medium` is ~1.5GB that otherwise gets
    read from disk and pushed to the GPU again for each call. The model is
    stateless between transcriptions, so one instance serves them all.
    """
    if model_name not in _WHISPER_MODELS:
        import whisper
        logging.info(f"Loading Whisper model '{model_name}' (first use in this process)...")
        _WHISPER_MODELS[model_name] = whisper.load_model(model_name)
    return _WHISPER_MODELS[model_name]


def _load_vad_model():
    """Load Silero VAD once per process. Same reasoning as the Whisper cache."""
    if not _VAD_MODEL:
        from silero_vad import load_silero_vad
        _VAD_MODEL.append(load_silero_vad())
    return _VAD_MODEL[0]


def _use_vad() -> bool:
    return os.getenv("WHISPER_USE_VAD", "true").lower() in ("1", "true", "yes", "on")


def _speech_regions(audio_path: Path) -> Optional[List[tuple]]:
    """Find the stretches that actually contain speech.

    Whisper decodes in 30-second windows. With loud game audio in the same
    window it mis-times quiet speech or discards it as non-speech: on real
    gameplay footage an "Ups" spoken at 0.4s was placed at 3.9s, and a "komm
    her" at 40.8s vanished entirely - both were transcribed correctly once
    handed over in isolation. A voice activity detector finds those stretches
    so each can be transcribed on its own.
    """
    try:
        from silero_vad import read_audio, get_speech_timestamps
    except ImportError:
        logging.warning("silero-vad not installed; transcribing the whole clip at once.")
        return None

    try:
        model = _load_vad_model()
        wav = read_audio(str(audio_path), sampling_rate=16000)
        stamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
    except Exception as e:
        logging.warning(f"Voice activity detection failed ({e}); transcribing the whole clip at once.")
        return None

    if not stamps:
        return []

    padding = _env_float("VAD_PADDING", 0.3)
    merge_gap = _env_float("VAD_MERGE_GAP", 0.5)
    max_region = _env_float("VAD_MAX_REGION", 10.0)
    min_region = _env_float("VAD_MIN_REGION", 0.5)

    regions = []
    for stamp in stamps:
        start = max(0.0, float(stamp["start"]) - padding)
        end = float(stamp["end"]) + padding
        # Merge neighbours: a region per breath would give Whisper too little
        # context to recognise words reliably. But stop merging before the
        # region gets long - chained into a 15s block the model starts
        # repeating itself ("Aber eigentlich... Aber eigentlich") and its word
        # alignment drifts by a second or more.
        if (regions
                and start - regions[-1][1] <= merge_gap
                and (max_region <= 0 or end - regions[-1][0] <= max_region)):
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))

    # A very short region is usually a game noise the detector mistook for
    # speech. Transcribed on its own it invites hallucination - one second of
    # Halo background produced Korean characters.
    kept = [r for r in regions if r[1] - r[0] >= min_region]
    if len(kept) < len(regions):
        logging.info(f"Skipped {len(regions) - len(kept)} speech region(s) shorter than {min_region}s.")
    return kept


# Scripts that cannot legitimately appear in a Latin-script transcript. Whisper
# falls back to these when it hallucinates on noise - one second of game
# background produced Korean characters in a German recording.
_FOREIGN_SCRIPT_RANGES = (
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x04FF),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana / Katakana
    (0x3400, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility
)

_LATIN_SCRIPT_LANGUAGES = {
    "de", "en", "es", "fr", "it", "pt", "nl", "pl", "sv", "da", "no", "fi",
    "cs", "sk", "sl", "hr", "hu", "ro", "tr", "id", "ms", "vi", "ca", "gl", "et", "lv", "lt",
}


def _has_foreign_script(text: str) -> bool:
    return any(
        any(low <= ord(char) <= high for low, high in _FOREIGN_SCRIPT_RANGES)
        for char in text
    )


def _drop_foreign_script(segments: list, language: Optional[str]) -> list:
    """Remove words written in a script the target language does not use.

    Only meaningful once the language is known and Latin-script: a German
    transcript never legitimately contains Hangul or CJK, so such a word is a
    hallucination on noise rather than something that was said.
    """
    if not language or language.lower() not in _LATIN_SCRIPT_LANGUAGES:
        return segments
    if os.getenv("WHISPER_DROP_FOREIGN_SCRIPT", "true").lower() not in ("1", "true", "yes", "on"):
        return segments

    cleaned = []
    dropped = 0
    for segment in segments:
        words = [w for w in (segment.get("words") or []) if not _has_foreign_script(str(w.get("word", "")))]
        dropped += len(segment.get("words") or []) - len(words)
        if not words:
            continue
        text = segment.get("text", "")
        if _has_foreign_script(text):
            text = "".join(str(w.get("word", "")) for w in words)
        cleaned.append({**segment, "words": words, "text": text})

    if dropped:
        logging.info(f"Dropped {dropped} hallucinated word(s) written in another script.")
    return cleaned


def _detect_language(model, full_wav: Path, regions, tmpdir, ffmpeg) -> Optional[str]:
    """Determine the spoken language once, for the whole clip.

    Uses the longest speech region rather than the opening seconds - the start
    of a clip is often game audio only, and language detection on noise is a
    coin flip. Set WHISPER_LANGUAGE to skip detection entirely.
    """
    configured = os.getenv("WHISPER_LANGUAGE", "").strip()
    if configured:
        logging.info(f"Transcribing as '{configured}' (WHISPER_LANGUAGE).")
        return configured

    import subprocess

    try:
        import whisper

        start, end = max(regions, key=lambda r: r[1] - r[0])
        sample = Path(tmpdir) / "lang.wav"
        cut = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", str(full_wav), str(sample)],
            capture_output=True, text=True,
        )
        source = sample if cut.returncode == 0 and sample.exists() else full_wav

        audio = whisper.pad_or_trim(whisper.load_audio(str(source)))
        mel = whisper.log_mel_spectrogram(audio, getattr(model.dims, "n_mels", 80)).to(model.device)
        _, probabilities = model.detect_language(mel)
        language = max(probabilities, key=probabilities.get)
        logging.info(f"Detected language for the whole clip: {language}")
        return language
    except Exception as e:
        logging.warning(f"Language detection failed ({e}); each region will decide for itself.")
        return None


def detect_speech_spans(media_path: Path) -> Optional[List[Tuple[float, float]]]:
    """Where speech happens in a media file, in seconds, without transcribing.

    The silence cutter only needs to know *whether* someone is talking, not
    what was said, and it used to answer that by running a full Whisper pass
    over every clip - a second transcription on top of the one the subtitles
    already need. Silero decides the same question in a fraction of the time.

    Returns None when the VAD is unavailable, so callers can fall back.
    """
    ffmpeg = os.getenv("FFMPEG_BINARY", "ffmpeg")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = Path(tmpdir) / "vad.wav"
        extract = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(media_path),
             "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            capture_output=True, text=True,
        )
        if extract.returncode != 0 or not wav.exists():
            logging.warning("Could not extract audio for speech detection.")
            return None
        return _speech_regions(wav)


def _transcribe_speech_regions(model, video_path: Path) -> Optional[dict]:
    """Transcribe each speech region separately and stitch the results together.

    Returns a result shaped like Whisper's own, with all times mapped back onto
    the clip's timeline, or None if the VAD is unavailable.
    """
    import subprocess
    import tempfile

    ffmpeg = os.getenv("FFMPEG_BINARY", "ffmpeg")

    with tempfile.TemporaryDirectory() as tmpdir:
        full_wav = Path(tmpdir) / "full.wav"
        extract = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", str(full_wav)],
            capture_output=True, text=True,
        )
        if extract.returncode != 0 or not full_wav.exists():
            logging.warning("Could not extract audio for VAD; transcribing the whole clip at once.")
            return None

        regions = _speech_regions(full_wav)
        if regions is None:
            return None
        if not regions:
            logging.info("No speech detected in this clip.")
            return {"text": "", "segments": []}

        # Detect the language once and impose it on every region. Left to decide
        # per region, Whisper guesses from a second or two of audio and gets it
        # wrong: on one clip three of eight regions came back as English in an
        # otherwise German recording, producing nonsense words.
        language = _detect_language(model, full_wav, regions, tmpdir, ffmpeg)

        segments = []
        for index, (start, end) in enumerate(regions):
            piece = Path(tmpdir) / f"part{index}.wav"
            cut = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}",
                 "-to", f"{end:.3f}", "-i", str(full_wav), str(piece)],
                capture_output=True, text=True,
            )
            if cut.returncode != 0 or not piece.exists():
                continue

            piece_result = model.transcribe(
                str(piece),
                task="transcribe",
                verbose=False,
                word_timestamps=True,
                condition_on_previous_text=False,
                language=language,
            )

            for segment in piece_result.get("segments", []):
                words = []
                for word in segment.get("words") or []:
                    words.append({
                        **word,
                        "start": float(word["start"]) + start,
                        "end": float(word["end"]) + start,
                    })
                segments.append({
                    **segment,
                    "start": float(segment.get("start", 0.0)) + start,
                    "end": float(segment.get("end", 0.0)) + start,
                    "words": words,
                })

    segments = _drop_foreign_script(segments, language)
    segments.sort(key=lambda s: s["start"])
    text = " ".join(s.get("text", "").strip() for s in segments).strip()
    logging.info(f"Transcribed {len(regions)} speech region(s) separately.")
    return {"text": text, "segments": segments}


# Lines Whisper invents on noise or silence because they close countless
# subtitle files in its training data - broadcaster credits, mostly. Nobody
# says them while playing. Measured on an eight-second piece of a German
# recording: one second of engine noise came back as "Untertitel im Auftrag
# des ZDF für funk, 2017". Each pattern needs a credit's shape - a sender with
# a year, "Untertitel im Auftrag ..." - so a sentence that merely contains the
# word "Untertitel" is left alone.
_STOCK_PHRASES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\buntertitel(ung)?\s+(im auftrag|der|des|von|durch)\b",
    r"amara\.org",
    r"\b(zdf|ard|wdr|swr|ndr|mdr|rbb|funk)\b.{0,40}\b(19|20)\d{2}\b",
    r"(\bcopyright\b|©).{0,40}\b(19|20)\d{2}\b",
    r"\bsubtitles?\s+(by|from|provided)\b",
    r"\btranscri(bed|ption)\s+by\b",
))


def _is_stock_phrase(text: str) -> bool:
    return any(pattern.search(text) for pattern in _STOCK_PHRASES)


def _drop_stock_phrases(result: dict) -> dict:
    """Remove segments that are a subtitle credit Whisper made up, not speech."""
    if os.getenv("WHISPER_DROP_STOCK_PHRASES", "true").lower() not in ("1", "true", "yes", "on"):
        return result
    segments = result.get("segments") or []
    kept = [s for s in segments if not _is_stock_phrase(str(s.get("text", "")))]
    if len(kept) == len(segments):
        return result
    for segment in segments:
        if segment not in kept:
            logging.info(f"Dropped a made-up subtitle credit: {str(segment.get('text', '')).strip()!r}")
    text = " ".join(str(s.get("text", "")).strip() for s in kept).strip()
    return {**result, "segments": kept, "text": text}


def word_timings_path(srt_path: Path) -> Path:
    """Sidecar path holding Whisper's per-word timings for a given SRT."""
    return srt_path.with_suffix(".words.json")


def _write_word_timings(whisper_result: dict, output_path: Path) -> Optional[Path]:
    """Persist Whisper's word-level timings alongside the SRT.

    Whisper is called with word_timestamps=True, but _generate_srt keeps only the
    segment boundaries. Writing the word timings out lets the caption renderer
    highlight each word when it is actually spoken instead of assuming every word
    in a segment takes the same amount of time.

    Returns the written path, or None if Whisper produced no word-level data.
    """
    segments = []
    for segment in whisper_result.get("segments", []):
        words = []
        for word in segment.get("words") or []:
            text = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")
            if not text or start is None or end is None:
                continue
            words.append({"text": text, "start": float(start), "end": float(end)})

        if not words:
            continue

        segments.append({
            "start": float(segment.get("start", words[0]["start"])),
            "end": float(segment.get("end", words[-1]["end"])),
            "words": words,
        })

    if not segments:
        logging.warning("Whisper returned no word-level timings; captions will use interpolated timing.")
        return None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False)

    logging.info(f"Word timings saved to: {output_path}")
    return output_path


def _generate_srt(whisper_result: dict) -> str:
    """Convert Whisper result to SRT format."""
    segments = whisper_result.get("segments", [])
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if not text:
            continue
        
        start_ts = _format_timestamp(start)
        end_ts = _format_timestamp(end)
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def apply_pycaps_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    config: Optional[SubtitleConfig] = None,
    output_codec: Optional[str] = None
) -> bool:
    """Apply animated subtitles using PyCaps (TemplateLoader API).
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video with subtitles
        config: Subtitle styling configuration
        output_codec: Codec the caption renderer should write, instead of its
            hardcoded libx264. None leaves it alone.

    Returns:
        True if successful
    """
    if config is None:
        config = SubtitleConfig.from_env()
    
    logging.info(f"Applying PyCaps subtitles to: {video_path.name}")
    
    # Run PyCaps in a separate process to avoid Playwright Sync API conflict with asyncio
    # and to ensure a clean environment.
    import multiprocessing
    
    # We need to pass strings, not Path objects, to be safe across processes
    ctx = multiprocessing.get_context("spawn")  # Use spawn for better compatibility
    p = ctx.Process(
        target=_run_pycaps_worker,
        args=(str(video_path), str(srt_path), str(output_path), config, output_codec)
    )
    p.start()
    p.join()
    
    # Check for output - PyCaps derives output name from SRT file, not video file
    # So when we pass extended video, it still outputs based on original SRT name
    # SRT path = scene-0.srt → PyCaps outputs scene-0_sub.mp4
    srt_stem = srt_path.stem  # e.g., "scene-0" from "scene-0.srt"
    pycaps_output = srt_path.parent / f"{srt_stem}_sub.mp4"
    
    # Also check based on video path (for non-extended cases)
    pycaps_output_from_video = video_path.with_stem(video_path.stem + "_sub").with_suffix(".mp4")
    
    if p.exitcode == 0:
        # Check our expected path first
        if output_path.exists():
            logging.info(f"PyCaps subtitled video saved to: {output_path}")
            return True
        # Check PyCaps output based on SRT name (most common case when video was extended)
        elif pycaps_output.exists():
            import shutil
            shutil.move(str(pycaps_output), str(output_path))
            logging.info(f"PyCaps subtitled video (from SRT-based _sub.mp4) saved to: {output_path}")
            return True
        # Check PyCaps output based on video name
        elif pycaps_output_from_video.exists():
            import shutil
            shutil.move(str(pycaps_output_from_video), str(output_path))
            logging.info(f"PyCaps subtitled video (from video-based _sub.mp4) saved to: {output_path}")
            return True
        # Check fallback locations
        elif _check_for_fallback_output(video_path, output_path):
            logging.info(f"PyCaps subtitled video saved to: {output_path}")
            return True
    
    logging.warning("PyCaps process failed or produced no output. Falling back to FFmpeg.")
    return _apply_ffmpeg_subtitles(video_path, srt_path, output_path, config)


def _check_for_fallback_output(video_path: Path, output_path: Path) -> bool:
    """Check if PyCaps saved to a default filename and move it."""
    import time
    
    # Check current directory
    cwd = Path.cwd()
    recent_outputs = sorted(
        cwd.glob("output_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for recent in recent_outputs[:3]:
        if time.time() - recent.stat().st_mtime < 60:
            import shutil
            shutil.move(str(recent), str(output_path))
            return True
            
    # Check video parent directory
    parent_outputs = sorted(
        video_path.parent.glob("output_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for recent in parent_outputs[:3]:
        if time.time() - recent.stat().st_mtime < 60:
            import shutil
            shutil.move(str(recent), str(output_path))
            return True
            
    return False


@contextmanager
def _caption_encoder(codec: Optional[str], quality: Optional[str]):
    """Make movielite's writer encode in `codec` instead of libx264.

    The caption stage decodes every frame and writes a new file, so its
    encoder decides what the viewer actually gets. movielite hardcodes
    libx264 ("We will soon support other encodings"), which used to make the
    finished clip H.264 whatever the source was, and cost a whole extra
    generation to put right afterwards.

    Replacing VideoWriter._render_range would mean carrying a copy of its
    render loop. Swapping the encoder arguments in the command it hands to
    Popen is the smaller surface: everything around it - part splitting,
    concat, audio muxing - copies the video stream and is untouched.

    A command that does not look the way it did when this was written is
    passed through unchanged, and video_io's final pass still puts the clip
    into the right codec, one generation later.
    """
    if not codec:
        yield False
        return
    try:
        from movielite.core import video_writer
    except ImportError:
        yield False
        return

    from video_io import video_encoder_args

    replacement = video_encoder_args(codec, quality)
    original_popen = video_writer.subprocess.Popen
    swapped = []

    def popen(args, *rest, **kwargs):
        # video_writer.subprocess is the stdlib module itself, so this hook
        # sees every Popen the worker makes while the pipeline runs. Match the
        # writer's command precisely - raw frames in on a pipe, libx264 out -
        # and let everything else through untouched.
        if (
            isinstance(args, (list, tuple))
            and "rawvideo" in args
            and "libx264" in args
            and "pipe:0" in args
            and "-c:v" in args
        ):
            args = list(args)
            start = args.index("-c:v")
            # The encoder block runs from -c:v up to the muxer flags that
            # follow it; bounded that way rather than by a token count, so a
            # changed preset or crf option does not shift the slice.
            end = args.index("-movflags") if "-movflags" in args else start + 6
            args = args[:start] + list(replacement) + args[end:]
            swapped.append(replacement[1])
        return original_popen(args, *rest, **kwargs)

    video_writer.subprocess.Popen = popen
    try:
        yield swapped
    finally:
        video_writer.subprocess.Popen = original_popen


class _Cv2WithFfmpegDecode:
    """cv2 as movielite's VideoClip sees it, with VideoCapture routed through
    video_io.open_capture and everything else (resize, cvtColor, the CAP_PROP
    constants) passed through to the real module."""

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def VideoCapture(self, path, *args, **kwargs):  # noqa: N802 - cv2's name
        from video_io import open_capture

        return open_capture(path)


@contextmanager
def _caption_decoder():
    """Let movielite read clips that OpenCV cannot decode.

    movielite decodes the clip it puts captions on with cv2.VideoCapture, and
    the FFmpeg inside the opencv-python wheel has no AV1 decoder. Every read
    failed, movielite substituted black frames, and the result was a valid,
    tiny AV1 file of captions on black. It used to work only because the
    render wrote HEVC, which put a generation of another codec into every AV1
    Short. Patched on the module movielite reads it from, for the duration of
    the render.
    """
    try:
        from movielite.video import video_clip
    except ImportError:
        yield
        return

    original = video_clip.cv2
    video_clip.cv2 = _Cv2WithFfmpegDecode(original)
    try:
        yield
    finally:
        video_clip.cv2 = original


def _run_pycaps_worker(video_path_str: str, srt_path_str: str, output_path_str: str, config: SubtitleConfig, output_codec: Optional[str] = None):
    """Worker function to run PyCaps in a separate process."""
    # Re-import necessary modules in the spawned process
    import sys
    import os
    
    try:
        from pycaps import CapsPipelineBuilder
        from pycaps.transcriber import AudioTranscriber
        from pycaps.common import Document, Segment, Line, Word, TimeFragment, Tag
        from pycaps.template import TemplateLoader
        
        # Define SRTTranscriber locally to ensure it's available in the worker process
        class SRTTranscriber(AudioTranscriber):
            def __init__(self, srt_path):
                self.srt_path = srt_path
                
            def _build_word(self, text, start, end, index, total):
                """Create a Word carrying the structure tags PyCaps templates rely on.

                No trailing space is appended: Line.get_text() already joins words
                with ' ', and the renderer rebuilds the DOM with text.split(' ').
                A trailing space therefore produces double spaces, which yields one
                empty <span> per word, shifts every word-N-in-line class by one and
                leaves the last word without a class at all - the highlight then
                lands on the wrong word. Word spacing comes from the template CSS.
                """
                word = Word(text=text, time=TimeFragment(start=start, end=end))
                if index == 0:
                    word.structure_tags.add(Tag(name="first-word-in-segment"))
                    word.structure_tags.add(Tag(name="first-word-in-line"))
                if index == total - 1:
                    word.structure_tags.add(Tag(name="last-word-in-segment"))
                    word.structure_tags.add(Tag(name="last-word-in-line"))
                return word

            def _document_from_word_timings(self, path) -> Optional[Document]:
                """Build the document from Whisper's per-word timings, if available."""
                if not os.path.exists(path):
                    return None
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Could not read word timings {path}: {e}")
                    return None

                # A caption block stays on screen from its first word to its
                # last, so a block that spans a speech pause shows words seconds
                # before they are spoken. The character splitter downstream only
                # cuts by length and never by time, so split on gaps here.
                gap_limit = _env_float("SUBTITLE_SPLIT_GAP", 0.7)

                document = Document()
                word_count = 0
                for entry in data.get("segments", []):
                    words = entry.get("words") or []
                    if not words:
                        continue

                    groups: List[list] = []
                    current: list = []
                    for w in words:
                        if current and gap_limit > 0:
                            if float(w["start"]) - float(current[-1]["end"]) > gap_limit:
                                groups.append(current)
                                current = []
                        current.append(w)
                    if current:
                        groups.append(current)

                    for group in groups:
                        spans = []
                        cursor = 0.0
                        for i, w in enumerate(group):
                            start = float(w["start"])
                            end = float(w["end"])
                            # Whisper emits zero-length spans for short words, and
                            # sometimes several in a row carry the exact same
                            # timestamp. Left as is they would all light up at
                            # once. Give each one room up to the next word, and
                            # keep them in sequence so the highlight never runs
                            # backwards or marks two words at the same time.
                            start = max(start, cursor)
                            if end <= start:
                                following = float(group[i + 1]["start"]) if i + 1 < len(group) else start + 0.30
                                end = min(start + 0.30, max(start + 0.08, following))
                            cursor = end
                            spans.append((w["text"], start, end))

                        segment_time = TimeFragment(start=spans[0][1], end=spans[-1][2])
                        segment = Segment(time=segment_time)
                        line = Line(time=segment_time)
                        segment.lines.add(line)

                        for i, (text, start, end) in enumerate(spans):
                            line.words.add(self._build_word(text, start, end, i, len(spans)))
                            word_count += 1

                        line.structure_tags.add(Tag(name="last-line-in-segment"))
                        segment.structure_tags.add(Tag(name="segment"))
                        document.segments.add(segment)

                if not document.segments:
                    return None

                print(f"Using Whisper word timings: {len(document.segments)} segments, {word_count} words.")
                return document

            def transcribe(self, audio_path: str) -> Document:
                # Prefer real per-word timings; fall back to the SRT (which only
                # has segment boundaries and needs the words spread evenly).
                from_words = self._document_from_word_timings(
                    os.path.splitext(self.srt_path)[0] + ".words.json"
                )
                if from_words is not None:
                    return from_words

                print("No word timings found; falling back to interpolated SRT timing.")
                document = Document()
                if not os.path.exists(self.srt_path):
                    print(f"SRT file not found: {self.srt_path}")
                    return document

                try:
                    with open(self.srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Normalize newlines
                    content = content.replace('\r\n', '\n').replace('\r', '\n')
                    
                    def parse_time(t_str):
                        t_str = t_str.strip().replace(',', '.')
                        parts = t_str.split(':')
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

                    import re
                    # Regex to find blocks: ID, Time, Text
                    # Matches:
                    # 1
                    # 00:00:01,000 --> 00:00:02,000
                    # Text content
                    block_pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\d+\s*\n|\Z)', re.DOTALL)
                    
                    matches = list(block_pattern.finditer(content))
                    print(f"Parsing SRT: Found {len(matches)} blocks in {self.srt_path}")
                    
                    for match in matches:
                        idx = match.group(1)
                        start_str = match.group(2)
                        end_str = match.group(3)
                        text = match.group(4).strip()
                        
                        if not text: continue
                        
                        start = parse_time(start_str)
                        end = parse_time(end_str)
                        
                        segment_time = TimeFragment(start=start, end=end)
                        segment = Segment(time=segment_time)
                        line = Line(time=segment_time)
                        segment.lines.add(line)
                        
                        # Pre-process words to merge detached emojis
                        # "BETRAYAL" + "🗡️💔" -> "BETRAYAL 🗡️💔"
                        # This prevents emojis from getting their own time slice and lingering
                        raw_words = text.split()
                        words = []
                        if raw_words:
                            words.append(raw_words[0])
                            for w in raw_words[1:]:
                                # Check if word is likely just emojis/symbols (non-alphanumeric and high unicode)
                                # isalnum() handles "café" correctly as True, so we skip merging it.
                                # Emojis "🗡️" returns isalnum() -> False.
                                is_symbol_or_emoji = (not w.isalnum()) and any(ord(c) > 2000 for c in w)
                                
                                if is_symbol_or_emoji and words:
                                    words[-1] += " " + w
                                else:
                                    words.append(w)
                        
                        if not words: continue
                        
                        # Interpolate word timings
                        duration = end - start
                        word_duration = duration / len(words)
                        
                        current_time = start
                        for i, w_text in enumerate(words):
                            w_start = current_time
                            w_end = current_time + word_duration
                            # Ensure no overlap issues or zero duration
                            if w_end > end: w_end = end

                            line.words.add(
                                self._build_word(w_text, w_start, w_end, i, len(words))
                            )
                            current_time += word_duration

                        # Explicitly tag the line
                        line.structure_tags.add(Tag(name="last-line-in-segment"))
                        segment.structure_tags.add(Tag(name="segment"))
                            
                        document.segments.add(segment)
                        
                    print(f"SRTTranscriber populated {len(document.segments)} segments.")
                    
                except Exception as e:
                    print(f"Error parsing SRT: {e}")
                    import traceback
                    traceback.print_exc()
                
                return document

        # Get template from environment or use default
        template_name = os.getenv("PYCAPS_TEMPLATE", "hype")
        
        # Use native PyCaps template styling (no custom CSS overrides)
        # The template handles fonts, colors, animations automatically
        print(f"Loading PyCaps template: {template_name}")
        builder = TemplateLoader(template_name).with_input_video(video_path_str).load(should_build_pipeline=False)
        
        # --- Caption layout -------------------------------------------------
        # The template ships its own layout and splitters (hype: max 2 lines,
        # limit_by_chars 10-15). Expose both so they can be tuned per channel
        # without editing the template JSON inside site-packages.
        #
        # PYCAPS_KEEP_SPLITTERS=false drops splitting entirely: a whole Whisper
        # block (often 10s of speech) then lands on screen at once, which is
        # 4-5 lines covering a third of a 9:16 frame. Exact SRT block boundaries,
        # bad readability.
        from pycaps import (
            SubtitleLayoutOptions,
            LimitByCharsSplitter,
            SplitIntoSentencesSplitter,
        )
        from pycaps.layout import VerticalAlignment, VerticalAlignmentType, TextOverflowStrategy
        from pycaps.transcriber.splitter.base_segment_splitter import BaseSegmentSplitter

        class SplitOnPauseSplitter(BaseSegmentSplitter):
            """Break a caption wherever the speaker pauses.

            A caption is on screen from its first word to its last, so a block
            spanning a pause shows words seconds before they are spoken. None of
            the built-in splitters look at time: the sentence splitter groups by
            punctuation and happily merges across a two-second silence, and the
            character splitter only counts letters. This one runs last and undoes
            those merges wherever the gap is audible.
            """

            def __init__(self, gap: float):
                self._gap = gap

            def split(self, document) -> None:
                new_segments = []
                for segment in document.segments:
                    words = list(segment.lines[0].words) if segment.lines else []
                    if len(words) < 2:
                        new_segments.append(segment)
                        continue

                    groups = []
                    current = [words[0]]
                    for previous, word in zip(words, words[1:]):
                        if word.time.start - previous.time.end > self._gap:
                            groups.append(current)
                            current = []
                        current.append(word)
                    groups.append(current)

                    if len(groups) == 1:
                        new_segments.append(segment)
                        continue

                    for group in groups:
                        time = TimeFragment(start=group[0].time.start, end=group[-1].time.end)
                        new_segment = Segment(time=time)
                        new_line = Line(time=time)
                        new_line.words.set_all(group)
                        new_line.structure_tags.add(Tag(name="last-line-in-segment"))
                        new_segment.lines.add(new_line)
                        new_segment.structure_tags.add(Tag(name="segment"))
                        new_segments.append(new_segment)

                document.segments.set_all(new_segments)

        pipeline_obj = getattr(builder, "_caps_pipeline", None)
        keep_splitters = os.getenv("PYCAPS_KEEP_SPLITTERS", "true").lower() in ("true", "1", "yes")

        if not keep_splitters:
            if pipeline_obj is not None and hasattr(pipeline_obj, "_segment_splitters"):
                pipeline_obj._segment_splitters = []
                print("Segment splitters cleared - whole SRT blocks shown at once.")
        else:
            max_chars = _env_int("SUBTITLE_MAX_CHARS", 15)
            min_chars = _env_int("SUBTITLE_MIN_CHARS", 10)
            if min_chars > max_chars:
                print(f"SUBTITLE_MIN_CHARS ({min_chars}) > SUBTITLE_MAX_CHARS ({max_chars}); using max for both.")
                min_chars = max_chars

            # Replace rather than append: the template's own char limits would
            # otherwise still apply and win over the configured ones.
            if pipeline_obj is not None and hasattr(pipeline_obj, "_segment_splitters"):
                pipeline_obj._segment_splitters = []
            builder.add_segment_splitter(SplitIntoSentencesSplitter())
            builder.add_segment_splitter(LimitByCharsSplitter(max_chars, min_chars))

            # Last, so it also undoes merges the two above made across a pause.
            split_gap = _env_float("SUBTITLE_SPLIT_GAP", 0.7)
            if split_gap > 0:
                builder.add_segment_splitter(SplitOnPauseSplitter(split_gap))
            print(f"Caption splitting: {min_chars}-{max_chars} chars, break on {split_gap}s pause.")

        max_lines = _env_int("SUBTITLE_MAX_LINES", 2)
        min_lines = _env_int("SUBTITLE_MIN_LINES", 1)
        if min_lines > max_lines:
            min_lines = max_lines

        align_name = os.getenv("SUBTITLE_VERTICAL_ALIGN", "bottom").strip().lower()
        try:
            align = VerticalAlignmentType(align_name)
        except ValueError:
            print(f"Unknown SUBTITLE_VERTICAL_ALIGN={align_name!r}; falling back to bottom.")
            align = VerticalAlignmentType.BOTTOM

        offset = _env_float("SUBTITLE_VERTICAL_OFFSET", -0.1)
        offset = max(-1.0, min(1.0, offset))
        width_ratio = _env_float("SUBTITLE_WIDTH_RATIO", 0.85)
        width_ratio = max(0.05, min(1.0, width_ratio))

        # Caption size. Templates set their own font-size (hype: 24px), which is
        # then rendered at the renderer's device scale - so the on-screen text is
        # roughly twice the CSS value on a 1080x1920 output.
        #
        # Padding and the outline are re-expressed in em so they grow with the
        # text; leaving them in px would give big captions a hairline outline.
        # The ratios are the hype template's own (3px/24px, 5px/24px, 2px/24px).
        font_size = _env_int("SUBTITLE_FONT_SIZE", 0)
        if font_size > 0:
            builder.add_css_content(
                ".word {"
                f" font-size: {font_size}px;"
                " padding: 0.125em 0.208em;"
                " text-shadow:"
                " -0.083em -0.083em 0 #000,"
                " 0.083em -0.083em 0 #000,"
                " -0.083em 0.083em 0 #000,"
                " 0.083em 0.083em 0 #000,"
                " 0.125em 0.125em 0.208em rgba(0,0,0,0.5);"
                "}"
            )
            print(f"Caption font size: {font_size}px (template default is 24px for 'hype').")

        # What gives when the text does not fit: by default PyCaps adds another
        # line, so "max lines" is only a target - a high char limit still spills
        # onto extra lines. "exceed_width" keeps the line count instead and lets
        # the last line run wider than the width ratio.
        overflow_name = os.getenv("SUBTITLE_OVERFLOW", "exceed_lines").strip().lower()
        try:
            overflow = TextOverflowStrategy(overflow_name)
        except ValueError:
            print(f"Unknown SUBTITLE_OVERFLOW={overflow_name!r}; falling back to exceed_lines.")
            overflow = TextOverflowStrategy.EXCEED_MAX_NUMBER_OF_LINES

        builder.with_layout_options(SubtitleLayoutOptions(
            max_number_of_lines=max_lines,
            min_number_of_lines=min_lines,
            max_width_ratio=width_ratio,
            on_text_overflow_strategy=overflow,
            vertical_align=VerticalAlignment(align=align, offset=offset),
        ))
        print(f"Caption layout: {min_lines}-{max_lines} lines, width {width_ratio}, "
              f"{align.value} {offset:+}, overflow={overflow.value}.")

        # EXPLICITLY tell PyCaps to use our SRT file instead of transcribing audio
        # rather than letting it run its own transcription over the game audio
        print(f"Using custom SRTTranscriber with: {srt_path_str}")
        builder = builder.with_custom_audio_transcriber(SRTTranscriber(str(srt_path_str)))

        builder = (
            builder
            .with_output_video(output_path_str)
        )

        # The caption renderer re-encodes every frame, so whatever it does is
        # the quality the viewer actually gets - the careful CQ 23 render
        # upstream is decoded and thrown away here. Its default is CRF 21 at
        # preset veryfast; "high" (CRF 19, preset fast) costs a little render
        # time and removes the visible mush in fast gameplay motion.
        quality_name = os.getenv("SUBTITLE_VIDEO_QUALITY", "high").strip().lower()
        try:
            from pycaps import VideoQuality

            builder = builder.with_video_quality(VideoQuality(quality_name))
            print(f"Caption render quality: {quality_name}.")
        except ValueError:
            valid = ", ".join(q.value for q in VideoQuality)
            print(f"Unknown SUBTITLE_VIDEO_QUALITY={quality_name!r}; valid: {valid}. Using the template default.")
        except ImportError:
            print("This pycaps build has no VideoQuality; using the template default.")
        
        # Build pipeline
        pipeline = builder.build()

        with _caption_encoder(output_codec, quality_name) as swapped, _caption_decoder():
            pipeline.run()
        if output_codec and not swapped:
            print(
                f"Caption renderer kept its own encoder; the clip will be "
                f"re-encoded to {output_codec} afterwards."
            )
        
    except Exception as e:
        # Use print because logging might not be configured in child process
        print(f"PyCaps worker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _apply_ffmpeg_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    config: SubtitleConfig
) -> bool:
    """Fallback: Apply subtitles using FFmpeg (less fancy, but reliable).
    
    Uses the ASS subtitle filter for styled text with the Bangers font.
    """
    logging.info("Using FFmpeg subtitle burn-in (fallback mode)")
    
    # Build FFmpeg subtitle filter with styling
    # Note: FFmpeg uses a different syntax for fonts and colors
    font_style = f"FontName={config.font_family},FontSize={config.font_size}"
    color_hex = config.text_color.lstrip("#")
    outline_hex = config.shadow_color.lstrip("#")
    highlight_hex = config.highlight_color.lstrip("#")
    
    # Convert SRT to styled ASS for better control
    ass_path = srt_path.with_suffix(".ass")
    _convert_srt_to_ass(srt_path, ass_path, config)
    
    # Use the ASS file for subtitle burn-in
    subtitle_filter = f"ass='{ass_path}'"
    
    # Kept in the clip's own codec, at the same quality setting the main
    # caption renderer uses - this replaces that render, it is not an extra one.
    from video_io import codec_of, video_encoder_args

    quality = os.getenv("SUBTITLE_VIDEO_QUALITY", "high").strip().lower()
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", subtitle_filter,
        *video_encoder_args(codec_of(video_path), quality),
        "-c:a", "copy",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"FFmpeg subtitle burn-in failed: {result.stderr[:200]}")
        return False
    
    # Clean up temp ASS file
    try:
        ass_path.unlink()
    except Exception:
        pass
    
    return output_path.exists()


def _convert_srt_to_ass(srt_path: Path, ass_path: Path, config: SubtitleConfig) -> None:
    """Convert SRT to ASS format with custom styling."""
    
    # Read SRT content
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()
    
    # Parse SRT
    segments = []
    current_segment = {}
    
    for line in srt_content.strip().split("\n"):
        line = line.strip()
        
        if not line:
            if current_segment:
                segments.append(current_segment)
                current_segment = {}
        elif "-->" in line:
            parts = line.split(" --> ")
            current_segment["start"] = _srt_to_ass_time(parts[0])
            current_segment["end"] = _srt_to_ass_time(parts[1])
        elif not line.isdigit():
            current_segment["text"] = current_segment.get("text", "") + line + " "
    
    if current_segment:
        segments.append(current_segment)
    
    # Generate ASS content
    text_color = _hex_to_ass_color(config.text_color)
    outline_color = _hex_to_ass_color(config.shadow_color)
    
    ass_header = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.font_family},{config.font_size},{text_color},&H000000FF,{outline_color},&H00000000,1,0,0,0,100,100,0,0,1,{config.shadow_offset},0,2,10,10,{config.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    ass_lines = [ass_header]
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            # Use uppercase for impact (common in short-form content)
            text = text.upper()
            ass_lines.append(
                f"Dialogue: 0,{seg['start']},{seg['end']},Default,,0,0,0,,{text}"
            )
    
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))


def _srt_to_ass_time(srt_time: str) -> str:
    """Convert SRT timestamp to ASS format."""
    # SRT: 00:00:01,500 -> ASS: 0:00:01.50
    srt_time = srt_time.strip().replace(",", ".")
    parts = srt_time.split(":")
    hours = int(parts[0])
    minutes = parts[1]
    seconds = parts[2][:5]  # Truncate to 2 decimal places
    return f"{hours}:{minutes}:{seconds}"


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert hex color to ASS format (&HAABBGGRR)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        # ASS uses BGR order with alpha prefix
        return f"&H00{b}{g}{r}"
    return "&H00FFFFFF"


def generate_subtitles(
    video_path: Path,
    output_path: Optional[Path] = None,
    render_meta: Optional[dict] = None,
) -> Optional[Path]:
    """Transcribe a clip with Whisper and burn animated captions in with PyCaps.

    Args:
        video_path: Path to input video
        output_path: Optional output path. If None, creates alongside input.
        render_meta: The clip's render parameters; its source_path decides the
            codec the captioned clip is written in.

    Returns:
        Path to subtitled video, or None if subtitles are disabled/failed
    """
    if not is_subtitles_enabled():
        logging.info("Subtitles disabled via ENABLE_SUBTITLES env var")
        return None

    if output_path is None:
        # Use .mp4 extension since PyCaps always outputs mp4 container
        output_path = video_path.with_stem(video_path.stem + "_subtitled").with_suffix(".mp4")
    else:
        # Ensure output is mp4 even if caller specified different extension
        output_path = output_path.with_suffix(".mp4")

    config = SubtitleConfig.from_env()
    srt_path = video_path.with_suffix(".srt")

    try:
        logging.info("Using speech transcription mode (Whisper)")
        srt_path = transcribe_audio(video_path, srt_path)

        # Check if we got any subtitles
        if not srt_path.exists() or srt_path.stat().st_size == 0:
            logging.warning("No subtitles generated (empty SRT)")
            return None

        # Let the caption renderer write the codec the source used, rather
        # than its hardcoded H.264 and a re-encode afterwards.
        output_codec = None
        if render_meta and render_meta.get("source_path"):
            try:
                from video_io import target_codec_for

                output_codec = target_codec_for(render_meta["source_path"])
            except Exception as e:
                logging.warning(f"Could not determine the output codec: {e}")

        if not apply_pycaps_subtitles(
            video_path, srt_path, output_path, config, output_codec=output_codec,
        ):
            logging.error("Subtitle burn-in failed")
            return None

        # Clean up temporary files: the transcript, the word timings and the
        # document PyCaps exports beside its output. Only the clip is kept.
        for leftover in (
            srt_path,
            srt_path.with_suffix(".json"),
            word_timings_path(srt_path),
            srt_path.with_name(f"{srt_path.stem}_sub.json"),
        ):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass

        return output_path

    except Exception as e:
        logging.error(f"Subtitle pipeline failed: {e}")
        return None
