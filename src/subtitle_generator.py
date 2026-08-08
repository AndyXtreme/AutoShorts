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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        return cls(
            font_family=os.getenv("SUBTITLE_FONT", "Bangers"),
            font_size=int(os.getenv("SUBTITLE_FONT_SIZE", "48")),
            text_color=os.getenv("SUBTITLE_TEXT_COLOR", "#FFFFFF"),
            highlight_color=os.getenv("SUBTITLE_HIGHLIGHT_COLOR", "#00ff88"),
            shadow_color=os.getenv("SUBTITLE_SHADOW_COLOR", "#000000"),
            shadow_offset=int(os.getenv("SUBTITLE_SHADOW_OFFSET", "2")),
            position=os.getenv("SUBTITLE_POSITION", "bottom"),
            margin_bottom=int(os.getenv("SUBTITLE_MARGIN_BOTTOM", "50")),
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
        
        # Load model
        model = whisper.load_model(model_name)

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
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    except ImportError:
        logging.warning("silero-vad not installed; transcribing the whole clip at once.")
        return None

    try:
        model = load_silero_vad()
        wav = read_audio(str(audio_path), sampling_rate=16000)
        stamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
    except Exception as e:
        logging.warning(f"Voice activity detection failed ({e}); transcribing the whole clip at once.")
        return None

    if not stamps:
        return []

    padding = _env_float("VAD_PADDING", 0.3)
    merge_gap = _env_float("VAD_MERGE_GAP", 0.5)

    regions = []
    for stamp in stamps:
        start = max(0.0, float(stamp["start"]) - padding)
        end = float(stamp["end"]) + padding
        # Merge neighbours: a region per breath would give Whisper too little
        # context to recognise words reliably.
        if regions and start - regions[-1][1] <= merge_gap:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
    return regions


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

    segments.sort(key=lambda s: s["start"])
    text = " ".join(s.get("text", "").strip() for s in segments).strip()
    logging.info(f"Transcribed {len(regions)} speech region(s) separately.")
    return {"text": text, "segments": segments}


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
    word_lists: dict = None
) -> bool:
    """Apply animated subtitles using PyCaps (TemplateLoader API).
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video with subtitles
        config: Subtitle styling configuration
        word_lists: Dictionary of word lists for SemanticTagger
        
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
        args=(str(video_path), str(srt_path), str(output_path), config, word_lists)
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


def _run_pycaps_worker(video_path_str: str, srt_path_str: str, output_path_str: str, config: SubtitleConfig, word_lists: dict = None):
    """Worker function to run PyCaps in a separate process."""
    # Re-import necessary modules in the spawned process
    import sys
    import os
    
    try:
        from pycaps import CapsPipelineBuilder
        from pycaps.transcriber import AudioTranscriber
        from pycaps.common import Document, Segment, Line, Word, TimeFragment, Tag
        from pycaps.tag import SemanticTagger
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
        # This fixes the issue where it ignores AI captions and transcribes game audio
        print(f"Using custom SRTTranscriber with: {srt_path_str}")
        builder = builder.with_custom_audio_transcriber(SRTTranscriber(str(srt_path_str)))

        builder = (
            builder
            .with_output_video(output_path_str)
        )
        
        # Configure Semantic Tagger for AI tags (word highlighting handled by PyCaps templates)
        if word_lists:
            tagger = SemanticTagger()
            from pycaps.common import Tag
            for tag_name, w_list in word_lists.items():
                if w_list:
                    tagger.add_wordlist_rule(Tag(name=tag_name), w_list)
            builder.with_semantic_tagger(tagger)
        
        # Build pipeline
        pipeline = builder.build()
        
        pipeline.run()
        
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
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", subtitle_filter,
        "-c:v", "hevc_nvenc",  # Use GPU encoding
        "-preset", "slow",
        "-cq", "23",
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


def get_subtitle_mode() -> str:
    """Get the configured subtitle mode.
    
    Returns:
        One of: "speech", "ai_captions", "none"
    """
    return os.getenv("SUBTITLE_MODE", "ai_captions").lower()


def get_caption_style() -> str:
    """Get the configured AI caption style.
    
    Returns:
        One of: "gaming", "dramatic", "funny", "minimal"
    """
    return os.getenv("CAPTION_STYLE", "gaming").lower()

VIDEO_TYPE_DEFAULT_CAPTION_STYLE = {
    "gaming": "gaming",
    "podcasts": "podcast_quote",
    "entertainment": "entertainment_reaction",
    "sports": "sports_playbyplay",
    "vlogs": "vlog_story",
    "tv_shows": "tv_recap",
    "documentaries": "documentary_insight",
    "music": "music_hype",
    "educational": "educational_explainer",
    "interviews": "interview_quote",
    "comedy": "comedy_punchline",
    "news_commentary": "news_breaking",
    "esports": "esports_playcast",
    "cooking_diy": "cooking_step",
    "fitness": "fitness_coach",
}


def generate_subtitles(
    video_path: Path, 
    output_path: Optional[Path] = None,
    detected_category: Optional[str] = None,
    story_narration: Optional[str] = None,
    render_meta: Optional[dict] = None
) -> Optional[Path]:
    """Full subtitle pipeline with mode selection.
    
    Modes (set via SUBTITLE_MODE env var):
    - "speech": Use Whisper to transcribe voice/commentary
    - "ai_captions": Use AI to generate contextual captions (for gameplay without voice)
    - "none": Skip subtitle generation entirely
    
    Args:
        video_path: Path to input video
        output_path: Optional output path. If None, creates alongside input.
        detected_category: Optional category from AI analysis ("action", "funny", "highlight")
                          Used when CAPTION_STYLE=auto to match style to content.
        story_narration: Pre-generated narration text for story modes (cross-clip narrative)
        render_meta: Optional dict with source_path, start_time, duration, crop params for re-rendering
        
    Returns:
        Path to subtitled video, or None if subtitles are disabled/failed
    """
    if not is_subtitles_enabled():
        logging.info("Subtitles disabled via ENABLE_SUBTITLES env var")
        return None
    
    mode = get_subtitle_mode()
    
    if mode == "none":
        logging.info("Subtitle mode set to 'none'. Skipping.")
        return None
    
    if output_path is None:
        # Use .mp4 extension since PyCaps always outputs mp4 container
        output_path = video_path.with_stem(video_path.stem + "_subtitled").with_suffix(".mp4")
    else:
        # Ensure output is mp4 even if caller specified different extension
        output_path = output_path.with_suffix(".mp4")
    
    config = SubtitleConfig.from_env()
    srt_path = video_path.with_suffix(".srt")
    
    # Initialize tagging data (word lists for PyCaps SemanticTagger)
    word_lists = {}
    
    # Determine caption style (for both ai_captions and TTS voice selection)
    caption_style = None
    if mode == "ai_captions":
        caption_style = get_caption_style()
        
        # Auto style matching based on detected category
        if caption_style == "auto" and detected_category:
            from ai_providers import ClipScore
            caption_style = ClipScore.CAPTION_STYLE_MAP.get(detected_category, "gaming")
            logging.info(f"Auto-matched caption style: {caption_style} (from category: {detected_category})")
        elif caption_style == "auto":
            video_type = os.getenv("VIDEO_TYPE", "gaming").strip().lower()
            caption_style = VIDEO_TYPE_DEFAULT_CAPTION_STYLE.get(video_type, "gaming")
            logging.info(f"Auto-matched caption style: {caption_style} (from VIDEO_TYPE: {video_type})")
    
    try:
        if mode == "speech":
            # Whisper transcription mode
            logging.info("Using speech transcription mode (Whisper)")
            srt_path = transcribe_audio(video_path, srt_path)
            
        elif mode == "ai_captions":
            # AI-generated captions mode
            from ai_providers import (
                generate_ai_captions, 
                captions_to_srt, 
                batch_tag_captions, 
                apply_tags_to_pycaps, 
                add_emojis_to_caption,
                Caption
            )
            
            # Check if we have pre-generated story narration
            if story_narration:
                logging.info("Using pre-generated story narration")
                
                # Get video duration
                import subprocess
                import re
                ffprobe_cmd = [
                    "ffprobe", "-v", "error", 
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ]
                try:
                    duration = float(subprocess.check_output(ffprobe_cmd).decode().strip())
                except:
                    duration = 30.0
                
                # Split narration into sentences
                # Enhanced regex to handle:
                # 1. English: [.!?] followed by whitespace
                # 2. Japanese/CJK: [。！？] (no whitespace needed)
                # 3. Newlines
                sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])', story_narration.strip())
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if not sentences:
                    sentences = [story_narration]
                
                # For story mode: Generate TTS FIRST to get actual durations
                # Then create SRT based on actual audio timing
                from tts_generator import is_tts_enabled, QwenTTS, TTSConfig, generate_voice_description
                
                if is_tts_enabled():
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logging.info("GPU memory cleared before TTS loading")
                    
                    try:
                        tts_config = TTSConfig.from_env()
                        tts = QwenTTS(tts_config)
                        
                        # Load the model explicitly
                        tts._ensure_initialized()
                        
                        # Get voice description for story mode
                        # For story modes, caption_style takes priority over detected_category
                        # because story modes have specific voice presets (e.g., story_dramatic = Female)
                        voice_context = caption_style if caption_style and caption_style.startswith("story_") else (detected_category or caption_style)
                        voice_desc = generate_voice_description(voice_context)
                        logging.info(f"Using voice preset for: {voice_context}")
                        
                        # Preprocess slang in sentences for TTS
                        from tts_generator import preprocess_text_for_tts
                        processed_sentences = [preprocess_text_for_tts(s) for s in sentences]
                        
                        # Generate TTS for each sentence and SAVE the audio for later
                        # This avoids the sync issue of generating TTS twice
                        sentence_durations = []
                        sentence_audio_segments = []  # Store actual audio to reuse
                        tts_sample_rate = 24000  # Default, will be updated
                        
                        for sentence in processed_sentences:
                            # Generate TTS
                            wavs, sr = tts._model.generate_voice_design(
                                text=sentence,
                                instruct=voice_desc,
                                language=tts.config.get_language_name(),
                            )
                            
                            tts_sample_rate = sr
                            
                            if wavs and len(wavs) > 0:
                                audio_duration = len(wavs[0]) / sr
                                sentence_durations.append(audio_duration)
                                sentence_audio_segments.append(wavs[0])  # Save audio
                            else:
                                # Fallback to word-count estimate
                                word_count = len(sentence.split())
                                sentence_durations.append(max(1.5, word_count * 0.4))
                                sentence_audio_segments.append(None)  # No audio
                        
                        # Now create SRT with ACTUAL TTS durations
                        result_captions = []
                        current_time = 0.5  # Small buffer
                        
                        # Split long sentences into smaller visual chunks specifically for story mode
                        # This prevents the "wall of text" issue in PyCaps
                        MAX_WORDS_PER_CAPTION = 7
                        
                        for sentence, tts_duration in zip(sentences, sentence_durations):
                            # --- CJK Detection ---
                            is_cjk = any("\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" for char in sentence)
                            MAX_CJK_CHARS = 18  # Characters per line for CJK
                            
                            if is_cjk:
                                # Character-based splitting for CJK
                                if len(sentence) <= MAX_CJK_CHARS:
                                    # Short sentence
                                    end_time = current_time + tts_duration
                                    result_captions.append(Caption(
                                        start_time=current_time,
                                        end_time=end_time,
                                        text=sentence,
                                        style="narrative"
                                    ))
                                    current_time = end_time
                                else:
                                    # Long sentence -> Split by chars
                                    chunks = [sentence[i:i+MAX_CJK_CHARS] for i in range(0, len(sentence), MAX_CJK_CHARS)]
                                    
                                    total_chars = len(sentence)
                                    chunk_start = current_time
                                    
                                    for i, chunk in enumerate(chunks):
                                        chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                        chunk_dur = tts_duration * chunk_ratio
                                        
                                        chunk_end = chunk_start + chunk_dur
                                        if i == len(chunks) - 1:
                                            chunk_end = current_time + tts_duration
                                            
                                        result_captions.append(Caption(
                                            start_time=chunk_start,
                                            end_time=chunk_end,
                                            text=chunk,
                                            style="narrative"
                                        ))
                                        chunk_start = chunk_end
                                    
                                    # Update current_time for the next sentence
                                    current_time = current_time + tts_duration
                            else:
                                # --- Standard Word-based Logic ---
                                words = sentence.split()
                                if not words:
                                    continue
                                
                                # If sentence is short enough, keep as one
                                if len(words) <= MAX_WORDS_PER_CAPTION:
                                    end_time = current_time + tts_duration
                                    result_captions.append(Caption(
                                        start_time=current_time,
                                        end_time=end_time,
                                        text=sentence,
                                        style="narrative"
                                    ))
                                    current_time = end_time
                                else:
                                    # Split into chunks
                                    chunks = []
                                    for i in range(0, len(words), MAX_WORDS_PER_CAPTION):
                                        chunk_words = words[i:i + MAX_WORDS_PER_CAPTION]
                                        chunks.append(" ".join(chunk_words))
                                    
                                    # Distribute duration proportionally
                                    total_chars = len(sentence)
                                    chunk_start = current_time
                                    
                                    for i, chunk in enumerate(chunks):
                                        # Calculate duration based on character count ratio
                                        # (characters correlate better with speaking time than word count)
                                        chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                        chunk_dur = tts_duration * chunk_ratio
                                        
                                        # For the last chunk, ensure we essentially align with the end
                                        # (floating point fix)
                                        chunk_end = chunk_start + chunk_dur
                                        if i == len(chunks) - 1:
                                            chunk_end = current_time + tts_duration
                                            
                                        result_captions.append(Caption(
                                            start_time=chunk_start,
                                            end_time=chunk_end,
                                            text=chunk,
                                            style="narrative"
                                        ))
                                        chunk_start = chunk_end
                                    
                                    # Update current_time for the next sentence
                                    current_time = current_time + tts_duration
                        
                        # === SAVE PRE-GENERATED TTS AUDIO ===
                        # Build the final audio from saved segments to avoid regenerating
                        import numpy as np
                        import scipy.io.wavfile as wav
                        
                        audio_parts = []
                        audio_current_time = 0.0
                        
                        for i, (tts_dur, audio_seg) in enumerate(zip(sentence_durations, sentence_audio_segments)):
                            # Calculate start time for this sentence (same logic as captions)
                            # Sentences start at 0.5s buffer, then sequentially
                            expected_start = 0.5 + sum(sentence_durations[:i])
                            
                            # Add silence gap if needed
                            silence_needed = expected_start - audio_current_time
                            if silence_needed > 0:
                                silence_samples = int(silence_needed * tts_sample_rate)
                                audio_parts.append(np.zeros(silence_samples, dtype=np.float32))
                                audio_current_time += silence_needed
                            
                            # Add audio segment
                            if audio_seg is not None:
                                audio_parts.append(audio_seg.astype(np.float32))
                                audio_current_time += len(audio_seg) / tts_sample_rate
                            else:
                                # Add silence for failed generations
                                silence_samples = int(tts_dur * tts_sample_rate)
                                audio_parts.append(np.zeros(silence_samples, dtype=np.float32))
                                audio_current_time += tts_dur
                        
                        # Save pre-generated TTS to a temp file for later use
                        if audio_parts:
                            story_tts_audio = np.concatenate(audio_parts)
                            story_tts_audio_int16 = (story_tts_audio * 32767).astype(np.int16)
                            
                            # Save to temp file that will be used instead of regenerating
                            story_tts_path = srt_path.with_suffix(".story_tts.wav")
                            wav.write(str(story_tts_path), tts_sample_rate, story_tts_audio_int16)
                            logging.info(f"Pre-generated story TTS saved: {audio_current_time:.1f}s")
                        
                        # Clean up TTS model (but keep the saved audio file)
                        del tts
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logging.warning(f"TTS pre-generation failed, falling back to word-count estimate: {e}")
                        # Fallback to original word-count based timing
                        total_words = sum(len(s.split()) for s in sentences)
                        result_captions = []
                        current_time = 0.5
                        
                        # Split into chunks logic for fallback
                        MAX_WORDS_PER_CAPTION = 7
                        
                        for sentence in sentences:
                            # Calculate TOTAL duration for this sentence
                            word_count = len(sentence.split())
                            if total_words > 0:
                                proportion = word_count / total_words
                                total_segment_duration = max(1.5, (duration - 1.0) * proportion)
                            else:
                                total_segment_duration = max(1.5, word_count * 0.4)
                            
                            words = sentence.split()
                            if not words: continue

                            if len(words) <= MAX_WORDS_PER_CAPTION:
                                end_time = min(current_time + total_segment_duration, duration - 0.5)
                                result_captions.append(Caption(
                                    start_time=current_time,
                                    end_time=end_time,
                                    text=sentence,
                                    style="narrative"
                                ))
                                current_time = end_time
                            else:
                                # Chunk logic
                                chunks = []
                                for i in range(0, len(words), MAX_WORDS_PER_CAPTION):
                                    chunk_words = words[i:i + MAX_WORDS_PER_CAPTION]
                                    chunks.append(" ".join(chunk_words))
                                
                                total_chars = len(sentence)
                                chunk_start = current_time
                                
                                for i, chunk in enumerate(chunks):
                                    chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                    chunk_dur = total_segment_duration * chunk_ratio
                                    
                                    # Alignment fix for last chunk
                                    if i == len(chunks) - 1:
                                        overall_end_time = min(current_time + total_segment_duration, duration - 0.5)
                                        chunk_end = overall_end_time
                                    else:
                                         chunk_end = chunk_start + chunk_dur
                                    
                                    result_captions.append(Caption(
                                        start_time=chunk_start,
                                        end_time=chunk_end,
                                        text=chunk,
                                        style="narrative"
                                    ))
                                    chunk_start = chunk_end
                                
                                current_time = chunk_start
                else:
                    # No TTS - use word-count estimate
                    total_words = sum(len(s.split()) for s in sentences)
                    result_captions = []
                    current_time = 0.5
                    
                    # Split into chunks logic for No-TTS mode
                    MAX_WORDS_PER_CAPTION = 7
                    
                    for sentence in sentences:
                        # Calculate TOTAL duration for this sentence
                        word_count = len(sentence.split())
                        if total_words > 0:
                            proportion = word_count / total_words
                            total_segment_duration = max(1.5, (duration - 1.0) * proportion)
                        else:
                            total_segment_duration = max(1.5, word_count * 0.4)
                        
                        words = sentence.split()
                        if not words: continue

                        if len(words) <= MAX_WORDS_PER_CAPTION:
                            end_time = min(current_time + total_segment_duration, duration - 0.5)
                            result_captions.append(Caption(
                                start_time=current_time,
                                end_time=end_time,
                                text=sentence,
                                style="narrative"
                            ))
                            current_time = end_time
                        else:
                            # Chunk logic
                            chunks = []
                            for i in range(0, len(words), MAX_WORDS_PER_CAPTION):
                                chunk_words = words[i:i + MAX_WORDS_PER_CAPTION]
                                chunks.append(" ".join(chunk_words))
                            
                            total_chars = len(sentence)
                            chunk_start = current_time
                            
                            for i, chunk in enumerate(chunks):
                                chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                chunk_dur = total_segment_duration * chunk_ratio
                                
                                # Alignment fix for last chunk
                                if i == len(chunks) - 1:
                                    overall_end_time = min(current_time + total_segment_duration, duration - 0.5)
                                    chunk_end = overall_end_time
                                else:
                                     chunk_end = chunk_start + chunk_dur
                                
                                result_captions.append(Caption(
                                    start_time=chunk_start,
                                    end_time=chunk_end,
                                    text=chunk,
                                    style="narrative"
                                ))
                                chunk_start = chunk_end
                            
                            current_time = chunk_start
                
                # Save SRT
                captions_to_srt(result_captions, srt_path)
                logging.info(f"Generated {len(result_captions)} story narration segments ({caption_style} style)")
                
            else:
                # Regular AI caption generation
                logging.info("Using AI caption generation mode")
                
                # Get configured max_captions (0 = auto/dynamic)
                max_captions_env = int(os.getenv("MAX_CAPTIONS", "0"))
                is_auto_captions = (max_captions_env == 0)
                
                # Get video duration for dynamic calculation
                import subprocess
                import json
                try:
                    probe_cmd = [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", str(video_path)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    probe_data = json.loads(probe_result.stdout)
                    duration = float(probe_data["format"]["duration"])
                    
                    # Dynamic caption count: ~1 caption per 2-4 seconds (varies by style)
                    # Story modes: 1 caption per 5-6 seconds (narrative pacing)
                    # Regular modes: 1 caption per 2-3 seconds (punchy)
                    is_story_mode = caption_style.startswith("story_")
                    
                    if is_story_mode:
                        # Story modes: narrative pacing, but not too sparse
                        # Aim for ~1 caption per 5-6 seconds for good coverage
                        ideal_captions = max(4, int(duration / 5.5))  # 1 caption per ~5.5 seconds
                    else:
                        # Regular modes: shorter punchy captions
                        ideal_captions = max(2, int(duration / 2.5))  # 1 caption per ~2.5 seconds
                    
                    if is_auto_captions:
                        # Auto mode: use ideal count (no cap)
                        max_captions = ideal_captions
                        logging.info(f"Dynamic caption count: {max_captions} captions for {duration:.1f}s video "
                                   f"(~{duration/max_captions:.1f}s per caption)")
                    else:
                        # Manual mode: cap at configured maximum
                        max_captions = min(ideal_captions, max_captions_env)
                        logging.info(f"Dynamic caption count: {max_captions} captions for {duration:.1f}s video "
                                   f"(~{duration/max_captions:.1f}s per caption, max={max_captions_env})")
                    
                except Exception as e:
                    # Fallback to env variable or default if duration detection fails
                    logging.warning(f"Could not determine video duration: {e}")
                    max_captions = max_captions_env if max_captions_env > 0 else 10
                
                result = generate_ai_captions(video_path, style=caption_style, max_captions=max_captions)
                
                if not result.success or not result.captions:
                    logging.warning(f"AI caption generation failed: {result.error}")
                    logging.info("Falling back to speech mode...")
                    
                    # Fallback to speech mode
                    try:
                        srt_path = transcribe_audio(video_path, srt_path)
                    except Exception as e:
                        logging.error(f"Speech fallback also failed: {e}")
                        return None
                else:
                    # Enhance captions with AI-powered emojis and tagging
                    category = detected_category or caption_style
                    
                    # 1. Generate tags and emojis
                    tag_results = batch_tag_captions(result.captions, category=category)
                    
                    # 2. Add emojis to text
                    add_emojis = os.getenv("ENABLE_CAPTION_EMOJIS", "true").lower() in ("true", "1", "yes")
                    enhanced_captions = []
                    for cap in result.captions:
                        tag_res = tag_results.get(cap.text)
                        if tag_res and add_emojis:
                            new_text = add_emojis_to_caption(cap.text, tag_res)
                            enhanced_captions.append(Caption(cap.start_time, cap.end_time, new_text, cap.style))
                        else:
                            enhanced_captions.append(cap)
                    
                    # 3. Generate PyCaps Tagging Data (word lists only, no custom CSS)
                    word_lists = apply_tags_to_pycaps(result.captions, tag_results)
                    
                    # Convert enhanced AI captions to SRT
                    captions_to_srt(enhanced_captions, srt_path)
                    logging.info(f"Generated {len(enhanced_captions)} AI captions ({caption_style} style)")
        else:
            logging.warning(f"Unknown subtitle mode: {mode}. Using ai_captions.")
            return generate_subtitles(video_path, output_path, detected_category, render_meta=render_meta)
        
        # Check if we got any subtitles
        if not srt_path.exists() or srt_path.stat().st_size == 0:
            logging.warning("No subtitles generated (empty SRT)")
            return None
        
        # === PRE-GENERATE TTS FOR AUTO MODE (non-story) ===
        # For auto/regular modes, generate TTS BEFORE PyCaps so we can:
        # 1. Adjust SRT timings to match actual TTS audio durations
        # 2. Extend video if TTS is longer than video
        # This mirrors story mode's approach and prevents subtitle/voiceover desync
        auto_tts_path = srt_path.with_suffix(".auto_tts.wav")
        
        if mode == "ai_captions" and not story_narration:  # Only for non-story caption flows
            try:
                from tts_generator import is_tts_enabled, QwenTTS, preprocess_text_for_tts, generate_voice_description
                
                if is_tts_enabled():
                    import gc
                    import torch
                    import numpy as np
                    import scipy.io.wavfile as wav_io
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logging.info("GPU memory cleared before TTS loading")
                    
                    tts = QwenTTS.get_instance()
                    tts._ensure_initialized()
                    if tts._model is None:
                        logging.warning("TTS model unavailable for auto pre-generation, keeping subtitles only")
                    else:
                        # Get voice description
                        voice_context = caption_style if caption_style else (detected_category or "action")
                        voice_desc = generate_voice_description(voice_context)
                        first_line = voice_desc.split('\n')[0] if '\n' in voice_desc else voice_desc[:60]
                        logging.info(f"Generating voice for context: {voice_context}")
                        logging.info(f"Voice: {first_line}...")
                        
                        # Parse the SRT to get captions
                        captions_for_tts = _parse_srt_with_timing(srt_path)
                        
                        if captions_for_tts:
                            audio_segments = []
                            updated_captions = []
                            current_audio_time = 0.0
                            sample_rate = 24000
                            
                            logging.info(f"Generating TTS for {len(captions_for_tts)} captions ({detected_category or 'auto'} mode)")
                            
                            for i, cap in enumerate(captions_for_tts):
                                raw_text = cap.get("text", "").strip()
                                start_time = cap.get("start", 0.0)
                                end_time = cap.get("end", start_time + 2.0)
                                fallback_start = max(start_time, current_audio_time)
                                fallback_end = max(fallback_start + 0.1, fallback_start + (end_time - start_time))
                                
                                text = preprocess_text_for_tts(_sanitize_tts_input(raw_text))
                                if not text:
                                    updated_captions.append({"start": fallback_start, "end": fallback_end, "text": raw_text})
                                    continue
                                
                                # Add silence gap to reach this caption's start
                                silence_dur = max(0, start_time - current_audio_time)
                                if silence_dur > 0:
                                    silence = np.zeros(int(silence_dur * sample_rate), dtype=np.float32)
                                    audio_segments.append(silence)
                                    current_audio_time += silence_dur
                                
                                # Track true audio start for this caption after any carry-over shift
                                actual_start = current_audio_time
                                
                                try:
                                    wavs, sr = tts._model.generate_voice_design(
                                        text=text,
                                        instruct=voice_desc,
                                        language=tts.config.get_language_name(),
                                    )
                                    sample_rate = sr
                                    
                                    if wavs and len(wavs) > 0:
                                        audio = wavs[0].astype(np.float32)
                                        tts_dur = len(audio) / sample_rate
                                        audio_segments.append(audio)
                                        current_audio_time += tts_dur
                                        updated_captions.append({
                                            "start": actual_start,
                                            "end": max(actual_start + 0.1, current_audio_time),
                                            "text": raw_text
                                        })
                                    else:
                                        updated_captions.append({"start": fallback_start, "end": fallback_end, "text": raw_text})
                                except Exception as e:
                                    logging.warning(f"TTS failed for caption {i}: {e}")
                                    updated_captions.append({"start": fallback_start, "end": fallback_end, "text": raw_text})
                            
                            if audio_segments:
                                final_audio = np.concatenate(audio_segments)
                                final_audio_int16 = (final_audio * 32767).astype(np.int16)
                                wav_io.write(str(auto_tts_path), sample_rate, final_audio_int16)
                                total_tts_dur = len(final_audio) / sample_rate
                                logging.info(f"TTS audio generated: {total_tts_dur:.1f}s for {len(captions_for_tts)} captions")
                                
                                # Rewrite SRT using actual generated audio timing to keep subtitle/voice sync exact.
                                original_end = max((cap.get("end", 0.0) for cap in captions_for_tts), default=0.0)
                                adjusted_end = max((cap.get("end", 0.0) for cap in updated_captions), default=0.0)
                                shift = max(0.0, adjusted_end - original_end)
                                logging.info(f"Adjusted SRT timing: shifted {shift:.1f}s to match TTS durations")
                                
                                srt_lines = []
                                for idx, cap in enumerate(updated_captions, 1):
                                    start_ts = _format_timestamp(cap["start"])
                                    end_ts = _format_timestamp(cap["end"])
                                    srt_lines.append(str(idx))
                                    srt_lines.append(f"{start_ts} --> {end_ts}")
                                    srt_lines.append(cap["text"])
                                    srt_lines.append("")
                                
                                with open(srt_path, "w", encoding="utf-8") as f:
                                    f.write("\n".join(srt_lines))
                        
                        # Clean up TTS model memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
            except ImportError:
                logging.debug("TTS module not available, skipping pre-generation")
            except Exception as e:
                logging.warning(f"TTS pre-generation failed for auto mode: {e}")
                # Clean up failed TTS file
                if auto_tts_path.exists():
                    try:
                        auto_tts_path.unlink()
                    except Exception:
                        pass
        
        # === EXTEND VIDEO TO MATCH TTS DURATION (if needed) ===
        # This MUST happen BEFORE PyCaps so subtitles are rendered on the full-length video
        story_tts_path = srt_path.with_suffix(".story_tts.wav")
        # Use whichever TTS audio exists (story mode or auto mode)
        pre_generated_tts_path = story_tts_path if story_tts_path.exists() else (auto_tts_path if auto_tts_path.exists() else None)
        video_for_pycaps = video_path
        extended_video_temp = None
        
        if pre_generated_tts_path and render_meta:
            from tts_generator import get_audio_duration, get_video_duration, rerender_video_longer, RenderMeta
            
            tts_duration = get_audio_duration(pre_generated_tts_path)
            video_duration = get_video_duration(video_path)
            
            if tts_duration > video_duration + 1.0:  # TTS is significantly longer
                logging.info(f"TTS ({tts_duration:.1f}s) longer than video ({video_duration:.1f}s)")
                logging.info("Re-rendering video to TTS duration BEFORE applying subtitles...")
                
                # Re-render video to match TTS duration + buffer
                extended_video_temp = video_path.with_stem(video_path.stem + "_extended_for_subs")
                target_duration = tts_duration + 1.0
                
                try:
                    tts_render_meta = RenderMeta(
                        source_path=Path(render_meta["source_path"]),
                        start_time=render_meta["start_time"],
                        original_duration=render_meta["duration"],
                        output_width=render_meta["output_width"],
                        output_height=render_meta["output_height"],
                        crop_x=render_meta["crop_x"],
                        crop_y=render_meta["crop_y"],
                        crop_w=render_meta["crop_w"],
                        crop_h=render_meta["crop_h"],
                        bg_width=render_meta.get("bg_width", render_meta["output_width"]),
                        bg_height=render_meta.get("bg_height", render_meta["output_height"]),
                        is_vertical_bg=render_meta.get("is_vertical_bg", True),
                    )
                    
                    if rerender_video_longer(tts_render_meta, target_duration, extended_video_temp):
                        video_for_pycaps = extended_video_temp
                        logging.info(f"Video extended to {target_duration:.1f}s for subtitle rendering")
                    else:
                        logging.warning("Video extension failed, subtitles may be truncated")
                except Exception as e:
                    logging.warning(f"Could not extend video: {e}")
        
        # Apply subtitles using PyCaps (to the potentially extended video)
        success = apply_pycaps_subtitles(video_for_pycaps, srt_path, output_path, config, word_lists=word_lists)
        
        # DON'T clean up extended video yet - we need its audio track for TTS mixing!
        # PyCaps strips audio, so we need it from the source video
        audio_source_video = extended_video_temp if extended_video_temp and extended_video_temp.exists() else video_path
        
        if not success:
            logging.error("Subtitle burn-in failed")
            # Cleanup extended video on failure
            if extended_video_temp and extended_video_temp.exists():
                try:
                    extended_video_temp.unlink()
                except Exception:
                    pass
            return None
        
        # --- TTS Voiceover (Optional) ---
        try:
            from tts_generator import is_tts_enabled
            
            if is_tts_enabled():
                logging.info("Generating TTS voiceover...")
                
                # Check if we have pre-generated TTS audio (story mode or auto mode)
                # This avoids regenerating TTS which would cause sync issues
                story_tts_path = srt_path.with_suffix(".story_tts.wav")
                
                # Pick whichever pre-generated TTS exists
                if story_tts_path.exists():
                    pre_gen_tts = story_tts_path
                elif auto_tts_path.exists():
                    pre_gen_tts = auto_tts_path
                else:
                    pre_gen_tts = None
                
                if pre_gen_tts:
                    # === USE PRE-GENERATED TTS (story or auto mode) ===
                    logging.info(f"Using pre-generated TTS audio (perfect sync)")
                    voiceover_path = pre_gen_tts
                    
                    # Mix voiceover with video
                    mixed_output = output_path.with_stem(output_path.stem + "_voiced")
                    
                    game_vol = float(os.getenv("TTS_GAME_AUDIO_VOLUME", "0.3"))
                    voice_vol = float(os.getenv("TTS_VOICEOVER_VOLUME", "1.0"))
                    
                    # PyCaps output has NO AUDIO, so we need to mix 3 streams:
                    # 1. Video from PyCaps output (output_path)
                    # 2. Audio from source video (audio_source_video)
                    # 3. TTS voiceover
                    logging.info(f"Mixing: video from PyCaps, audio from source, TTS voiceover")
                    
                    filter_complex = (
                        f"[1:a]volume={game_vol}[game];"
                        f"[2:a]volume={voice_vol}[voice];"
                        "[game][voice]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
                    )
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(output_path),         # Video from PyCaps (no audio)
                        "-i", str(audio_source_video),  # Audio source (original/extended video)
                        "-i", str(voiceover_path),      # TTS voiceover
                        "-filter_complex", filter_complex,
                        "-map", "0:v",                  # Video from PyCaps
                        "-map", "[aout]",               # Mixed audio
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        str(mixed_output)
                    ]
                    
                    import subprocess
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and mixed_output.exists():
                        # Replace subtitled version with voiced version
                        output_path.unlink()
                        mixed_output.rename(output_path)
                        logging.info(f"TTS voiceover added successfully")
                    else:
                        logging.warning(f"TTS audio mixing failed: {result.stderr[:300]}")
                    
                    # Cleanup the pre-generated TTS file
                    try:
                        pre_gen_tts.unlink()
                    except Exception:
                        pass
                        
                    # Cleanup extended video temp file if it was used
                    if extended_video_temp and extended_video_temp.exists():
                        try:
                            extended_video_temp.unlink()
                        except Exception:
                            pass
                else:
                    logging.warning("No pre-generated TTS audio found; skipping legacy on-the-fly TTS fallback to preserve sync")
                    if extended_video_temp and extended_video_temp.exists():
                        try:
                            extended_video_temp.unlink()
                        except Exception:
                            pass
                    
        except ImportError:
            logging.debug("TTS module not available, skipping voiceover")
        except Exception as e:
            logging.warning(f"TTS generation failed: {e}, keeping subtitles only")
        
        # Clean up temporary files
        try:
            srt_path.unlink()
            json_path = srt_path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
        except Exception:
            pass
        
        return output_path
            
    except Exception as e:
        logging.error(f"Subtitle pipeline failed: {e}")
        return None


def _extract_srt_text(srt_path: Path) -> str:
    """Extract all text from an SRT file for TTS."""
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        text_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines, index numbers, and timestamps
            if not line or line.isdigit() or "-->" in line:
                continue
            text_lines.append(line)
        
        return " ".join(text_lines)
    except Exception:
        return ""


def _sanitize_tts_input(text: str) -> str:
    """Sanitize caption text for TTS while preserving visible subtitle text."""
    import unicodedata

    if not text:
        return ""

    cleaned = re.sub(r"\*[^*]+\*", "", text).strip()
    cleaned = cleaned.replace("\u200d", "").replace("\ufe0f", "").replace("\ufe0e", "")
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) not in ("So", "Sk"))
    cleaned = re.sub(r"[*#_~`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_srt_with_timing(srt_path: Path) -> list:
    """Parse SRT file and return captions with timing info.
    
    Returns:
        List of {"start": float, "end": float, "text": str}
    """
    def parse_timestamp(ts: str) -> float:
        """Convert SRT timestamp to seconds."""
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    captions = []
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by double newlines (caption blocks)
        blocks = content.strip().split("\n\n")
        
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            
            # Find timestamp line
            timestamp_line = None
            text_start_idx = 0
            for i, line in enumerate(lines):
                if "-->" in line:
                    timestamp_line = line
                    text_start_idx = i + 1
                    break
            
            if not timestamp_line:
                continue
            
            # Parse timestamps
            parts = timestamp_line.split("-->")
            if len(parts) != 2:
                continue
            
            start = parse_timestamp(parts[0])
            end = parse_timestamp(parts[1])
            
            # Get text (remaining lines)
            text = " ".join(lines[text_start_idx:]).strip()
            
            # Preserve original subtitle text (including emojis) for rendering.
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text:
                captions.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
        
        return captions
    except Exception as e:
        logging.warning(f"Failed to parse SRT for timing: {e}")
        return []
