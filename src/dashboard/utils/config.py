from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import dotenv_values

ENV_PATH = Path(".env")


@dataclass(frozen=True)
class EnvField:
    name: str
    label: str
    field_type: str
    default: Any
    options: Optional[Sequence[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    help_text: Optional[str] = None
    multiline: bool = False


@dataclass(frozen=True)
class EnvSection:
    title: str
    fields: Sequence[EnvField]
    expanded: bool = False


def get_schema() -> List[EnvSection]:
    return [
        EnvSection(
            title="Core Settings",
            expanded=True,
            fields=[
                EnvField("X_CENTER", "Crop center X", "float", 0.5, min_value=0.0, max_value=1.0, step=0.01,
                         help_text="Where to center the crop horizontally. 0.5 = middle, 0 = left, 1 = right"),
                EnvField("Y_CENTER", "Crop center Y", "float", 0.5, min_value=0.0, max_value=1.0, step=0.01,
                         help_text="Where to center the crop vertically. 0.5 = middle, 0 = top, 1 = bottom"),
                EnvField("MAX_ERROR_DEPTH", "Max retry depth", "int", 3, min_value=1, max_value=10,
                         help_text="How many times to retry if rendering fails"),
                EnvField("SCENE_LIMIT", "Scene limit", "int", 4, min_value=1, max_value=50,
                         help_text="Maximum number of clips to generate"),
                EnvField("TARGET_RATIO_W", "Aspect ratio width", "int", 9, min_value=1, max_value=32,
                         help_text="Width part of the output aspect ratio. 9 with height 16 = vertical Shorts format"),
                EnvField("TARGET_RATIO_H", "Aspect ratio height", "int", 16, min_value=1, max_value=32,
                         help_text="Height part of the output aspect ratio. 16 with width 9 = vertical Shorts format"),
                EnvField("MAX_OUTPUT_HEIGHT", "Max output height (px)", "select", "1920",
                         options=["1280", "1920", "2560", "3840"],
                         help_text="Upper limit for the rendered resolution: 1280 = 720x1280, 1920 = 1080x1920, "
                                   "2560 = 1440x2560, 3840 = 2160x3840. The clip uses the smallest of these that "
                                   "is still at least as tall as the cropped source, so nothing gets downscaled "
                                   "needlessly. 1920 is the right choice for YouTube Shorts"),
                EnvField("OUTPUT_VIDEO_CODEC", "Output codec", "select", "auto",
                         options=["auto", "hevc", "av1", "h264"],
                         help_text="Codec the finished Shorts are written in. 'auto' matches the source: "
                                   "an AV1 recording gives AV1 Shorts, an H.265 one gives H.265. The caption "
                                   "renderer is switched to this codec directly, so there is no H.264 "
                                   "generation in between"),
            ],
        ),
        EnvSection(
            title="Action Detection",
            expanded=True,
            fields=[
                EnvField("ACTION_W_AUDIO", "Audio weight", "float", 0.6, min_value=0.0, max_value=1.0, step=0.05,
                         help_text="How much loudness spikes (gunfire, shouting, impacts) drive clip selection"),
                EnvField("ACTION_W_VIDEO", "Motion weight", "float", 0.4, min_value=0.0, max_value=1.0, step=0.05,
                         help_text="How much frame-to-frame motion drives clip selection. Raise above the audio "
                                   "weight to favour visually busy moments over loud ones. Only the ratio matters"),
                EnvField("ACTION_SCORE_MODE", "Score scenes by", "select", "sum", options=["sum", "mean"],
                         help_text="sum: total action in the scene, so a long average scene can outrank a short "
                                   "intense one. mean: action per second, which surfaces short highlights but "
                                   "favours very short scenes where one spike carries the average"),
                EnvField("SCENE_THRESHOLD", "Scene cut sensitivity", "float", 27.0,
                         min_value=5.0, max_value=60.0, step=1.0,
                         help_text="How different two frames must look to count as a cut. Lower finds more, "
                                   "shorter scenes - useful for continuous footage without hard cuts, where the "
                                   "default leaves too few candidates for the scene limit to choose from"),
                EnvField("ACTION_FPS", "Motion sampling rate", "int", 6, min_value=1, max_value=30,
                         help_text="Frames per second examined when measuring motion. Higher catches brief "
                                   "spikes but slows the analysis pass down roughly proportionally"),
            ],
        ),
        EnvSection(
            title="Dead Air Removal",
            expanded=True,
            fields=[
                EnvField("REMOVE_SILENCE", "Cut out dead air", "bool", True,
                         help_text="Cut stretches out of a clip where nothing is said and nothing happens, "
                                   "then stitch the rest back together"),
                EnvField("SILENCE_MIN_GAP", "Min gap to cut (s)", "float", 1.0,
                         min_value=0.2, max_value=10.0, step=0.1,
                         help_text="Only gaps at least this long are removed. Around 1s keeps normal speech "
                                   "rhythm intact; 0.5s gives the hard jump-cut style"),
                EnvField("SILENCE_PADDING", "Keep around speech (s)", "float", 0.15,
                         min_value=0.0, max_value=1.0, step=0.05,
                         help_text="Breathing room kept before and after each word so cuts do not clip syllables"),
                EnvField("SILENCE_EDGE_PADDING", "Air at start and end (s)", "float", 0.8,
                         min_value=0.0, max_value=5.0, step=0.1,
                         help_text="Silence kept at the very beginning and end of a clip. Cutting it flush "
                                   "makes the clip start and stop on the exact frame a word begins or ends, "
                                   "which reads as an abrupt cut"),
                EnvField("SILENCE_PEAK_LEAD", "Keep before the highlight (s)", "float", 5.0,
                         min_value=0.0, max_value=15.0, step=0.5,
                         help_text="Seconds before a clip's action peak that are never cut. Tension before "
                                   "a payoff is quiet and still, so it looks like dead air - without this "
                                   "the successful jump was cut and the clip went from the failed tries "
                                   "straight to the cheering. 0 disables it"),
                EnvField("SILENCE_MIN_RESULT", "Min length after cutting (s)", "float", 8.0,
                         min_value=1.0, max_value=60.0, step=0.5,
                         help_text="Floor for the finished clip. Cutting stops here even if more dead air "
                                   "remains. Independent of the clip length settings below, which govern how "
                                   "much source material is selected - pick generously, cut tightly"),
                EnvField("SILENCE_MOTION_KEEP", "Protect motion above", "float", 0.5,
                         min_value=-1.0, max_value=3.0, step=0.1,
                         help_text="Silent moments with at least this much motion are kept anyway, measured in "
                                   "standard deviations above the video's average. Lower keeps more silent "
                                   "action, higher cuts more aggressively"),
            ],
        ),
        EnvSection(
            title="Clip Length Settings",
            expanded=True,
            fields=[
                EnvField("CLIP_LENGTH_MODE", "Window length", "select", "max", options=["max", "random"],
                         help_text="max: always take the longest allowed window and let dead-air removal "
                                   "tighten it - reproducible, and nothing at the end gets lost. "
                                   "random: draw a length per clip, so reruns end at different points"),
                EnvField("TARGET_SHORT_LENGTH", "Target clip length (s)", "int", 35,
                         min_value=0, max_value=300,
                         help_text="How long clips should aim to be. Scene detection cuts on what the picture "
                                   "does, not on where a move starts or stops, so the scene says where to look "
                                   "rather than how much to take - a window this long is placed inside it by "
                                   "the action profile and may run past its end. 0 goes back to using the "
                                   "scene's own length, which makes every clip a different length and puts one "
                                   "clip per run at the maximum"),
                EnvField("CLIP_LENGTH_PRESET", "Length preset", "select", "balanced",
                         options=["fixed", "balanced", "loose"],
                         help_text="fixed: every clip runs the target length and starts inside its scene - "
                                   "what runs before October 2026 used, so pick it to reproduce one of "
                                   "those. balanced (default): the clip is built around the scene's "
                                   "peak and may start before the scene, so the attempts leading up to a "
                                   "payoff are kept, and it runs longer than the target only for clearly "
                                   "active material or shorter for clearly dead air (measured on ten "
                                   "recordings: median 33s, 15 of 40 clips longer than 35s). loose: the "
                                   "same, but gives in to extra material more readily (median 35s, 14 of "
                                   "40 between 40 and 45s). All three are reproducible. Measured with "
                                   "motion weighted 0.8 - other weights shift the dynamic presets"),
                EnvField("DYNAMIC_MAX_LENGTH", "Longest dynamic clip (s)", "int", 45,
                         min_value=15, max_value=120,
                         help_text="Upper limit for balanced and loose, tail included - never above the "
                                   "max short length. At 59 instead of 45, balanced let 4 of 40 clips run "
                                   "the full minute"),
                EnvField("MIN_SHORT_LENGTH", "Min short length (s)", "int_auto", 15, min_value=5, max_value=120,
                         help_text="Shortest allowed clip. Auto mode picks based on video length"),
                EnvField("RENDER_TAIL_BUFFER", "Tail after the window (s)", "float", 1.5,
                         min_value=0.0, max_value=10.0, step=0.5,
                         help_text="Seconds of source material rendered after the chosen window, reserved from "
                                   "within the maximum length. Without it the window boundary lands during or "
                                   "right after speech and the clip stops on the exact frame - or mid-word"),
                EnvField("MAX_SHORT_LENGTH", "Max short length (s)", "int_auto", 59, min_value=15, max_value=300,
                         help_text="Longest allowed clip. 59s keeps clips inside the YouTube Shorts limit"),
                EnvField("CLIP_END_SNAP", "Snap the end to a lull (s)", "float", 6.0,
                         min_value=0.0, max_value=25.0, step=0.5,
                         help_text="How far the clip end may move to land on a quiet moment instead of "
                                   "wherever the length budget ran out. Stops clips ending halfway through "
                                   "a move, or a second into the next one - and doubles as the spread "
                                   "control: a small radius only finds shallow dips near the target, so "
                                   "every clip comes out the same length, while a larger one insists on a "
                                   "real break and lets the material decide. Measured on one recording: "
                                   "3s gave lengths within 1.8s of each other, 6s within 11s, 10s within "
                                   "18s. Never goes below the minimum length or above the maximum, and it "
                                   "stays reproducible - the spread comes from the action profile, not from "
                                   "chance. 0 disables it"),
                EnvField("MAX_COMBINED_SCENE_LENGTH", "Max combined scene (s)", "int_auto", 0, min_value=30, max_value=600,
                         help_text="Max length when merging multiple scenes. Auto adjusts based on scene limit"),
            ],
        ),
        EnvSection(
            title="Subtitles",
            expanded=True,
            fields=[
                EnvField("ENABLE_SUBTITLES", "Enable subtitles", "bool", True,
                         help_text="Add animated captions to generated clips"),
                EnvField(
                    "WHISPER_MODEL",
                    "Whisper model",
                    "select",
                    "medium",
                    options=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "turbo"],
                    help_text="Speech recognition model. Larger = more accurate timing but slower. "
                              "large-v3 fits comfortably in 16GB VRAM and is noticeably better on German",
                ),
                EnvField("WHISPER_LANGUAGE", "Spoken language", "select", "",
                         options=["", "de", "en", "es", "fr", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
                         help_text="Language of your commentary. Empty detects it once per clip from the "
                                   "longest speech region. Set it explicitly if detection ever picks wrong"),
                EnvField("WHISPER_USE_VAD", "Detect speech first", "bool", True,
                         help_text="Find the stretches that contain speech and transcribe each on its own. "
                                   "Whisper decodes in 30-second windows and mis-times or drops quiet speech "
                                   "when loud game audio shares the window. Slower, but catches lines that "
                                   "are otherwise lost"),
                EnvField("WHISPER_DROP_FOREIGN_SCRIPT", "Drop foreign-script words", "bool", True,
                         help_text="Remove words written in a script the chosen language does not use. Whisper "
                                   "falls back to Korean or CJK characters when it hallucinates on game noise; "
                                   "those can never be something you actually said"),
                EnvField("VAD_MAX_REGION", "Max speech region (s)", "float", 10.0,
                         min_value=2.0, max_value=30.0, step=1.0,
                         help_text="Stop merging speech regions at this length. Over roughly 15s Whisper starts "
                                   "repeating itself and its word timing drifts"),
                EnvField("VAD_MIN_REGION", "Min speech region (s)", "float", 0.5,
                         min_value=0.0, max_value=3.0, step=0.1,
                         help_text="Skip regions shorter than this - usually game noise mistaken for speech. "
                                   "Do not raise it far: a short interjection like 'Ups' is only half a second"),
                EnvField("VAD_PADDING", "Speech padding (s)", "float", 0.3, min_value=0.0, max_value=2.0, step=0.1,
                         help_text="Extra audio kept around each detected speech region, so words are not "
                                   "clipped at the edges"),
                EnvField("VAD_MERGE_GAP", "Merge speech within (s)", "float", 0.5,
                         min_value=0.0, max_value=5.0, step=0.1,
                         help_text="Join speech regions closer than this. One region per breath would leave "
                                   "Whisper too little context to recognise words"),
            ],
        ),
        EnvSection(
            title="Caption Style & Layout",
            expanded=True,
            fields=[
                EnvField(
                    "PYCAPS_TEMPLATE",
                    "Caption template",
                    "select",
                    "hype",
                    options=["hype", "vibrant", "explosive", "fast", "classic", "minimalist",
                             "neo-minimal", "line-focus", "word-focus", "retro-gaming", "default"],
                    help_text="Look of the captions: font, colours and the animation of the spoken word",
                ),
                EnvField("SUBTITLE_VIDEO_QUALITY", "Caption render quality", "select", "high",
                         options=["low", "middle", "high", "very_high"],
                         help_text="The caption renderer re-encodes every frame, so this decides the quality of "
                                   "the finished clip - the render settings before it are discarded here. "
                                   "high is a good balance; very_high is noticeably slower for little visible gain"),
                EnvField("SUBTITLE_FONT_SIZE", "Caption font size", "int", 0, min_value=0, max_value=120,
                         help_text="0 keeps the template's own size (24 for 'hype'). Captions are rendered at "
                                   "twice this value in pixels, so the limit is the longest single word - it "
                                   "cannot wrap, and once it is wider than the frame it gets cut off at the "
                                   "edges. Around 30 stays safe for long German compounds; check your own "
                                   "longest word. Outline and padding scale with the size"),
                EnvField("SUBTITLE_MAX_LINES", "Max caption lines", "int", 2, min_value=1, max_value=5,
                         help_text="How many lines a caption may wrap to. 1-2 keeps the frame clear; "
                                   "more lines cover a large part of a vertical video"),
                EnvField("SUBTITLE_MIN_LINES", "Min caption lines", "int", 1, min_value=1, max_value=5,
                         help_text="Reserve at least this many lines, so captions do not jump vertically"),
                EnvField("SUBTITLE_MAX_CHARS", "Max chars per caption", "int", 15, min_value=5, max_value=120,
                         help_text="Longer text is split into the next caption. Small values give the fast, "
                                   "word-by-word look typical for Shorts"),
                EnvField("SUBTITLE_SPLIT_GAP", "Split on pause (s)", "float", 0.7,
                         min_value=0.0, max_value=5.0, step=0.1,
                         help_text="Start a new caption when the speaker pauses this long. A caption stays on "
                                   "screen from its first word to its last, so without this a block spanning a "
                                   "pause shows words seconds before they are spoken. 0 disables it"),
                EnvField("SUBTITLE_MIN_CHARS", "Min chars per caption", "int", 10, min_value=1, max_value=120,
                         help_text="Avoids very short leftover captions. Must not exceed the maximum"),
                EnvField("SUBTITLE_WIDTH_RATIO", "Caption width ratio", "float", 0.85,
                         min_value=0.1, max_value=1.0, step=0.05,
                         help_text="How much of the frame width captions may use before wrapping"),
                EnvField("SUBTITLE_OVERFLOW", "When text does not fit", "select", "exceed_lines",
                         options=["exceed_lines", "exceed_width"],
                         help_text="exceed_lines: add another line, so the line limit above is only a target. "
                                   "exceed_width: keep the line count and let the last line run wider"),
                EnvField("SUBTITLE_VERTICAL_ALIGN", "Vertical position", "select", "bottom",
                         options=["bottom", "center", "top"],
                         help_text="Where captions sit in the frame"),
                EnvField("SUBTITLE_VERTICAL_OFFSET", "Vertical offset", "float", -0.1,
                         min_value=-1.0, max_value=1.0, step=0.05,
                         help_text="Nudge captions away from the chosen edge. Negative moves up from the bottom"),
                EnvField("PYCAPS_KEEP_SPLITTERS", "Split long captions", "bool", True,
                         help_text="On: text is chunked by the char limits above. Off: a whole transcript block "
                                   "is shown at once, which matches the SRT exactly but produces walls of text"),
            ],
        ),
        EnvSection(
            title="Decord & Debug",
            fields=[
                EnvField("DEBUG_SKIP_ANALYSIS", "Debug: skip analysis", "bool", False,
                         help_text="Skip video analysis and use cached data (for testing)"),
                EnvField("DEBUG_SKIP_RENDER", "Debug: skip render", "bool", False,
                         help_text="Skip clip rendering (for testing subtitles only)"),
                EnvField("DEBUG_RENDERED_CLIPS", "Debug: rendered clips", "text", "", multiline=True,
                         help_text="Comma-separated list of pre-rendered clip paths (for testing)"),
            ],
        ),
    ]


def _field_map() -> Dict[str, EnvField]:
    return {field.name: field for field in iter_fields()}


def iter_fields() -> Iterable[EnvField]:
    for section in get_schema():
        for field in section.fields:
            yield field


def load_env_values() -> Tuple[Dict[str, str], Dict[str, str]]:
    defaults = {field.name: str(field.default) for field in iter_fields()}
    env_values = dotenv_values(str(ENV_PATH)) if ENV_PATH.exists() else {}
    values = dict(defaults)
    for key, value in env_values.items():
        if value is None:
            continue
        values[key] = value
    extras = {k: v for k, v in env_values.items() if k not in defaults and v is not None}
    return values, extras


def coerce_value(field: EnvField, raw_value: Optional[str]) -> Any:
    if raw_value is None or raw_value == "":
        return field.default
    if field.field_type == "bool":
        return str(raw_value).strip().lower() in TRUTHY
    if field.field_type == "int":
        try:
            return int(raw_value)
        except Exception:
            return field.default
    if field.field_type == "float":
        try:
            return float(raw_value)
        except Exception:
            return field.default
    return str(raw_value)


TRUTHY = ("1", "true", "yes", "on")


def normalize_value(field: EnvField, value: Any) -> str:
    if field.field_type == "bool":
        # Values coming from load_env_values() are strings straight out of the
        # .env file, and the string "false" is truthy in Python. Checking it
        # directly silently flipped every disabled toggle back to true on save,
        # including DEBUG_SKIP_ANALYSIS and DEBUG_SKIP_RENDER - which makes the
        # pipeline skip its actual work while still reporting success.
        if isinstance(value, str):
            return "true" if value.strip().lower() in TRUTHY else "false"
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


_NEEDS_QUOTING = ('#', '"', "'", "\\", "\n", "\r")


def _quote(value: str) -> str:
    """Quote a value that dotenv would otherwise read back differently.

    An unquoted `#` starts an inline comment and surrounding whitespace is
    stripped, so a free-text field such as DEBUG_RENDERED_CLIPS could come
    back truncated on the next load.
    """
    if value == "":
        return value
    if not any(ch in value for ch in _NEEDS_QUOTING) and value == value.strip():
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "")
        .replace("\n", "\n")
    )
    return '"' + escaped + '"'


def save_env_values(values: Dict[str, Any], extras: Optional[Dict[str, str]] = None) -> None:
    extras = extras or {}
    lines: List[str] = []
    for field in iter_fields():
        raw_value = values.get(field.name, field.default)
        lines.append(f"{field.name}={_quote(normalize_value(field, raw_value))}")
    if extras:
        lines.append("")
        lines.append("# Extra settings")
        for key, value in sorted(extras.items()):
            if value is None:
                continue
            lines.append(f"{key}={_quote(str(value))}")

    # Written beside the target and moved into place: the pipeline reads .env
    # while the dashboard may be saving it, and a half-written file makes that
    # run fall back to defaults without any error to show for it.
    payload = "\n".join(lines) + "\n"
    temp_path = ENV_PATH.with_name(ENV_PATH.name + ".saving")
    temp_path.write_text(payload, encoding="utf-8")
    try:
        os.replace(temp_path, ENV_PATH)
    except OSError:
        # Docker bind-mounts .env as a single file, so the path itself is a
        # mount point and cannot be renamed over. Overwrite it in place and
        # accept the small window rather than losing the save entirely.
        try:
            ENV_PATH.write_text(payload, encoding="utf-8")
        finally:
            temp_path.unlink(missing_ok=True)
