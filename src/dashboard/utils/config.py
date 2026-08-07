from __future__ import annotations

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
                EnvField("MAX_OUTPUT_HEIGHT", "Max output height (px)", "int", 1920, min_value=480, max_value=3840,
                         help_text="Caps the rendered resolution. 1920 gives 1080x1920 at 9:16"),
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
                EnvField("MIN_SHORT_LENGTH", "Min short length (s)", "int_auto", 15, min_value=5, max_value=120,
                         help_text="Shortest allowed clip. Auto mode picks based on video length"),
                EnvField("MAX_SHORT_LENGTH", "Max short length (s)", "int_auto", 59, min_value=15, max_value=300,
                         help_text="Longest allowed clip. 59s keeps clips inside the YouTube Shorts limit"),
                EnvField("MAX_COMBINED_SCENE_LENGTH", "Max combined scene (s)", "int_auto", 0, min_value=30, max_value=600,
                         help_text="Max length when merging multiple scenes. Auto adjusts based on scene limit"),
            ],
        ),
        EnvSection(
            title="AI Providers",
            expanded=True,
            fields=[
                EnvField("AI_PROVIDER", "AI provider", "select", "local", options=["local", "openai", "gemini"],
                         help_text="local = heuristics only, no API calls and no cost. openai/gemini rank scenes "
                                   "semantically (funny, clutch, epic fail) but send clips to that provider"),
                EnvField(
                    "VIDEO_TYPE",
                    "Video type",
                    "select",
                    "gaming",
                    options=[
                        "gaming", "podcasts", "entertainment", "sports", "vlogs",
                        "tv_shows", "documentaries", "music", "educational",
                        "interviews", "comedy", "news_commentary", "esports",
                        "cooking_diy", "fitness",
                    ],
                    help_text="Primary source content category used for AI clipping strategy and caption style presets",
                ),
                EnvField("AI_ANALYSIS_ENABLED", "AI analysis enabled", "bool", True,
                         help_text="Use AI to rank scenes by content quality. Disable for faster but less smart selection"),
                EnvField("GEMINI_API_KEY", "Gemini API key", "password", "",
                         help_text="Your Google Gemini API key (get from ai.google.dev)"),
                EnvField("OPENAI_API_KEY", "OpenAI API key", "password", "",
                         help_text="Your OpenAI API key (get from platform.openai.com)"),
                EnvField("GEMINI_MODEL", "Gemini model", "select", "gemini-3-flash-preview",
                         options=["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
                         help_text="Gemini model for video analysis"),
                EnvField("GEMINI_DEEP_ANALYSIS", "Enable Gemini Deep Analysis", "bool", False,
                         help_text="Upload entire video for better context (slow initial upload)"),
                EnvField("OPENAI_MODEL", "OpenAI model", "select", "gpt-5-mini",
                         options=["gpt-5-mini", "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                         help_text="OpenAI model for video analysis"),
                EnvField("OPENAI_TAGGING_MODEL", "OpenAI tagging model", "select", "gpt-5-mini",
                         options=["gpt-5-mini", "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                         help_text="Model for caption generation (separate from video analysis). Gemini uses same model for both"),
                EnvField("AI_SCORE_WEIGHT", "AI score weight", "float", 0.7, min_value=0.0, max_value=1.0, step=0.05,
                         help_text="How much AI ranking matters vs action detection. 1.0 = AI only, 0 = action only"),
            ],
        ),
        EnvSection(
            title="Semantic Analysis",
            fields=[
                EnvField("CANDIDATE_CLIP_COUNT", "Candidate clip count", "int_auto", 0, min_value=10, max_value=200,
                         help_text="How many clips to send to AI for ranking. More = better selection but higher cost"),
                EnvField("CANDIDATE_CLIP_DURATION", "Candidate clip duration (s)", "int", 120, min_value=10, max_value=600,
                         help_text="Gemini only: Length of video clips for analysis. OpenAI uses frames instead"),
            ],
        ),
        EnvSection(
            title="Subtitles",
            expanded=True,
            fields=[
                EnvField("ENABLE_SUBTITLES", "Enable subtitles", "bool", True,
                         help_text="Add animated captions to generated clips"),
                EnvField(
                    "SUBTITLE_MODE",
                    "Subtitle mode",
                    "select",
                    "speech",
                    options=["speech", "ai_captions", "none"],
                    help_text="speech = transcribe what is actually said (local, via Whisper). "
                              "ai_captions = AI-written commentary, requires an OpenAI or Gemini key",
                ),
                EnvField(
                    "WHISPER_MODEL",
                    "Whisper model",
                    "select",
                    "medium",
                    options=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "turbo"],
                    help_text="Speech recognition model. Larger = more accurate timing but slower. "
                              "large-v3 fits comfortably in 16GB VRAM and is noticeably better on German",
                ),
                EnvField("MAX_CAPTIONS", "Max captions", "int", 0, min_value=0, max_value=50,
                         help_text="Maximum captions per clip. 0 = Auto (dynamic based on video duration)"),
                EnvField("ENABLE_AI_CAPTION_ENHANCEMENT", "AI caption enhancement", "bool", False,
                         help_text="Rewrite captions to be punchier. Needs an AI provider, ignored in local mode"),
                EnvField("ENABLE_CAPTION_EMOJIS", "Caption emojis", "bool", False,
                         help_text="Sprinkle emojis into captions. Needs an AI provider, ignored in local mode"),
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
                EnvField(
                    "CAPTION_STYLE",
                    "Caption tone",
                    "select",
                    "auto",
                    options=["auto", "gaming", "funny", "dramatic", "esports_playcast", "comedy_punchline",
                             "vlog_story", "podcast_quote", "educational_explainer"],
                    help_text="Wording style for AI-written captions. Only used in ai_captions mode",
                ),
            ],
        ),
        EnvSection(
            title="TTS Voiceover",
            fields=[
                EnvField("ENABLE_TTS", "Enable TTS", "bool", False,
                         help_text="Overlay an AI voiceover. Off keeps the original audio, which is usually what "
                                   "you want for gameplay, and saves VRAM and processing time"),
                EnvField("TTS_MODEL", "TTS model", "select", "qwen", options=["qwen"],
                         help_text="Text-to-speech model (Qwen is fast and high quality)"),
                EnvField("TTS_LANGUAGE", "Voiceover language", "select", "de",
                         options=["de", "en", "es", "fr", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
                         help_text="Language the generated voiceover speaks"),
                EnvField("TTS_SPEED", "Voiceover speed", "float", 1.0, min_value=0.5, max_value=2.0, step=0.05,
                         help_text="Playback rate of the voiceover. 1.0 = normal"),
                EnvField("TTS_DEVICE", "TTS device", "select", "cuda", options=["cuda", "cpu"],
                         help_text="cuda = GPU (fast), cpu = processor (slower but works everywhere)"),
                EnvField(
                    "TTS_VOICE_DESCRIPTION",
                    "Voice description override",
                    "text",
                    "",
                    help_text="Custom voice style (e.g. 'energetic male gamer'). Leave blank for auto",
                    multiline=True,
                ),
                EnvField("TTS_GAME_AUDIO_VOLUME", "Game audio volume", "float", 0.3, min_value=0.0, max_value=1.0, step=0.05,
                         help_text="How loud the original game audio is when voiceover plays"),
                EnvField("TTS_VOICEOVER_VOLUME", "Voiceover volume", "float", 1.0, min_value=0.0, max_value=2.0, step=0.05,
                         help_text="How loud the AI voiceover narration is"),
            ],
        ),
        EnvSection(
            title="Decord & Debug",
            fields=[
                EnvField("DECORD_EOF_RETRY_MAX", "Decord EOF retry max", "int", 65536, min_value=1, max_value=200000,
                         help_text="Max retries for video decoding errors (increase if videos fail to load)"),
                EnvField("DECORD_SKIP_TAIL_FRAMES", "Decord skip tail frames", "int", 0, min_value=0, max_value=1000,
                         help_text="Skip frames at end of video (helps with corrupted endings)"),
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


def save_env_values(values: Dict[str, Any], extras: Optional[Dict[str, str]] = None) -> None:
    extras = extras or {}
    field_map = _field_map()
    lines: List[str] = []
    for field in iter_fields():
        raw_value = values.get(field.name, field.default)
        lines.append(f"{field.name}={normalize_value(field, raw_value)}")
    if extras:
        lines.append("")
        lines.append("# Extra settings")
        for key, value in sorted(extras.items()):
            if value is None:
                continue
            lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
