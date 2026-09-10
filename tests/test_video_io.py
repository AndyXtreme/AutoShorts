"""Codec handling: which reader a file gets, and which encoder its short gets.

The pipeline used to assume every source was something decord could decode
and every output should be whatever movielite happened to write. An AV1
recording broke the first assumption outright and quietly lost to the second,
so both decisions live in video_io now and are pinned here.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import video_io


@pytest.fixture(autouse=True)
def clear_caches(monkeypatch):
    video_io._PROBE_CACHE.clear()
    monkeypatch.setattr(video_io, "_available_encoders", None, raising=False)
    monkeypatch.delenv("OUTPUT_VIDEO_CODEC", raising=False)


def _probe(codec):
    return video_io.VideoProbe(
        codec=codec, width=2560, height=1440, fps=60.0,
        duration=739.3, frame_count=44358,
    )


# --- rate parsing ---------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [("60/1", 60.0), ("30000/1001", pytest.approx(29.97, abs=0.01)),
     ("0/0", 0.0), ("", 0.0), (None, 0.0), ("25", 25.0)],
)
def test_frame_rates_are_read_as_rationals(raw, expected):
    assert video_io._parse_rate(raw) == expected


# --- reader choice --------------------------------------------------------
@pytest.mark.parametrize("codec", ["h264", "hevc", "vp9"])
def test_decord_handles_the_codecs_it_was_built_for(codec, monkeypatch):
    monkeypatch.setattr(video_io, "probe_video", lambda p: _probe(codec))
    assert video_io.decord_can_read("clip.mkv")


def test_av1_does_not_go_to_decord(monkeypatch):
    """decord's NVDEC parser rejects AV1 and its FFmpeg 4.x cannot decode it."""
    monkeypatch.setattr(video_io, "probe_video", lambda p: _probe("av1"))
    assert not video_io.decord_can_read("clip.mkv")


def test_an_unreadable_file_does_not_go_to_decord(monkeypatch):
    monkeypatch.setattr(video_io, "probe_video", lambda p: _probe(""))
    assert not video_io.decord_can_read("clip.mkv")


# --- output codec ---------------------------------------------------------
@pytest.mark.parametrize(
    "source, expected",
    [("av1", "av1"), ("hevc", "hevc"), ("h264", "h264"),
     ("avc1", "h264"), ("hvc1", "hevc")],
)
def test_the_output_codec_follows_the_source(source, expected):
    assert video_io.target_codec(source) == expected


def test_an_unknown_source_codec_falls_back_to_hevc():
    assert video_io.target_codec("prores") == "hevc"


def test_the_output_codec_can_be_forced(monkeypatch):
    monkeypatch.setenv("OUTPUT_VIDEO_CODEC", "av1")
    assert video_io.target_codec("hevc") == "av1"


def test_a_nonsense_override_still_matches_the_source(monkeypatch):
    monkeypatch.setenv("OUTPUT_VIDEO_CODEC", "vp9")
    assert video_io.target_codec("av1") == "av1"


# --- encoder selection ----------------------------------------------------
def test_hardware_encoders_are_preferred(monkeypatch):
    monkeypatch.setattr(
        video_io, "available_encoders",
        lambda: frozenset({"av1_nvenc", "libsvtav1", "hevc_nvenc", "libx265"}),
    )
    assert video_io.video_encoder_args("av1")[:2] == ["-c:v", "av1_nvenc"]
    assert video_io.video_encoder_args("hevc")[:2] == ["-c:v", "hevc_nvenc"]


def test_software_encoders_are_used_when_nvenc_is_missing(monkeypatch):
    """An image built without NVENC should still produce a file."""
    monkeypatch.setattr(
        video_io, "available_encoders", lambda: frozenset({"libsvtav1", "libx265"})
    )
    assert video_io.video_encoder_args("av1")[:2] == ["-c:v", "libsvtav1"]
    assert video_io.video_encoder_args("hevc")[:2] == ["-c:v", "libx265"]


def test_encoder_parsing_reads_the_ffmpeg_table(monkeypatch):
    table = (
        "Encoders:\n"
        " V..... libaom-av1           libaom AV1 (codec av1)\n"
        " V....D av1_nvenc            NVIDIA NVENC av1 encoder (codec av1)\n"
        " A..... aac                  AAC\n"
    )

    class Result:
        stdout = table

    monkeypatch.setattr(video_io.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(video_io, "_available_encoders", None, raising=False)
    found = video_io.available_encoders()
    assert {"libaom-av1", "av1_nvenc", "aac"} <= found
    assert "Encoders:" not in found


# --- the final pass -------------------------------------------------------
def test_a_clip_already_in_the_right_codec_is_left_alone(tmp_path, monkeypatch):
    """Re-encoding an HEVC clip to HEVC would cost a generation for nothing."""
    clip = tmp_path / "scene-0.mp4"
    clip.write_bytes(b"not really a video")
    source = tmp_path / "source.mkv"
    source.write_bytes(b"not really a video either")

    monkeypatch.setattr(video_io, "probe_video", lambda p: _probe("hevc"))
    called = []
    monkeypatch.setattr(video_io.subprocess, "run", lambda *a, **k: called.append(a))

    assert video_io.match_output_codec(clip, source) is False
    assert called == []


def test_a_missing_clip_is_not_an_error(tmp_path):
    assert video_io.match_output_codec(tmp_path / "gone.mp4", tmp_path / "src.mkv") is False


def test_a_failed_re_encode_keeps_the_original(tmp_path, monkeypatch):
    clip = tmp_path / "scene-0.mp4"
    clip.write_bytes(b"the original")
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")

    def probe(path):
        return _probe("av1" if Path(path).name == "source.mkv" else "h264")

    monkeypatch.setattr(video_io, "probe_video", probe)
    monkeypatch.setattr(video_io, "available_encoders", lambda: frozenset({"av1_nvenc"}))

    class Failed:
        returncode = 1
        stderr = "nvenc unavailable"

    monkeypatch.setattr(video_io.subprocess, "run", lambda *a, **k: Failed())

    assert video_io.match_output_codec(clip, source) is False
    assert clip.read_bytes() == b"the original"
    assert not list(tmp_path.glob("*.recode.*"))
