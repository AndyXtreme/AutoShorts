"""Reading and writing video regardless of the codec it happens to be in.

decord decodes H.264 and HEVC through NVDEC and is the fast path, but it
cannot touch AV1 at all: its NVDEC parser has no AV1 entry ("Invalid codec:
av1") and the FFmpeg 4.x libraries it is linked against - the version its API
requires - have no AV1 decoder either. AV1 sources therefore read through an
FFmpeg pipe instead.

The same split decides how the finished short is written. A source is
recorded in one codec because that is what the capture card or the platform
produces, and a pipeline that silently hands back H.264 throws that choice
away, so the encoder is chosen to match the source unless OUTPUT_VIDEO_CODEC
overrides it.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Binaries
# --------------------------------------------------------------------------
def ffmpeg_binary() -> str:
    return os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_binary() -> str:
    return os.getenv("FFPROBE_BINARY") or shutil.which("ffprobe") or "ffprobe"


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoProbe:
    codec: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int


_PROBE_CACHE: Dict[Tuple[str, int, int], VideoProbe] = {}


def _parse_rate(value: Optional[str]) -> float:
    """Turn FFmpeg's "60/1" rational into a float, tolerating "0/0"."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe_video(path) -> VideoProbe:
    """Read codec, size, rate and length of a file's first video stream.

    Cached on path, size and mtime: every reader and every encoder choice asks
    for this, and ffprobe on a large file is not free.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), int(stat.st_mtime), stat.st_size)
    except OSError:
        key = (str(path), 0, 0)
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached

    cmd = [
        ffprobe_binary(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError) as exc:
        logging.warning(f"ffprobe failed on {path.name}: {exc}")
        data = {}

    streams = data.get("streams") or [{}]
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps <= 0:
        fps = 30.0
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    # Matroska rarely carries nb_frames. Counting packets is exact but reads
    # the whole file, so derive it from the duration instead - the readers
    # clamp to what the decoder actually hands over anyway.
    try:
        frames = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    if frames <= 0:
        frames = int(duration * fps)

    probe = VideoProbe(
        codec=str(stream.get("codec_name") or "").lower(),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
        duration=duration,
        frame_count=max(0, frames),
    )
    _PROBE_CACHE[key] = probe
    return probe


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
# What the bundled decord build can decode. Kept as an allow-list rather than
# a deny-list: guessing wrong the other way costs a hard failure deep inside a
# render, while a needless FFmpeg pipe only costs some speed.
DECORD_CODECS = frozenset({"h264", "hevc", "mpeg4", "mpeg2video", "vp8", "vp9"})


def decord_can_read(path) -> bool:
    return probe_video(path).codec in DECORD_CODECS


class FfmpegVideoReader:
    """A decord-shaped reader that decodes through an FFmpeg pipe.

    Only what the pipeline actually uses is implemented: len(), get_avg_fps(),
    indexing and get_batch(). Frames arrive in decode order down a pipe, so a
    request that runs backwards or skips far ahead restarts FFmpeg with a seek.
    Both callers - the analysis loop and the render loop - walk forward, which
    is the case this is built for.
    """

    # Restarting costs a seek plus a keyframe decode; skipping costs one pipe
    # read per frame. Around a second of footage the seek becomes the cheaper
    # of the two.
    _SEEK_AHEAD_THRESHOLD = 60

    def __init__(
        self,
        path,
        width: Optional[int] = None,
        height: Optional[int] = None,
        device: str = "cpu",
        pix_fmt: str = "rgb24",
    ) -> None:
        self._path = Path(path)
        self._pix_fmt = pix_fmt
        self._probe = probe_video(self._path)
        self._width = int(width or self._probe.width)
        self._height = int(height or self._probe.height)
        if self._width <= 0 or self._height <= 0:
            raise ValueError(f"No decodable video stream in {self._path}")
        self._device = device
        self._frame_bytes = self._width * self._height * 3
        self._proc: Optional[subprocess.Popen] = None
        self._cursor = 0
        self._last: Optional[torch.Tensor] = None

    # -- lifecycle ---------------------------------------------------------
    def _spawn(self, index: int) -> None:
        self.close()
        cmd = [ffmpeg_binary(), "-nostdin", "-loglevel", "error"]
        if index > 0:
            cmd += ["-ss", f"{index / self._probe.fps:.6f}"]
        cmd += ["-i", str(self._path), "-map", "0:v:0"]
        if (self._width, self._height) != (self._probe.width, self._probe.height):
            cmd += ["-vf", f"scale={self._width}:{self._height}"]
        cmd += ["-f", "rawvideo", "-pix_fmt", self._pix_fmt, "-"]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._cursor = index

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- decord surface ----------------------------------------------------
    def __len__(self) -> int:
        return self._probe.frame_count

    def get_avg_fps(self) -> float:
        return self._probe.fps

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._to_device(self._frame_at(int(index)))

    def get_batch(self, indices: Iterable[int]) -> torch.Tensor:
        frames = [self._frame_at(int(i)) for i in indices]
        if not frames:
            return torch.empty(
                (0, self._height, self._width, 3), dtype=torch.uint8
            )
        return self._to_device(torch.stack(frames))

    # -- internals ---------------------------------------------------------
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device) if self._device != "cpu" else tensor

    def _read_one(self) -> Optional[torch.Tensor]:
        if self._proc is None or self._proc.stdout is None:
            return None
        buf = self._proc.stdout.read(self._frame_bytes)
        if not buf or len(buf) < self._frame_bytes:
            return None
        array = np.frombuffer(buf, dtype=np.uint8).reshape(
            self._height, self._width, 3
        )
        return torch.from_numpy(array.copy())

    def _frame_at(self, index: int) -> torch.Tensor:
        limit = len(self) - 1
        if limit >= 0:
            index = max(0, min(index, limit))
        else:
            index = max(0, index)

        # The render loop maps output frames onto source frames, so the same
        # index can come round twice when the rates differ.
        if self._last is not None and index == self._cursor - 1:
            return self._last

        if (
            self._proc is None
            or index < self._cursor
            or index > self._cursor + self._SEEK_AHEAD_THRESHOLD
        ):
            self._spawn(index)

        while self._cursor <= index:
            frame = self._read_one()
            if frame is None:
                # Ran off the end: nb_frames is derived from the duration and
                # can overshoot by a frame or two. Repeating the last frame
                # keeps batch shapes intact and reads as no motion, which is
                # the truth about footage that is not there.
                if self._last is None:
                    raise RuntimeError(f"No frames decoded from {self._path}")
                return self._last
            self._cursor += 1
            self._last = frame
        return self._last  # type: ignore[return-value]


# What the FFmpeg bundled with the opencv-python wheel decodes - OpenCV 4.9,
# avcodec 59, no AV1 decoder. movielite reads every frame of the caption
# stage through it, and an AV1 clip came back as 1902 failed reads and a
# black video with only the captions on it.
OPENCV_CODECS = frozenset({"h264", "hevc", "mpeg4", "mpeg2video", "vp8", "vp9"})


class FfmpegCapture:
    """The part of cv2.VideoCapture that movielite uses, decoding through the
    image's own FFmpeg instead of OpenCV's: open, read properties, seek by
    frame index, read, release. Frames come back as BGR uint8 arrays, as
    OpenCV would return them."""

    def __init__(self, path) -> None:
        import cv2

        self._cv2 = cv2
        self._probe = probe_video(path)
        self._reader: Optional[FfmpegVideoReader] = None
        if self._probe.width > 0 and self._probe.height > 0:
            self._reader = FfmpegVideoReader(path, pix_fmt="bgr24")
        self._next = 0

    def isOpened(self) -> bool:
        return self._reader is not None

    def get(self, prop) -> float:
        cv2 = self._cv2
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: self._probe.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self._probe.height,
            cv2.CAP_PROP_FPS: self._probe.fps,
            cv2.CAP_PROP_FRAME_COUNT: self._probe.frame_count,
            cv2.CAP_PROP_POS_FRAMES: self._next,
        }
        return float(values.get(prop, 0.0))

    def set(self, prop, value) -> bool:
        if prop == self._cv2.CAP_PROP_POS_FRAMES:
            self._next = max(0, int(value))
            return True
        return False

    def read(self):
        if self._reader is None or self._next >= self._probe.frame_count:
            return False, None
        try:
            frame = self._reader[self._next]
        except RuntimeError:
            return False, None
        self._next += 1
        return True, np.asarray(frame)

    def release(self) -> None:
        if self._reader is not None:
            self._reader.close()


def open_capture(path):
    """cv2.VideoCapture for codecs OpenCV can decode, FfmpegCapture otherwise.

    HEVC and H.264 stay on OpenCV, so their captioned output is exactly what
    it was before AV1 needed a way through."""
    import cv2

    codec = probe_video(path).codec
    if codec in OPENCV_CODECS:
        return cv2.VideoCapture(str(path))
    logging.info(
        f"{Path(path).name} is {codec or 'an unknown codec'}, which OpenCV cannot "
        "decode - reading it through FFmpeg."
    )
    return FfmpegCapture(path)


def open_reader(
    path,
    want_gpu: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_threads: Optional[int] = None,
):
    """Open `path` with decord where possible, through FFmpeg where not.

    Returns an object with decord's reader surface either way, so callers do
    not have to care which one they got.
    """
    path = Path(path)
    on_gpu = bool(want_gpu and torch.cuda.is_available())

    if decord_can_read(path):
        from decord import VideoReader, cpu, gpu

        kwargs = {}
        if width and height:
            kwargs.update(width=int(width), height=int(height))
        if num_threads:
            kwargs["num_threads"] = int(num_threads)
        if on_gpu:
            try:
                return VideoReader(str(path), ctx=gpu(0), **kwargs)
            except Exception as exc:
                logging.warning(f"decord GPU open failed: {exc}. Falling back to CPU.")
        return VideoReader(str(path), ctx=cpu(0), **kwargs)

    logging.info(
        f"{path.name} is {probe_video(path).codec or 'an unknown codec'}, "
        "which decord cannot decode - reading it through FFmpeg."
    )
    return FfmpegVideoReader(
        path,
        width=width,
        height=height,
        device="cuda" if on_gpu else "cpu",
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
# Hardware first, software second. Each entry is tried in order against what
# the FFmpeg build actually reports, so an image without NVENC still produces
# a file rather than an error.
_ENCODER_CHAINS: Dict[str, Tuple[Tuple[str, List[str]], ...]] = {
    "av1": (
        ("av1_nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "30",
                       "-maxrate", "40M", "-bufsize", "60M"]),
        ("libsvtav1", ["-preset", "6", "-crf", "30"]),
        ("libaom-av1", ["-cpu-used", "6", "-crf", "30", "-b:v", "0"]),
    ),
    "hevc": (
        ("hevc_nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "23",
                        "-maxrate", "80M", "-bufsize", "100M"]),
        ("libx265", ["-preset", "medium", "-crf", "23"]),
    ),
    "h264": (
        ("h264_nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "21",
                        "-maxrate", "80M", "-bufsize", "100M"]),
        ("libx264", ["-preset", "medium", "-crf", "20"]),
    ),
}

_SOURCE_ALIASES = {
    "av1": "av1",
    "libaom-av1": "av1",
    "libdav1d": "av1",
    "av01": "av1",
    "hevc": "hevc",
    "h265": "hevc",
    "hvc1": "hevc",
    "h264": "h264",
    "avc1": "h264",
    "avc": "h264",
}

_available_encoders: Optional[frozenset] = None


def available_encoders() -> frozenset:
    """Encoder names this FFmpeg build reports, queried once per process."""
    global _available_encoders
    if _available_encoders is None:
        names = set()
        try:
            out = subprocess.run(
                [ffmpeg_binary(), "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=False,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == 6:
                    names.add(parts[1])
        except OSError as exc:
            logging.warning(f"Could not list FFmpeg encoders: {exc}")
        _available_encoders = frozenset(names)
    return _available_encoders


def target_codec(source_codec: str) -> str:
    """Which codec a short made from `source_codec` should be written in."""
    requested = os.getenv("OUTPUT_VIDEO_CODEC", "auto").strip().lower()
    if requested in _ENCODER_CHAINS:
        return requested
    if requested not in ("", "auto", "source", "match"):
        logging.warning(
            f"Unknown OUTPUT_VIDEO_CODEC '{requested}', matching the source instead. "
            f"Valid: auto, {', '.join(sorted(_ENCODER_CHAINS))}."
        )
    return _SOURCE_ALIASES.get((source_codec or "").lower(), "hevc")


def target_codec_for(source_path) -> str:
    return target_codec(probe_video(source_path).codec)


# The chains above are tuned for "high". Other quality levels shift the
# quantiser and, on NVENC, the preset - p1 is fastest, p7 slowest and best.
_QUALITY_TUNING: Dict[str, Tuple[int, str]] = {
    "low": (6, "p3"),
    "middle": (3, "p4"),
    "high": (0, "p5"),
    "very_high": (-3, "p6"),
}


def _apply_quality(args: List[str], quality: Optional[str]) -> List[str]:
    """Shift an encoder's quantiser and preset to match a quality level."""
    if not quality:
        return args
    tuning = _QUALITY_TUNING.get(quality.strip().lower())
    if tuning is None:
        logging.warning(
            f"Unknown quality '{quality}', encoding at the default instead. "
            f"Valid: {', '.join(_QUALITY_TUNING)}."
        )
        return args

    offset, nvenc_preset = tuning
    tuned = list(args)
    for flag in ("-cq", "-crf"):
        if flag in tuned:
            index = tuned.index(flag) + 1
            try:
                tuned[index] = str(max(0, int(tuned[index]) + offset))
            except (ValueError, IndexError):
                pass
    # Only NVENC uses the pN scale; libx264/libx265/SVT-AV1 presets are named
    # or numeric and are left as the chain set them.
    if "-preset" in tuned:
        index = tuned.index("-preset") + 1
        if index < len(tuned) and tuned[index].startswith("p") and tuned[index][1:].isdigit():
            tuned[index] = nvenc_preset
    return tuned


# Every stage before the last re-encodes a clip that gets encoded again, so
# those stages run a step above the final quality: the extra generation should
# cost as little as possible of what the viewer ends up seeing.
INTERMEDIATE_QUALITY = "very_high"


def video_encoder_candidates(codec: str, quality: Optional[str] = None) -> List[List[str]]:
    """Output arguments for every encoder of `codec` this build has, best first.

    For callers that can retry: NVENC can be listed by the build and still
    fail at runtime (no GPU in the container, sessions exhausted), and the
    software encoder of the same codec is the right fallback - not a
    different codec.
    """
    chain = _ENCODER_CHAINS.get(codec) or _ENCODER_CHAINS["hevc"]
    encoders = available_encoders()
    found = [
        ["-c:v", name] + _apply_quality(args, quality)
        for name, args in chain
        if not encoders or name in encoders
    ]
    if found:
        return found
    name, args = chain[-1]
    logging.warning(f"No encoder for {codec} found in this FFmpeg build; trying {name}.")
    return [["-c:v", name] + _apply_quality(args, quality)]


def video_encoder_args(codec: str, quality: Optional[str] = None) -> List[str]:
    """FFmpeg output arguments for `codec`, hardware where the build has it."""
    return video_encoder_candidates(codec, quality)[0]


def codec_of(path) -> str:
    """The codec a stage that only sees a clip should write it back in: the
    one it is already in, or OUTPUT_VIDEO_CODEC when that is forced."""
    return target_codec(probe_video(path).codec)


def match_output_codec(clip_path: Path, source_path: Path) -> bool:
    """Re-encode a finished clip so it matches the codec of its source.

    Every stage now writes the target codec itself, so this normally finds
    nothing to do. It stays as the safety net for a stage that falls back -
    or a new one that forgets - and would otherwise ship the wrong codec.

    Returns True when the clip was rewritten.
    """
    clip_path = Path(clip_path)
    if not clip_path.exists():
        return False

    wanted = target_codec_for(source_path)
    current = _SOURCE_ALIASES.get(probe_video(clip_path).codec)
    if current == wanted:
        return False

    temp_path = clip_path.with_name(f"{clip_path.stem}.recode{clip_path.suffix}")
    cmd = [ffmpeg_binary(), "-y", "-loglevel", "error", "-i", str(clip_path)]
    cmd += video_encoder_args(wanted)
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(temp_path)]

    logging.info(f"Re-encoding {clip_path.name} to {wanted} to match its source")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
        logging.warning(
            f"Could not re-encode {clip_path.name} to {wanted}, keeping it as it is: "
            f"{result.stderr.strip()[:300]}"
        )
        temp_path.unlink(missing_ok=True)
        return False

    os.replace(temp_path, clip_path)
    return True
