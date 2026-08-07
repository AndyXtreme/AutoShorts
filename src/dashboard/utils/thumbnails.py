from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from moviepy import VideoFileClip


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    size_mb: float
    resolution: str
    thumbnail: Optional[Path]


THUMB_DIR = Path("generated/.thumbnails")


def _ensure_thumb_dir() -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)


def _thumb_name(video_path: Path) -> str:
    """Cache name that survives clips being grouped into per-source folders.

    Clips are named scene-0.mp4, scene-1.mp4 ... inside a folder per source
    video, so the stem alone is no longer unique - every recording would fight
    over the same scene-0.png. Including the parent folder keeps them apart.
    """
    parent = video_path.parent.name or "root"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{parent}__{video_path.stem}")
    return f"{safe}.png"


def build_thumbnail(video_path: Path) -> Optional[Path]:
    _ensure_thumb_dir()
    thumb_path = THUMB_DIR / _thumb_name(video_path)
    if thumb_path.exists():
        return thumb_path
    try:
        with VideoFileClip(str(video_path)) as clip:
            frame_time = min(1.0, clip.duration / 2.0) if clip.duration else 0.0
            frame = clip.get_frame(frame_time)
            clip.save_frame(str(thumb_path), t=frame_time)
        return thumb_path
    except Exception:
        return None


def get_video_info(video_path: Path) -> VideoInfo:
    try:
        with VideoFileClip(str(video_path)) as clip:
            duration = clip.duration or 0.0
            size_mb = video_path.stat().st_size / (1024 * 1024)
            resolution = f"{clip.w}x{clip.h}"
    except Exception:
        duration = 0.0
        size_mb = video_path.stat().st_size / (1024 * 1024)
        resolution = "unknown"
    thumbnail = build_thumbnail(video_path)
    return VideoInfo(
        path=video_path,
        duration=duration,
        size_mb=size_mb,
        resolution=resolution,
        thumbnail=thumbnail,
    )


def list_videos(folder: Path, recursive: bool = False) -> List[VideoInfo]:
    """List videos in a folder.

    Pass recursive=True for the output folder, where clips live in one
    subfolder per source video. The input queue stays flat on purpose - its
    .disabled subfolder holds videos that are deliberately excluded.
    """
    if not folder.exists():
        return []
    video_paths = []
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    for path in candidates:
        # Never surface the thumbnail cache or other dot-folders as content.
        if any(part.startswith(".") for part in path.relative_to(folder).parts[:-1]):
            continue
        # Check if it's a regular file or a symlink pointing to a file
        if (path.is_file() or path.is_symlink()) and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}:
            # Verify symlink points to existing file
            if path.is_symlink():
                try:
                    if not path.resolve().exists():
                        continue  # Skip broken symlinks
                except OSError:
                    continue  # Skip inaccessible symlinks
            video_paths.append(path)
    return [get_video_info(path) for path in sorted(video_paths)]
