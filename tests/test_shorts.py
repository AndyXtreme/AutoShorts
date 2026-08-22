import sys
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

# --- Mock GPU libraries BEFORE importing shorts ---
# We must mock decord, cupy, torchaudio, torch so that shorts.py can be imported
# even if these libraries are missing or if we are on a CPU-only node.

# Mock torch
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.device.return_value = "cpu"
torch_mock.tensor = lambda x, **kwargs: MagicMock()
# Mock basic tensor ops used in shorts
torch_mock.abs = MagicMock()
torch_mock.mean = MagicMock()
torch_mock.sqrt = MagicMock()
torch_mock.cat = MagicMock()
torch_mock.from_numpy = lambda x: x
torch_mock.from_dlpack = MagicMock()
torch_mock.to_dlpack = MagicMock()
torch_mock.nn.functional.interpolate = MagicMock()
sys.modules["torch"] = torch_mock

# Mock torchaudio
torchaudio_mock = MagicMock()
sys.modules["torchaudio"] = torchaudio_mock

# Mock decord
decord_mock = MagicMock()
decord_mock.bridge.set_bridge = MagicMock()
decord_mock.cpu = lambda x: f"cpu({x})"
decord_mock.gpu = lambda x: f"gpu({x})"
sys.modules["decord"] = decord_mock

# Mock cupy
cupy_mock = MagicMock()
cupy_mock.asarray = MagicMock(side_effect=lambda x: x)
cupy_mock.asnumpy = MagicMock(side_effect=lambda x: x)
cupy_mock.from_dlpack = MagicMock()
cupy_mock.to_dlpack = MagicMock()
sys.modules["cupy"] = cupy_mock

# Mock cupyx
cupyx_mock = MagicMock()
sys.modules["cupyx"] = cupyx_mock
sys.modules["cupyx.scipy"] = MagicMock()
sys.modules["cupyx.scipy.ndimage"] = MagicMock()


# Ensure the src directory is on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import shorts AFTER mocking
import shorts  # noqa: E402
from shorts import (  # noqa: E402
    blur_gpu,
    combine_scenes,
    select_background_resolution,
    ProcessingConfig,
    analysis_settings,
    _SecondsTime,
)


# Helper to create scene tuples
def make_scene(start: float, end: float):
    return (_SecondsTime(start), _SecondsTime(end))


def test_select_background_resolution(monkeypatch):
    monkeypatch.delenv("MAX_OUTPUT_HEIGHT", raising=False)
    # Assumed 9:16 when only a width is given.
    assert select_background_resolution(700) == (720, 1280)
    assert select_background_resolution(1000) == (1080, 1920)


def test_resolution_does_not_downscale_a_tall_crop(monkeypatch):
    """A 9:16 crop out of 2560x1440 is 810x1440 - it must not land at 720p.

    Selecting by width used to put 810 below the 840px threshold and render
    1440 rows of source into 1280.
    """
    monkeypatch.setenv("MAX_OUTPUT_HEIGHT", "1920")
    assert select_background_resolution(810, 1440) == (1080, 1920)


def test_resolution_respects_max_output_height(monkeypatch):
    monkeypatch.setenv("MAX_OUTPUT_HEIGHT", "1280")
    assert select_background_resolution(810, 1440) == (720, 1280)
    monkeypatch.setenv("MAX_OUTPUT_HEIGHT", "3840")
    assert select_background_resolution(1215, 2160) == (1440, 2560)


def test_action_score_mode(monkeypatch):
    """mean scores intensity per second, sum totals it over the scene."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    long_scene = make_scene(0.0, 4.0)
    short_scene = make_scene(0.0, 2.0)

    monkeypatch.setenv("ACTION_SCORE_MODE", "sum")
    assert shorts.scene_action_score(long_scene, times, scores, w_audio=1.0, w_video=0.0) >         shorts.scene_action_score(short_scene, times, scores, w_audio=1.0, w_video=0.0)

    monkeypatch.setenv("ACTION_SCORE_MODE", "mean")
    assert shorts.scene_action_score(long_scene, times, scores, w_audio=1.0, w_video=0.0) ==         shorts.scene_action_score(short_scene, times, scores, w_audio=1.0, w_video=0.0)


def test_pick_window_length_is_deterministic_by_default(monkeypatch):
    monkeypatch.delenv("CLIP_LENGTH_MODE", raising=False)
    assert shorts._pick_window_length(15, 59) == 59
    monkeypatch.setenv("CLIP_LENGTH_MODE", "max")
    assert shorts._pick_window_length(15, 59) == 59


def test_blur_gpu_delegates_to_the_torch_implementation():
    """The blur is pure PyTorch now; it used to round-trip through CuPy.

    The old assertions checked for torch.to_dlpack / cupy calls that the
    current implementation never makes - they passed only because nothing ran
    the suite.
    """
    image = MagicMock()
    with patch.object(shorts, "gaussian_blur_torch", return_value="blurred") as blur:
        assert blur_gpu(image, sigma=4.0) == "blurred"
    blur.assert_called_once_with(image, 4.0)


def test_combine_scenes_merges_short_scenes():
    config = ProcessingConfig(min_short_length=5, max_short_length=10, max_combined_scene_length=15)
    scenes = [
        make_scene(0, 5),
        make_scene(5, 7),
        make_scene(7, 9),
        make_scene(9, 11),
        make_scene(11, 13),
        make_scene(13, 18),
    ]
    combined = combine_scenes(scenes, config)
    assert len(combined) == 1
    start, end = combined[0]
    assert start.get_seconds() == 5
    assert end.get_seconds() == 13

# render_video (legacy) has been removed.
# render_video_gpu logic is verified via mocks in separate flows or implicitly here if we add such tests.


def test_analysis_settings_come_from_the_environment(monkeypatch):
    """Replaces a test for compute_video_action_profile, which no longer exists.

    That function was folded into analyze_video_content when scene detection
    and motion profiling were merged into a single decode pass, and the test
    kept importing it - so the whole suite failed to collect and had in fact
    never run. Its subject (batch-wise reading) is now an implementation detail
    of a function that needs a real decoder; what stayed worth asserting is
    that the knobs reach it.
    """
    monkeypatch.delenv("SCENE_THRESHOLD", raising=False)
    monkeypatch.delenv("ACTION_FPS", raising=False)
    assert analysis_settings() == (27.0, 6)

    monkeypatch.setenv("SCENE_THRESHOLD", "18.5")
    monkeypatch.setenv("ACTION_FPS", "12")
    assert analysis_settings() == (18.5, 12)

    # Explicit arguments still win over the environment.
    assert analysis_settings(30.0, 4) == (30.0, 4)

    # A nonsensical sampling rate is clamped rather than dividing by zero.
    monkeypatch.setenv("ACTION_FPS", "0")
    assert analysis_settings()[1] == 1


def test_main_processes_only_the_files_it_is_given(tmp_path, monkeypatch):
    """`run.py <file>` used to be silently ignored and scan the whole folder."""
    wanted = tmp_path / "wanted.mkv"
    wanted.write_bytes(b"")
    (tmp_path / "other.mkv").write_bytes(b"")

    processed = []
    monkeypatch.setattr(shorts, "process_video",
                        lambda video, config, out: processed.append(Path(video).name))
    monkeypatch.setattr(shorts, "config_from_env", lambda: MagicMock())
    monkeypatch.chdir(tmp_path)

    shorts.main([str(wanted)])
    assert processed == ["wanted.mkv"]


def test_main_skips_files_that_are_not_videos(tmp_path, monkeypatch):
    note = tmp_path / "notes.txt"
    note.write_text("not a video")

    processed = []
    monkeypatch.setattr(shorts, "process_video",
                        lambda video, config, out: processed.append(video))
    monkeypatch.setattr(shorts, "config_from_env", lambda: MagicMock())
    monkeypatch.chdir(tmp_path)

    shorts.main([str(note)])
    assert processed == []
