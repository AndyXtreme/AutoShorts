import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
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
    snap_clip_end,
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


# --- where a clip ends -----------------------------------------------------
def _burst_profile(start: float, end: float, span: float = 30.0, step: float = 0.1):
    """A flat action profile with one burst of activity in [start, end)."""
    times = np.arange(0.0, span, step)
    scores = np.where((times >= start) & (times < end), 1.0, 0.0)
    return times, scores


def test_the_end_moves_past_an_unfinished_move():
    """A scene shorter than the minimum length used to stop dead mid-action.

    The clip cannot be shortened here (earliest == the nominal end), so the
    only way out is forward, to where the burst finishes.
    """
    times, scores = _burst_profile(8.0, 12.0)
    end = snap_clip_end(
        10.0, earliest=10.0, latest=30.0,
        audio_times=times, audio_score=scores, search=3.0,
        w_audio=1.0, w_video=0.0,
    )
    # The profile is smoothed over a second before the lull is looked for, so
    # the envelope reaches its floor about half a window after the burst
    # actually stops. What matters is that the end is past it, not inside.
    assert 12.0 <= end <= 13.0


def test_the_end_pulls_back_out_of_the_next_move():
    """The budget ran a moment into whatever started next; drop that moment."""
    times, scores = _burst_profile(20.0, 26.0)
    end = snap_clip_end(
        21.0, earliest=10.0, latest=21.5,
        audio_times=times, audio_score=scores, search=3.0,
        w_audio=1.0, w_video=0.0,
    )
    assert 18.5 <= end <= 20.0


def test_an_end_that_is_already_quiet_does_not_move():
    times, scores = _burst_profile(5.0, 9.0)
    end = snap_clip_end(
        15.0, earliest=10.0, latest=25.0,
        audio_times=times, audio_score=scores, search=3.0,
        w_audio=1.0, w_video=0.0,
    )
    assert end == pytest.approx(15.0, abs=0.15)


def test_the_minimum_length_is_a_hard_floor():
    """Pulling back must never take the clip under the length that was asked for."""
    times, scores = _burst_profile(18.0, 26.0)
    end = snap_clip_end(
        20.0, earliest=19.5, latest=30.0,
        audio_times=times, audio_score=scores, search=3.0,
        w_audio=1.0, w_video=0.0,
    )
    assert end >= 19.5


def test_the_maximum_length_is_a_hard_ceiling():
    """Growing past the platform's limit is worse than an imperfect cut."""
    times, scores = _burst_profile(8.0, 20.0)
    end = snap_clip_end(
        10.0, earliest=10.0, latest=11.0,
        audio_times=times, audio_score=scores, search=5.0,
        w_audio=1.0, w_video=0.0,
    )
    assert end <= 11.0


def test_a_flat_profile_leaves_the_end_alone():
    times = np.arange(0.0, 30.0, 0.1)
    scores = np.zeros_like(times)
    end = snap_clip_end(
        15.0, earliest=10.0, latest=25.0,
        audio_times=times, audio_score=scores, search=3.0,
        w_audio=1.0, w_video=0.0,
    )
    assert end == pytest.approx(15.0)


def test_snapping_can_be_switched_off():
    times, scores = _burst_profile(8.0, 12.0)
    end = snap_clip_end(
        10.0, earliest=5.0, latest=25.0,
        audio_times=times, audio_score=scores, search=0.0,
        w_audio=1.0, w_video=0.0,
    )
    assert end == pytest.approx(10.0)


def test_motion_counts_towards_the_lull_as_well():
    """Silent but busy footage is not a good place to end either."""
    times = np.arange(0.0, 30.0, 0.1)
    quiet_audio = np.zeros_like(times)
    motion = np.where((times >= 8.0) & (times < 12.0), 1.0, 0.0)
    end = snap_clip_end(
        10.0, earliest=10.0, latest=30.0,
        audio_times=times, audio_score=quiet_audio,
        video_times=times, video_score=motion,
        search=3.0, w_audio=0.6, w_video=0.4,
    )
    assert 12.0 <= end <= 13.0


def test_no_action_profile_leaves_the_end_alone():
    assert snap_clip_end(
        10.0, earliest=5.0, latest=25.0,
        audio_times=np.array([]), audio_score=np.array([]), search=3.0,
    ) == pytest.approx(10.0)


def test_a_noisy_profile_still_finds_the_end_of_the_burst():
    """The reason the profile is smoothed at all.

    Audio action is sampled every 11ms and neighbouring samples routinely
    differ by a whole standard deviation. Searching that for the single
    quietest sample finds noise, not a lull, and hands back the frame the
    length budget had already chosen.
    """
    rng = np.random.default_rng(0)
    times = np.arange(0.0, 30.0, 0.011)
    burst = np.where((times >= 8.0) & (times < 12.0), 1.0, 0.0)
    noisy = burst + rng.normal(0.0, 0.8, size=times.shape)

    end = snap_clip_end(
        10.0, earliest=10.0, latest=30.0,
        audio_times=times, audio_score=noisy, search=3.0,
        w_audio=1.0, w_video=0.0,
    )
    assert 12.0 <= end <= 13.0

    unsmoothed = snap_clip_end(
        10.0, earliest=10.0, latest=30.0,
        audio_times=times, audio_score=noisy, search=3.0,
        smooth=0.0, w_audio=1.0, w_video=0.0,
    )
    assert unsmoothed < 12.0


# --- how long a clip should be --------------------------------------------
def _window_budget(scene_length, room, target, monkeypatch,
                   min_short=15, max_short=59):
    """Reproduce the budget arithmetic the clip loop does per scene."""
    monkeypatch.setenv("TARGET_SHORT_LENGTH", str(target))
    duration = int(scene_length)
    budget = min(target, max_short) if target > 0 else min(max_short, duration)
    effective_max = max(1, min(budget, int(room)))
    effective_min = min(min_short, effective_max)
    return effective_min, effective_max


def test_a_target_length_ignores_how_long_the_scene_is(monkeypatch):
    """The complaint this exists for: one clip at the ceiling, three short."""
    lengths = [
        _window_budget(scene, room=600, target=35, monkeypatch=monkeypatch)[1]
        for scene, room in ((15.5, 600), (87.1, 600), (33.0, 600), (37.1, 600))
    ]
    assert lengths == [35, 35, 35, 35]


def test_without_a_target_the_scene_still_decides(monkeypatch):
    """0 keeps the old behaviour, including the clip stuck at the maximum."""
    lengths = [
        _window_budget(scene, room=600, target=0, monkeypatch=monkeypatch)[1]
        for scene in (15.5, 87.1, 33.0, 37.1)
    ]
    assert lengths == [15, 59, 33, 37]


def test_the_target_never_beats_the_platform_limit(monkeypatch):
    assert _window_budget(200, room=600, target=120, monkeypatch=monkeypatch)[1] == 59


def test_the_end_of_the_video_still_wins(monkeypatch):
    """Asking for 35s eight seconds before the end has to give eight."""
    assert _window_budget(87.1, room=8, target=35, monkeypatch=monkeypatch)[1] == 8


def test_a_window_longer_than_its_scene_starts_at_the_scene(monkeypatch):
    """The overrun hangs off the end; the start stays where the action was found."""
    scene = make_scene(99.75, 115.22)
    window = 35.0
    overrun = max(0.0, window - (115.22 - 99.75))
    region = make_scene(99.75, 115.22 + overrun)
    times = np.arange(0.0, 200.0, 0.1)
    scores = np.zeros_like(times)

    start = shorts.best_action_window_start(region, window, times, scores, times, scores)
    assert start == pytest.approx(99.75, abs=0.5)


def test_clips_from_adjacent_scenes_do_not_overlap():
    """Windows may run past their scene, so neighbours can collide.

    Reproduces the measured case: with mean scoring the ranking put three
    scenes from the same stretch of video in one selection, and the first two
    clips shared 20.3 seconds of footage.
    """
    scenes = [(99.75, 115.22), (115.22, 144.13), (144.13, 160.22)]
    window = 35.0
    previous_end = 0.0
    spans = []
    for scene_start, scene_end in scenes:
        region_start = max(scene_start, previous_end)
        region_end = scene_end + max(0.0, window - (scene_end - scene_start))
        if region_start >= region_end:
            continue
        start = region_start
        spans.append((start, start + window))
        previous_end = start + window

    for (_, earlier_end), (later_start, _) in zip(spans, spans[1:]):
        assert later_start >= earlier_end
