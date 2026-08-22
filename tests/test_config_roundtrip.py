"""The dashboard writes .env; these guard the round-trip through that file.

A truthiness bug here once flipped every disabled toggle back to true on save -
including DEBUG_SKIP_RENDER, which makes the pipeline skip its actual work
while still reporting success. It was invisible in the UI and cost a full
debugging session; one assertion catches it.
"""
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "utils"))

import config as cfg


@pytest.fixture(autouse=True)
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(cfg, "ENV_PATH", path)
    return path


def _field(name):
    return next(f for f in cfg.iter_fields() if f.name == name)


def test_false_stays_false_through_a_save_cycle(env_file):
    """The string "false" is truthy in Python - the bug that started this."""
    field = _field("DEBUG_SKIP_RENDER")
    assert cfg.normalize_value(field, "false") == "false"
    assert cfg.normalize_value(field, False) == "false"
    assert cfg.normalize_value(field, "true") == "true"
    assert cfg.normalize_value(field, True) == "true"


def test_every_bool_survives_a_full_round_trip(env_file):
    bools = [f for f in cfg.iter_fields() if f.field_type == "bool"]
    assert bools, "schema has no boolean fields"

    values = {f.name: "false" for f in bools}
    cfg.save_env_values(values)
    reloaded, _ = cfg.load_env_values()

    for field in bools:
        assert reloaded[field.name] == "false", f"{field.name} flipped on save"
        assert cfg.coerce_value(field, reloaded[field.name]) is False


def test_values_with_hashes_are_not_truncated(env_file):
    cfg.save_env_values({"TTS_VOICE_DESCRIPTION": "warm voice # not a comment"})
    assert dotenv_values(str(env_file))["TTS_VOICE_DESCRIPTION"] == "warm voice # not a comment"


def test_save_leaves_no_temporary_file_behind(env_file):
    cfg.save_env_values({"SCENE_LIMIT": 4})
    assert env_file.exists()
    assert not list(env_file.parent.glob("*.saving"))


def test_unknown_settings_are_preserved(env_file):
    cfg.save_env_values({"SCENE_LIMIT": 4}, extras={"MY_OWN_FLAG": "1"})
    assert dotenv_values(str(env_file))["MY_OWN_FLAG"] == "1"


def test_schema_field_names_are_unique():
    names = [f.name for f in cfg.iter_fields()]
    assert len(names) == len(set(names))
