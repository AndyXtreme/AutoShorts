"""Every module must at least import.

A helper once got inserted between a @dataclass decorator and its class, which
turns the decorator into a function call on a function. Nothing noticed,
because the module is only imported when TTS is enabled - and the failure looks
like an unrelated AttributeError deep in dataclasses.
"""
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

MODULES = sorted(p.stem for p in SRC.glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
