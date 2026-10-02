"""What Whisper writes that nobody said."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import subtitle_generator as sg


def _result(*texts):
    return {
        "text": " ".join(texts),
        "segments": [
            {"text": t, "start": float(i), "end": float(i) + 1.0, "words": [{"word": t, "start": float(i), "end": float(i) + 1.0}]}
            for i, t in enumerate(texts)
        ],
    }


@pytest.mark.parametrize("credit", [
    "Untertitel im Auftrag des ZDF für funk, 2017",   # measured on Haku Challenge #16
    "Untertitel der Amara.org-Community",
    "Untertitelung des ZDF, 2020",
    "Copyright WDR 2021",
    "Subtitles by the Amara.org community",
])
def test_made_up_credits_are_dropped(credit, monkeypatch):
    monkeypatch.delenv("WHISPER_DROP_STOCK_PHRASES", raising=False)
    cleaned = sg._drop_stock_phrases(_result(credit, "Ja ja ja ja ja ist das geil"))
    assert [s["text"] for s in cleaned["segments"]] == ["Ja ja ja ja ja ist das geil"]
    assert cleaned["text"] == "Ja ja ja ja ja ist das geil"


@pytest.mark.parametrize("speech", [
    "Die Untertitel sind viel zu klein",
    "Das war 2017 schon so schwer",
    "Jetzt aber, Jungs, auf geht's",
])
def test_ordinary_speech_is_kept(speech, monkeypatch):
    monkeypatch.delenv("WHISPER_DROP_STOCK_PHRASES", raising=False)
    result = _result(speech)
    assert sg._drop_stock_phrases(result) == result


def test_the_filter_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("WHISPER_DROP_STOCK_PHRASES", "false")
    result = _result("Untertitel im Auftrag des ZDF für funk, 2017")
    assert sg._drop_stock_phrases(result) == result
