"""Tests for commentator.playback pure helpers."""

import base64
from unittest.mock import patch

import pytest

from commentator.playback import data_uri_audio_html, streaming_audio_html, synthesize_wav


def test_synthesize_wav_empty() -> None:
    with patch("commentator.playback.iter_audio_chunks", return_value=iter([])):
        assert synthesize_wav("hi", "zac", 1.0) == (None, None)


def test_synthesize_wav_returns_wav_and_duration() -> None:
    # 24000 samples of silence @ 24kHz = 1.0s PCM → duration max(1.0,1.0)+0.75
    pcm = b"\x00\x00" * 24000
    with (
        patch("commentator.playback.iter_audio_chunks", return_value=iter([pcm])),
        patch("commentator.playback.SAMPLE_RATE", 24000),
    ):
        wav, dur = synthesize_wav("hi", "zac", 1.0)
    assert wav is not None and wav[:4] == b"RIFF"
    assert dur == pytest.approx(1.75)


def test_data_uri_audio_html_autoplay_includes_id() -> None:
    audio = b"RIFF" + b"\x00" * 40
    html = data_uri_audio_html(audio, "AAPL", autoplay=True, uid="a123")
    assert 'id="a123"' in html
    assert "autoplay" in html
    assert "data:audio/wav;base64," in html
    assert base64.b64encode(audio).decode() in html
    assert "AAPL" in html


def test_data_uri_audio_html_no_autoplay_omits_id() -> None:
    html = data_uri_audio_html(b"RIFF", "MSFT", autoplay=False)
    assert "autoplay" not in html
    assert "id=" not in html


def test_streaming_audio_html_points_at_local_server() -> None:
    html = streaming_audio_html("sid42", 8765, "TSLA", uid="a9")
    assert 'id="a9"' in html
    assert "http://127.0.0.1:8765/audio/sid42.wav" in html
    assert "autoplay" in html
    assert "TSLA" in html
