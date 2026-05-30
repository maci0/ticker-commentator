"""Tests for the switchable TTS engine dispatch and alt-engine adapters.

These run without any alt-engine package installed, so they verify the dispatch
wiring, PCM conversion, and the clear error raised when an engine's library is
missing — not actual chatterbox/kokoro synthesis.
"""


import numpy as np
import pytest

from commentator import tts_engines

# ── _float_to_pcm16 ──────────────────────────────────────────────────


def test_float_to_pcm16_numpy_roundtrip() -> None:
    audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    pcm = tts_engines._float_to_pcm16(audio)
    out = np.frombuffer(pcm, dtype=np.int16)
    assert out.tolist() == [0, 32767, -32767, 16383]


def test_float_to_pcm16_clips_out_of_range() -> None:
    audio = np.array([2.0, -2.0], dtype=np.float32)
    out = np.frombuffer(tts_engines._float_to_pcm16(audio), dtype=np.int16)
    assert out.tolist() == [32767, -32767]


def test_float_to_pcm16_accepts_tensor_like() -> None:
    """A torch-like object (has .detach().cpu().numpy()) is handled."""

    class FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    pcm = tts_engines._float_to_pcm16(FakeTensor(np.array([0.0, 1.0], dtype=np.float32)))
    assert np.frombuffer(pcm, dtype=np.int16).tolist() == [0, 32767]


def test_float_to_pcm16_flattens_2d() -> None:
    audio = np.zeros((1, 5), dtype=np.float32)
    pcm = tts_engines._float_to_pcm16(audio)
    assert len(pcm) == 5 * 2  # 5 int16 samples


# ── dispatch ─────────────────────────────────────────────────────────


def test_unknown_engine_raises() -> None:
    with pytest.raises(ValueError, match="Unknown TTS_ENGINE"):
        list(tts_engines.iter_audio_chunks("bogus", "hello"))


def test_missing_chatterbox_gives_install_hint() -> None:
    """chatterbox-tts is not installed in the project venv → clear RuntimeError."""
    with pytest.raises(RuntimeError, match="chatterbox-tts"):
        list(tts_engines.iter_audio_chunks("chatterbox", "hello"))


def test_missing_kokoro_gives_install_hint() -> None:
    with pytest.raises(RuntimeError, match="kokoro"):
        list(tts_engines.iter_audio_chunks("kokoro", "hello"))


def test_sample_rate_matches_orpheus() -> None:
    from commentator.tts import SAMPLE_RATE

    assert tts_engines.SAMPLE_RATE == SAMPLE_RATE


# ── tts.iter_audio_chunks routing ────────────────────────────────────


def test_default_engine_is_orpheus(monkeypatch: pytest.MonkeyPatch) -> None:
    """With TTS_ENGINE unset/orpheus, a bad voice still raises the Orpheus
    ValueError (i.e. it did not route away to an alt engine)."""
    import commentator.tts as tts

    monkeypatch.setattr(tts, "TTS_ENGINE", "orpheus")
    with pytest.raises(ValueError, match="Unknown voice"):
        tts.iter_audio_chunks("hello", voice="not_a_voice")


def test_non_default_engine_routes_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-orpheus TTS_ENGINE routes through tts_engines (here surfacing the
    missing-package RuntimeError rather than the Orpheus voice check)."""
    import commentator.tts as tts

    monkeypatch.setattr(tts, "TTS_ENGINE", "chatterbox")
    with pytest.raises(RuntimeError, match="chatterbox-tts"):
        list(tts.iter_audio_chunks("hello", voice="not_a_voice"))
