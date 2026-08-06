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


def test_missing_qwen_gives_install_hint() -> None:
    # qwen-tts is an in-project extra (uv sync --extra qwen); only the
    # not-installed path raises the hint, so skip when it is present.
    try:
        import qwen_tts  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="qwen-tts"):
            list(tts_engines.iter_audio_chunks("qwen", "hello"))
    else:
        pytest.skip("qwen-tts installed; missing-package hint path not exercised")


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


# ── mocked adapter happy paths ────────────────────────────────────────


def test_chatterbox_chunks_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCB:
        def generate(self, text: str, exaggeration: float, cfg_weight: float):
            assert text == "hello bulls"
            assert exaggeration == 0.8
            assert cfg_weight == 0.5
            return np.array([0.0, 0.5], dtype=np.float32)

    monkeypatch.setattr(tts_engines, "_CHATTERBOX_MODEL", FakeCB())
    chunks = list(tts_engines.iter_audio_chunks("chatterbox", "hello bulls"))
    assert len(chunks) == 1
    assert len(chunks[0]) == 4  # 2 int16 samples


def test_kokoro_chunks_with_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pipe(text: str, voice: str):
        assert text == "hi"
        assert voice == "am_michael"
        yield ("gs", "ps", np.array([1.0, -1.0], dtype=np.float32))

    monkeypatch.setattr(tts_engines, "_KOKORO_PIPELINE", fake_pipe)
    chunks = list(tts_engines.iter_audio_chunks("kokoro", "hi"))
    assert len(chunks) == 1
    out = np.frombuffer(chunks[0], dtype=np.int16).tolist()
    assert out == [32767, -32767]


def test_qwen_chunks_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQwen:
        def generate_custom_voice(self, text, speaker, language, instruct):
            assert text == "yo"
            assert speaker == "ryan"
            assert language == "english"
            assert instruct is None
            return [np.array([0.0], dtype=np.float32)], 24000

    monkeypatch.setattr(tts_engines, "_QWEN_MODEL", FakeQwen())
    chunks = list(tts_engines.iter_audio_chunks("qwen", "yo"))
    assert chunks == [b"\x00\x00"]


def test_get_chatterbox_caches_and_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = object()
    calls = {"n": 0}

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = _Cuda()

    class FakeChatterboxTTS:
        @staticmethod
        def from_pretrained(device: str):
            calls["n"] += 1
            assert device == "cpu"
            return fake_model

    import sys
    import types

    chatterbox_mod = types.ModuleType("chatterbox")
    tts_mod = types.ModuleType("chatterbox.tts")
    tts_mod.ChatterboxTTS = FakeChatterboxTTS
    monkeypatch.setitem(sys.modules, "chatterbox", chatterbox_mod)
    monkeypatch.setitem(sys.modules, "chatterbox.tts", tts_mod)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(tts_engines, "_CHATTERBOX_MODEL", None)

    m1 = tts_engines._get_chatterbox()
    m2 = tts_engines._get_chatterbox()
    assert m1 is fake_model and m2 is fake_model
    assert calls["n"] == 1


def test_get_kokoro_caches_and_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pipe = object()
    calls = {"n": 0}

    class FakeKPipeline:
        def __init__(self, lang_code: str) -> None:
            calls["n"] += 1
            assert lang_code == "a"
            self.obj = fake_pipe

    import sys
    import types

    kokoro_mod = types.ModuleType("kokoro")
    kokoro_mod.KPipeline = FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_mod)
    monkeypatch.setattr(tts_engines, "_KOKORO_PIPELINE", None)

    p1 = tts_engines._get_kokoro()
    p2 = tts_engines._get_kokoro()
    assert p1 is p2
    assert calls["n"] == 1


def test_get_qwen_caches_and_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = object()
    calls = {"n": 0}

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        bfloat16 = "bf16"
        float32 = "f32"
        cuda = _Cuda()

    class FakeQwen3TTSModel:
        @staticmethod
        def from_pretrained(repo, device_map, dtype):
            calls["n"] += 1
            assert device_map == "cpu"
            assert dtype == "f32"
            return fake_model

    import sys
    import types

    qwen_mod = types.ModuleType("qwen_tts")
    qwen_mod.Qwen3TTSModel = FakeQwen3TTSModel
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_mod)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(tts_engines, "_QWEN_MODEL", None)

    m1 = tts_engines._get_qwen()
    m2 = tts_engines._get_qwen()
    assert m1 is fake_model and m2 is fake_model
    assert calls["n"] == 1
