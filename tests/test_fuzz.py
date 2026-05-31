"""Property-based / fuzz tests (hypothesis) for the pure functions.

Per the project testing preference, functions get a fuzz harness alongside their
unit tests. These focus on never-crash + invariant guarantees under arbitrary
(including hostile) input — especially `_parse_range`, which parses untrusted
HTTP Range headers.
"""

import math
import os

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from commentator import audio_server, tts_engines
from commentator._config import env_float, env_int, parse_tensor_split
from commentator.analysis import analyze_stock
from commentator.commentary import (
    _COMMON_RULES,
    _EMOTION_TAG_RE,
    _inject_emotion_tags,
    _system_prompt,
)
from commentator.data import _TICKER_RE, _validate_ticker
from commentator.tts import (
    _iter_custom_tokens_from_text_stream,
    _speed_to_generation,
    _turn_token_into_id,
    pcm_chunks_to_wav,
)

# ── audio_server._parse_range (untrusted HTTP header) ────────────────


@given(header=st.text(), total=st.integers(min_value=1, max_value=10**9))
def test_parse_range_never_crashes(header: str, total: int) -> None:
    out = audio_server._parse_range(header, total)
    assert out is None or (
        isinstance(out, tuple)
        and len(out) == 2
        and all(isinstance(x, int) for x in out)
    )


@given(header=st.one_of(st.none(), st.just("")))
def test_parse_range_empty_is_none(header: "str | None") -> None:
    assert audio_server._parse_range(header, 1000) is None


@given(
    start=st.integers(min_value=0, max_value=10**6),
    length=st.integers(min_value=0, max_value=10**6),
    total=st.integers(min_value=1, max_value=10**6),
)
def test_parse_range_explicit_bytes(start: int, length: int, total: int) -> None:
    end = start + length
    out = audio_server._parse_range(f"bytes={start}-{end}", total)
    assert out == (start, end)


@given(total=st.integers(min_value=1, max_value=10**6))
def test_parse_range_open_ended_fills_to_end(total: int) -> None:
    out = audio_server._parse_range("bytes=0-", total)
    assert out == (0, total - 1)


# ── audio_server WAV headers ─────────────────────────────────────────


@given(sr=st.integers(min_value=1, max_value=384000))
def test_streaming_header_is_44_bytes_riff(sr: int) -> None:
    hdr = audio_server._streaming_wav_header(sr)
    assert len(hdr) == 44 and hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE"


@given(
    sr=st.integers(min_value=1, max_value=384000),
    pcm=st.binary(max_size=4096),
)
def test_complete_wav_roundtrip_length(sr: int, pcm: bytes) -> None:
    wav = audio_server._complete_wav(sr, pcm)
    assert wav[:4] == b"RIFF"
    assert wav[44:] == pcm
    assert len(wav) == 44 + len(pcm)


# ── tts_engines._float_to_pcm16 ──────────────────────────────────────


@given(
    audio=st.lists(
        st.floats(allow_nan=True, allow_infinity=True, width=32), max_size=512
    )
)
def test_float_to_pcm16_handles_any_floats(audio: list) -> None:
    arr = np.array(audio, dtype=np.float32)
    out = tts_engines._float_to_pcm16(arr)
    assert isinstance(out, bytes)
    assert len(out) == 2 * len(audio)  # one int16 per sample
    # No NaN/garbage: every value is a valid in-range int16.
    samples = np.frombuffer(out, dtype=np.int16)
    assert np.all(samples >= -32767) and np.all(samples <= 32767)


# ── tts._turn_token_into_id ──────────────────────────────────────────


@given(raw=st.integers(min_value=0, max_value=200000), index=st.integers(0, 10**6))
def test_turn_token_into_id_formula(raw: int, index: int) -> None:
    assert _turn_token_into_id(raw, index) == raw - 10 - ((index % 7) * 4096)


# ── tts._speed_to_generation (bounded UI value) ──────────────────────


@given(speed=st.floats(min_value=-100, max_value=100, allow_nan=False))
def test_speed_to_generation_bounds(speed: float) -> None:
    from commentator.tts import _TEMPERATURE, _TOP_P

    o = _speed_to_generation(speed)
    assert _TEMPERATURE - 1e-6 <= o["temperature"] <= _TEMPERATURE + 0.4 + 1e-6
    assert 1.1 - 1e-6 <= o["repeat_penalty"] <= 1.3 + 1e-6
    assert o["top_p"] == _TOP_P


# ── tts._iter_custom_tokens_from_text_stream ─────────────────────────


@given(chunks=st.lists(st.text(), max_size=20))
def test_custom_token_stream_never_crashes(chunks: list) -> None:
    out = list(_iter_custom_tokens_from_text_stream(iter(chunks)))
    assert all(isinstance(t, int) for t in out)


@given(n=st.integers(min_value=0, max_value=99999))
def test_custom_token_stream_extracts_value(n: int) -> None:
    out = list(_iter_custom_tokens_from_text_stream(iter([f"<custom_token_{n}>"])))
    assert out == [n]


# ── tts.pcm_chunks_to_wav ────────────────────────────────────────────


@given(chunks=st.lists(st.binary(max_size=64).map(lambda b: b[: len(b) // 2 * 2])))
def test_pcm_chunks_to_wav_frame_count(chunks: list) -> None:
    import io
    import wave

    wav = pcm_chunks_to_wav(chunks)
    total = sum(len(c) for c in chunks)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnframes() == total // 2  # 16-bit mono => 2 bytes/frame


# ── data._validate_ticker (untrusted user input) ────────────────────


@given(ticker=st.text(max_size=60))
def test_validate_ticker_returns_valid_or_raises(ticker: str) -> None:
    """Any input either normalizes to a regex-valid ticker or raises ValueError —
    never any other exception (which would mean an unhandled crash path)."""
    try:
        out = _validate_ticker(ticker)
    except ValueError:
        return
    assert _TICKER_RE.match(out) and out == out.strip().upper()


# ── commentary._inject_emotion_tags (arbitrary LLM text) ─────────────


_sentiment = st.fixed_dictionaries(
    {
        "trend": st.sampled_from(["bullish", "bearish", "sideways", "unknown"]),
        "price_change_pct": st.floats(min_value=-50, max_value=50, allow_nan=False),
        "volatility": st.sampled_from(["high", "normal", "low", "unknown"]),
    }
)


@given(text=st.text(max_size=120), analysis=_sentiment)
def test_inject_emotion_tags_never_crashes_and_only_known_tags(
    text: str, analysis: dict
) -> None:
    out = _inject_emotion_tags(text, analysis)
    assert isinstance(out, str)
    # Any emotion tag in the output must come from the pruned pool the tuning
    # restricted us to (arbitrary <...> in the fuzzed input is not our concern).
    for tag in _EMOTION_TAG_RE.findall(out):
        assert f"<{tag}>" in {"<laugh>", "<chuckle>", "<sigh>"}


# ── commentary._system_prompt (personality selection) ───────────────


@given(personality=st.text(max_size=40))
def test_system_prompt_always_valid(personality: str) -> None:
    """Any personality string yields a non-empty prompt containing the shared
    rules (unknown ones fall back to a valid persona)."""
    out = _system_prompt(personality)
    assert isinstance(out, str) and _COMMON_RULES in out


# ── analysis.analyze_stock (OHLCV math, NaN/edge guards) ─────────────

# Bounded prices/volumes plus explicit NaN (which analyze_stock must guard).
_price = st.one_of(
    st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
)
_vol = st.one_of(
    st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
)
_ohlcv_rows = st.lists(st.tuples(_price, _price, _price, _price, _vol), max_size=60)

_TRENDS = {"bullish", "bearish", "sideways"}
_VOLS = {"high", "medium", "low", "unknown"}
_VOLT = {"heavy", "light", "normal"}
_CROSS = {"golden_cross", "death_cross", None}


@settings(max_examples=250)
@given(rows=_ohlcv_rows)
def test_analyze_stock_never_crashes_finite_output(rows: list) -> None:
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    if len(df):
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="min")
    res = analyze_stock(df)
    assert isinstance(res, dict)
    if "error" in res:
        return
    # Numeric fields must be finite — the NaN guards must hold for any input.
    for k in ("price_change_pct", "current_price", "open_price", "high", "low"):
        assert math.isfinite(res[k]), (k, res[k])
    assert res["trend"] in _TRENDS
    assert res["volatility"] in _VOLS
    assert res["volume_trend"] in _VOLT
    assert res["sma_cross"] in _CROSS
    assert res["rsi"] is None or (0.0 <= res["rsi"] <= 100.0)


# ── _config helpers ──────────────────────────────────────────────────

# Env values can't contain null bytes or surrogates (OS-enforced), so fuzz
# within the valid env-string domain.
_env_text = st.text(
    alphabet=st.characters(min_codepoint=1, blacklist_categories=("Cs",)), max_size=40
)


@given(val=_env_text, default=st.integers(-1000, 1000))
def test_env_int_never_crashes(val: str, default: int) -> None:
    os.environ["FUZZ_INT"] = val
    try:
        assert isinstance(env_int("FUZZ_INT", default), int)
    finally:
        os.environ.pop("FUZZ_INT", None)


@given(val=_env_text, default=st.floats(allow_nan=False, allow_infinity=False))
def test_env_float_never_crashes(val: str, default: float) -> None:
    os.environ["FUZZ_FLOAT"] = val
    try:
        assert isinstance(env_float("FUZZ_FLOAT", default), float)
    finally:
        os.environ.pop("FUZZ_FLOAT", None)


@settings(max_examples=200)
@given(val=_env_text)
def test_parse_tensor_split_never_crashes(val: str) -> None:
    os.environ["FUZZ_TS"] = val
    try:
        out = parse_tensor_split("FUZZ_TS")
    finally:
        os.environ.pop("FUZZ_TS", None)
    assert isinstance(out, list) and all(isinstance(x, float) for x in out)
