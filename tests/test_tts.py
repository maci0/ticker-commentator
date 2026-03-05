"""Tests for commentator.tts pure functions.

Tests _format_prompt, _speed_to_generation, _turn_token_into_id,
_iter_custom_tokens_from_text_stream, and pcm_chunks_to_wav without
requiring GPU or model downloads.
"""

import struct
import wave
import io

from commentator.tts import (
    _format_prompt,
    _speed_to_generation,
    _turn_token_into_id,
    _iter_custom_tokens_from_text_stream,
    pcm_chunks_to_wav,
)


# ── _format_prompt ──────────────────────────────────────────────────


def test_format_prompt_structure() -> None:
    result = _format_prompt("Hello world", "leo")
    assert result == "<|audio|>leo: Hello world<|eot_id|><custom_token_4>"


def test_format_prompt_empty_text() -> None:
    result = _format_prompt("", "zac")
    assert result == "<|audio|>zac: <|eot_id|><custom_token_4>"


# ── _speed_to_generation ────────────────────────────────────────────


def test_speed_at_lower_bound() -> None:
    """Speed 0.8 should produce baseline temperature."""
    result = _speed_to_generation(0.8)
    assert abs(result["temperature"] - 0.6) < 0.01
    assert abs(result["repeat_penalty"] - 1.1) < 0.01


def test_speed_at_upper_bound() -> None:
    """Speed 1.4 should produce max temperature adjustment."""
    result = _speed_to_generation(1.4)
    assert abs(result["temperature"] - 1.0) < 0.01
    assert abs(result["repeat_penalty"] - 1.3) < 0.01


def test_speed_below_range_clamped() -> None:
    """Speed below 0.8 should clamp to baseline."""
    result = _speed_to_generation(0.0)
    assert abs(result["temperature"] - 0.6) < 0.01


def test_speed_above_range_clamped() -> None:
    """Speed above 1.4 should clamp to max."""
    result = _speed_to_generation(5.0)
    assert abs(result["temperature"] - 1.0) < 0.01


def test_speed_midpoint() -> None:
    """Speed at midpoint (1.1) should produce interpolated values."""
    result = _speed_to_generation(1.1)
    assert 0.6 < result["temperature"] < 1.0


# ── _turn_token_into_id ────────────────────────────────────────────


def test_token_id_at_index_zero() -> None:
    # index=0 → offset = 0*4096 = 0 → result = raw - 10
    assert _turn_token_into_id(110, 0) == 100


def test_token_id_at_index_one() -> None:
    # index=1 → offset = 1*4096 = 4096 → result = raw - 10 - 4096
    assert _turn_token_into_id(4200, 1) == 94


def test_token_id_wraps_at_seven() -> None:
    # index=7 → 7 % 7 = 0 → same as index 0
    assert _turn_token_into_id(110, 7) == _turn_token_into_id(110, 0)


def test_token_id_negative_result() -> None:
    """Small raw_token should produce a negative result (filtered by caller)."""
    result = _turn_token_into_id(5, 0)
    assert result < 0


# ── _iter_custom_tokens_from_text_stream ────────────────────────────


def test_single_token_extraction() -> None:
    stream = iter(["<custom_token_42>"])
    tokens = list(_iter_custom_tokens_from_text_stream(stream))
    assert tokens == [42]


def test_multiple_tokens_in_one_chunk() -> None:
    stream = iter(["<custom_token_10><custom_token_20>"])
    tokens = list(_iter_custom_tokens_from_text_stream(stream))
    assert tokens == [10, 20]


def test_tokens_split_across_chunks() -> None:
    stream = iter(["<custom_tok", "en_99>"])
    tokens = list(_iter_custom_tokens_from_text_stream(stream))
    assert tokens == [99]


def test_no_tokens_in_stream() -> None:
    stream = iter(["hello world", "no tokens here"])
    tokens = list(_iter_custom_tokens_from_text_stream(stream))
    assert tokens == []


def test_mixed_text_and_tokens() -> None:
    stream = iter(["text before <custom_token_5> middle <custom_token_6> end"])
    tokens = list(_iter_custom_tokens_from_text_stream(stream))
    assert tokens == [5, 6]


def test_large_buffer_trimmed() -> None:
    """Buffer should be trimmed to prevent unbounded growth."""
    # Send a large chunk with no tokens, followed by a token.
    large = "x" * 500
    stream = iter([large, "<custom_token_7>"])
    tokens = list(_iter_custom_tokens_from_text_stream(stream))
    assert tokens == [7]


# ── pcm_chunks_to_wav ───────────────────────────────────────────────


def test_wav_output_valid() -> None:
    """Output should be a valid WAV file with correct parameters."""
    # 100 samples of silence (16-bit mono)
    pcm = b"\x00\x00" * 100
    result = pcm_chunks_to_wav([pcm], sample_rate=24000)

    # Parse the WAV header
    buf = io.BytesIO(result)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 100


def test_wav_multiple_chunks() -> None:
    """Multiple PCM chunks should be concatenated correctly."""
    chunk1 = b"\x00\x00" * 50
    chunk2 = b"\x00\x00" * 75
    result = pcm_chunks_to_wav([chunk1, chunk2], sample_rate=16000)

    buf = io.BytesIO(result)
    with wave.open(buf, "rb") as wf:
        assert wf.getnframes() == 125
        assert wf.getframerate() == 16000


def test_wav_empty_chunks() -> None:
    """Empty chunk list should produce a valid zero-length WAV."""
    result = pcm_chunks_to_wav([], sample_rate=24000)

    buf = io.BytesIO(result)
    with wave.open(buf, "rb") as wf:
        assert wf.getnframes() == 0
