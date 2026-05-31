"""Tests for commentator.tts pure functions.

Tests _format_prompt, _speed_to_generation, _turn_token_into_id,
_iter_custom_tokens_from_text_stream, _decode_frames_to_pcm,
pcm_chunks_to_wav, iter_audio_chunks (voice validation), and VALID_VOICES
without requiring GPU or model downloads.
"""

import io
import wave
from unittest.mock import patch

import pytest

from commentator.tts import (
    _TEMPERATURE,
    VALID_VOICES,
    _decode_frames_to_pcm,
    _format_prompt,
    _iter_custom_tokens_from_text_stream,
    _speed_to_generation,
    _turn_token_into_id,
    iter_audio_chunks,
    numbers_to_speech,
    pcm_chunks_to_wav,
    text_to_speech,
)

# ── numbers_to_speech ────────────────────────────────────────────────


def test_numbers_to_speech_price_with_cents() -> None:
    out = numbers_to_speech("Apple at $312.07 now")
    assert "three hundred" in out and "twelve" in out and "seven cents" in out
    assert "$" not in out and "312" not in out


def test_numbers_to_speech_whole_dollars() -> None:
    out = numbers_to_speech("crossed $50 today")
    assert "fifty dollars" in out and "$" not in out


def test_numbers_to_speech_thousands() -> None:
    assert "one thousand" in numbers_to_speech("$1000.50").lower()


def test_numbers_to_speech_percent() -> None:
    out = numbers_to_speech("up +1.2% on the day")
    assert "percent" in out and "%" not in out and "1.2" not in out


def test_numbers_to_speech_negative_percent() -> None:
    assert "negative three percent" in numbers_to_speech("down -3% hard")


def test_numbers_to_speech_no_digits_remain() -> None:
    out = numbers_to_speech("RSI 68, $312.07, +1.2%, range $310.50-$313.11")
    assert not any(ch.isdigit() for ch in out)


def test_numbers_to_speech_leaves_plain_text() -> None:
    assert numbers_to_speech("bulls smashing resistance") == "bulls smashing resistance"

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
    assert abs(result["temperature"] - _TEMPERATURE) < 0.01
    assert abs(result["repeat_penalty"] - 1.1) < 0.01


def test_speed_at_upper_bound() -> None:
    """Speed 1.4 should produce max temperature adjustment (+0.4 above baseline)."""
    result = _speed_to_generation(1.4)
    assert abs(result["temperature"] - (_TEMPERATURE + 0.4)) < 0.01
    assert abs(result["repeat_penalty"] - 1.3) < 0.01


def test_speed_below_range_clamped() -> None:
    """Speed below 0.8 should clamp to baseline and emit a warning."""
    result = _speed_to_generation(0.0)
    assert abs(result["temperature"] - _TEMPERATURE) < 0.01


def test_speed_above_range_clamped() -> None:
    """Speed above 1.4 should clamp to max (+0.4 above baseline) and emit a warning."""
    result = _speed_to_generation(5.0)
    assert abs(result["temperature"] - (_TEMPERATURE + 0.4)) < 0.01


def test_speed_out_of_range_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Out-of-range speed values should log a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="commentator.tts"):
        _speed_to_generation(0.0)
    assert any("outside [0.8, 1.4]" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="commentator.tts"):
        _speed_to_generation(2.0)
    assert any("outside [0.8, 1.4]" in r.message for r in caplog.records)


def test_speed_in_range_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """In-range speed values should not log a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="commentator.tts"):
        _speed_to_generation(1.0)
    assert not any("outside [0.8, 1.4]" in r.message for r in caplog.records)


def test_speed_midpoint() -> None:
    """Speed at midpoint (1.1) should produce the exact interpolated values."""
    speed_norm = (1.1 - 0.8) / 0.6  # 0.5
    result = _speed_to_generation(1.1)
    assert result["temperature"] == pytest.approx(_TEMPERATURE + 0.4 * speed_norm, abs=0.001)
    assert result["repeat_penalty"] == pytest.approx(1.1 + 0.2 * speed_norm, abs=0.001)


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


# ── _decode_frames_to_pcm (early returns, no SNAC required) ─────────


def test_decode_frames_insufficient_frames_returns_none() -> None:
    """Fewer than 4 complete frames (28 tokens) should return None without loading SNAC."""
    # 3 frames = 21 tokens, below the 4-frame minimum
    assert _decode_frames_to_pcm(list(range(21))) is None


def test_decode_frames_empty_returns_none() -> None:
    """Zero tokens (0 frames) should return None."""
    assert _decode_frames_to_pcm([]) is None


def test_decode_frames_out_of_range_high_returns_none() -> None:
    """Any token >= 4096 in any codebook position should return None before loading SNAC."""
    # 4 frames (28 tokens), first token is out of range
    tokens = [0] * 28
    tokens[0] = 4096
    assert _decode_frames_to_pcm(tokens) is None


def test_decode_frames_negative_token_returns_none() -> None:
    """Any negative token should return None before loading SNAC."""
    tokens = [0] * 28
    tokens[0] = -1
    assert _decode_frames_to_pcm(tokens) is None


def test_decode_frames_out_of_range_in_codes1_returns_none() -> None:
    """An out-of-range token at position 1 (codes_1 column) should return None."""
    # 4 frames = 28 tokens; position 1 maps to arr[0, 1] → codes_1
    tokens = [0] * 28
    tokens[1] = 4096
    assert _decode_frames_to_pcm(tokens) is None


def test_decode_frames_out_of_range_in_codes2_returns_none() -> None:
    """An out-of-range token at position 2 (codes_2 column) should return None."""
    # 4 frames = 28 tokens; position 2 maps to arr[0, 2] → codes_2
    tokens = [0] * 28
    tokens[2] = 4096
    assert _decode_frames_to_pcm(tokens) is None


def test_decode_frames_at_27_tokens_returns_none() -> None:
    """27 tokens = 3 complete frames, one below the 4-frame minimum; must return None."""
    assert _decode_frames_to_pcm(list(range(27))) is None


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


# ── VALID_VOICES and voice validation ───────────────────────────────


def test_valid_voices_non_empty() -> None:
    assert len(VALID_VOICES) > 0


def test_valid_voices_contains_defaults() -> None:
    """The two default voice values used in the codebase must be in VALID_VOICES."""
    assert "zac" in VALID_VOICES
    assert "leo" in VALID_VOICES


def test_iter_audio_chunks_invalid_voice_raises() -> None:
    with pytest.raises(ValueError, match="Unknown voice"):
        iter_audio_chunks("hello", voice="not_a_real_voice")


# ── text_to_speech ───────────────────────────────────────────────────


def test_text_to_speech_invalid_voice_raises() -> None:
    """text_to_speech should propagate ValueError for an unknown voice."""
    with pytest.raises(ValueError, match="Unknown voice"):
        text_to_speech("hello", voice="not_a_real_voice")


def test_text_to_speech_returns_none_when_no_audio() -> None:
    """text_to_speech should return None when iter_audio_chunks yields no chunks."""
    with patch("commentator.tts.iter_audio_chunks", return_value=iter([])):
        result = text_to_speech("hello", voice="zac")
    assert result is None


def test_text_to_speech_returns_wav_bytes() -> None:
    """text_to_speech should return valid WAV bytes when chunks are produced."""
    pcm = b"\x00\x00" * 100
    with patch("commentator.tts.iter_audio_chunks", return_value=iter([pcm])):
        result = text_to_speech("hello", voice="zac")
    assert result is not None
    assert result[:4] == b"RIFF"  # WAV file magic bytes
