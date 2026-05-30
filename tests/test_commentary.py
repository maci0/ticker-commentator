"""Tests for commentator.commentary — emotion tag injection and generate_commentary.

These tests exercise all sentiment paths and edge cases of
_inject_emotion_tags without requiring the LLM.
"""

import re
from unittest.mock import patch

from commentator.analysis import AnalysisResult
from commentator.commentary import (
    _generate_with_llama_cpp,
    _inject_emotion_tags,
    generate_commentary,
)

_TAG_RE = re.compile(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>")


def _count_tags(text: str) -> int:
    return len(_TAG_RE.findall(text))


def _get_tags(text: str) -> list[str]:
    return _TAG_RE.findall(text)


# ── Empty / no-op ───────────────────────────────────────────────────


def test_empty_text_returns_empty() -> None:
    assert _inject_emotion_tags("", {"trend": "bullish"}) == ""


def test_whitespace_only_returns_empty() -> None:
    assert _inject_emotion_tags("   ", {"trend": "bullish"}) == ""


# ── Tag injection by sentiment ──────────────────────────────────────


def test_bullish_uses_positive_tags() -> None:
    """Force both random checks to pass to guarantee 2 tags."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0  # Always below threshold
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Bulls charging ahead!",
            {"trend": "bullish", "price_change_pct": 1.5, "volatility": "low"},
        )
    tags = _get_tags(result)
    assert len(tags) == 2  # both injections fire when random=0.0
    assert all(t in ("laugh", "chuckle") for t in tags)


def test_bearish_uses_negative_tags() -> None:
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Bears dragging it down!",
            {"trend": "bearish", "price_change_pct": -2.0, "volatility": "low"},
        )
    tags = _get_tags(result)
    assert len(tags) == 2  # both injections fire when random=0.0
    assert all(t in ("sigh", "groan") for t in tags)


def test_sideways_uses_neutral_tags() -> None:
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Market holding steady.",
            {"trend": "sideways", "price_change_pct": 0.3, "volatility": "low"},
        )
    tags = _get_tags(result)
    assert len(tags) == 2  # both injections fire when random=0.0
    assert all(t in ("chuckle", "sniffle", "yawn") for t in tags)


def test_tag_prepended_when_no_punctuation() -> None:
    """When text has no punctuation pause, the first tag is prepended to the front."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Market holding steady",  # No punctuation
            {"trend": "sideways", "price_change_pct": 0.3, "volatility": "low"},
        )
    tags = _get_tags(result)
    assert len(tags) == 2  # both injections fire when random=0.0
    assert result.startswith("<")


def test_high_volatility_uses_drama_pool() -> None:
    """High volatility should use the high-drama pool (surprise tag = laugh)."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        # Force selection of the last element (the surprise tag added by high drama)
        mock_rng.choice.side_effect = lambda pool: pool[-1]
        result = _inject_emotion_tags(
            "Wild swings here!",
            {"trend": "bullish", "price_change_pct": 5.0, "volatility": "high"},
        )
    assert "laugh" in _get_tags(result)


def test_big_move_uses_drama_pool() -> None:
    """Price change > 3% should use the high-drama pool regardless of volatility."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[-1]
        result = _inject_emotion_tags(
            "Massive rally!",
            {"trend": "bullish", "price_change_pct": 4.0, "volatility": "low"},
        )
    assert "laugh" in _get_tags(result)


# ── Tag stripping ──────────────────────────────────────────────────


def test_existing_tags_stripped_before_injection() -> None:
    """Any pre-existing emotion tags in the text should be removed."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 1.0  # Suppress both injections
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "<laugh> Some text <gasp>",
            {"trend": "sideways"},
        )
    assert _count_tags(result) == 0
    assert "Some text" in result


# ── No tags when random rolls high ──────────────────────────────────


def test_no_tags_when_random_exceeds_threshold() -> None:
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 1.0  # Always above threshold
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Steady as she goes.",
            {"trend": "sideways"},
        )
    assert _count_tags(result) == 0


# ── Tag placement ───────────────────────────────────────────────────


def test_tag_inserted_after_punctuation_not_prepended() -> None:
    """When text has a punctuation pause, the first tag goes after it, not at the start."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.side_effect = [0.0, 1.0]  # first injection on, second off
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Bulls charging, ahead!",
            {"trend": "bullish", "price_change_pct": 1.5, "volatility": "low"},
        )
    assert not result.startswith("<"), "tag should be after the comma, not prepended"
    assert len(_get_tags(result)) == 1


def test_second_tag_appended_to_end() -> None:
    """When second random check passes but first fails, a tag is appended at the end."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.side_effect = [1.0, 0.0]  # first off, second on
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Bulls charging.",
            {"trend": "bullish", "price_change_pct": 1.5, "volatility": "low"},
        )
    tags = _get_tags(result)
    assert len(tags) == 1
    assert result.endswith(">"), "tag should be appended at the end"


# ── Default analysis values ─────────────────────────────────────────


def test_missing_analysis_keys_uses_defaults() -> None:
    """An empty analysis dict should not crash — defaults applied and neutral tags injected."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags("Some commentary.", {})
    # Empty dict → trend defaults to "sideways" → neutral tag pool (chuckle/sniffle/yawn)
    # random.random()=0.0 fires both injection thresholds → 2 neutral tags
    assert "Some commentary" in result
    tags = _get_tags(result)
    assert len(tags) == 2
    assert all(t in ("chuckle", "sniffle", "yawn") for t in tags)


# ── generate_commentary live_move partial args ───────────────────────


def _make_analysis() -> AnalysisResult:
    return AnalysisResult(
        trend="bullish",
        price_change_pct=1.5,
        current_price=150.0,
        open_price=148.0,
        high=151.0,
        low=147.0,
        volume_trend="normal",
        sma_cross=None,
        volatility="low",
        rsi=55.0,
    )


def test_generate_commentary_live_move_without_pct_uses_opening_prompt() -> None:
    """Passing live_move without live_move_pct falls back to the opening-commentary branch."""
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        return_value="Test commentary.",
    ) as mock_llm:
        result = generate_commentary(
            _make_analysis(),
            "AAPL",
            "Apple",
            live_move=1.5,
            # live_move_pct and live_direction intentionally omitted
        )
    assert isinstance(result, str)
    assert len(result) > 0
    prompt_used = mock_llm.call_args[0][0]
    assert "Opening commentary" in prompt_used
    assert "just moved" not in prompt_used


def test_generate_commentary_exception_returns_fallback() -> None:
    """LLM failure should return the fallback string and never raise."""
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        side_effect=RuntimeError("model crashed"),
    ):
        result = generate_commentary(_make_analysis(), "AAPL", "Apple")
    assert result == "The commentator is having technical difficulties!"


def test_generate_commentary_live_move_all_args() -> None:
    """When all live_move args are provided, the prompt uses the live-move branch."""
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        return_value="Up we go.",
    ) as mock_llm:
        result = generate_commentary(
            _make_analysis(),
            "AAPL",
            "Apple",
            live_move=1.5,
            live_move_pct=1.0,
            live_direction="up",
        )
    assert isinstance(result, str)
    prompt_used = mock_llm.call_args[0][0]
    assert "just moved" in prompt_used


# ── generate_commentary prompt content ──────────────────────────────


def test_generate_commentary_with_previous_commentary() -> None:
    """Previous commentary lines should appear verbatim in the prompt."""
    prior = ["Bulls are charging!", "Price holding firm."]
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        return_value="Great.",
    ) as mock_llm:
        generate_commentary(_make_analysis(), "AAPL", "Apple", previous_commentary=prior)
    prompt_used = mock_llm.call_args[0][0]
    assert "Prior commentary" in prompt_used
    assert "Bulls are charging!" in prompt_used
    assert "Price holding firm." in prompt_used


def test_generate_commentary_previous_commentary_truncated_to_five() -> None:
    """Only the last 5 prior lines are included; earlier ones are omitted."""
    prior = [f"Line {i}" for i in range(8)]  # 8 lines → only last 5 (3–7) should appear
    with patch("commentator.commentary._generate_with_llama_cpp", return_value="OK.") as mock_llm:
        generate_commentary(_make_analysis(), "AAPL", "Apple", previous_commentary=prior)
    prompt_used = mock_llm.call_args[0][0]
    assert "Line 7" in prompt_used
    assert "Line 3" in prompt_used  # 5th from last (index 3)
    assert "Line 2" not in prompt_used  # 6th from last, excluded


def test_generate_commentary_empty_previous_commentary_omits_prior_section() -> None:
    """An empty previous_commentary list should not add the prior section."""
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        return_value="Quiet.",
    ) as mock_llm:
        generate_commentary(_make_analysis(), "AAPL", "Apple", previous_commentary=[])
    prompt_used = mock_llm.call_args[0][0]
    assert "Prior commentary" not in prompt_used


def test_generate_commentary_prompt_includes_rsi() -> None:
    """RSI should appear in the prompt when the analysis includes a value."""
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        return_value="On target.",
    ) as mock_llm:
        generate_commentary(_make_analysis(), "AAPL", "Apple")  # rsi=55.0 in _make_analysis
    prompt_used = mock_llm.call_args[0][0]
    assert "RSI: 55.0" in prompt_used


def test_generate_commentary_prompt_excludes_rsi_when_none() -> None:
    """RSI should not appear in the prompt when the analysis has rsi=None."""
    analysis = AnalysisResult(
        trend="sideways",
        price_change_pct=0.0,
        current_price=100.0,
        open_price=100.0,
        high=101.0,
        low=99.0,
        volume_trend="normal",
        sma_cross=None,
        volatility="low",
        rsi=None,
    )
    with patch("commentator.commentary._generate_with_llama_cpp", return_value="Flat.") as mock_llm:
        generate_commentary(analysis, "AAPL", "Apple")
    prompt_used = mock_llm.call_args[0][0]
    assert "RSI" not in prompt_used


def test_generate_commentary_prompt_includes_golden_cross() -> None:
    """A golden_cross in the analysis should produce the label in the prompt."""
    analysis = AnalysisResult(
        trend="bullish",
        price_change_pct=2.0,
        current_price=150.0,
        open_price=148.0,
        high=151.0,
        low=147.0,
        volume_trend="heavy",
        sma_cross="golden_cross",
        volatility="low",
        rsi=60.0,
    )
    with patch(
        "commentator.commentary._generate_with_llama_cpp",
        return_value="Golden!",
    ) as mock_llm:
        generate_commentary(analysis, "AAPL", "Apple")
    prompt_used = mock_llm.call_args[0][0]
    assert "Golden Cross" in prompt_used


def test_generate_commentary_prompt_includes_death_cross() -> None:
    """A death_cross in the analysis should produce the label in the prompt."""
    analysis = AnalysisResult(
        trend="bearish",
        price_change_pct=-2.0,
        current_price=90.0,
        open_price=92.0,
        high=93.0,
        low=89.0,
        volume_trend="light",
        sma_cross="death_cross",
        volatility="medium",
        rsi=38.0,
    )
    with patch("commentator.commentary._generate_with_llama_cpp", return_value="Down.") as mock_llm:
        generate_commentary(analysis, "AAPL", "Apple")
    prompt_used = mock_llm.call_args[0][0]
    assert "Death Cross" in prompt_used


# ── _generate_with_llama_cpp post-processing ─────────────────────────


def _llm_response(content: str) -> dict:
    """Minimal llama.cpp chat-completion response for testing."""
    return {"choices": [{"message": {"content": content}}]}


def test_llm_postprocess_strips_think_block() -> None:
    """<think>...</think> blocks should be stripped from the returned text."""
    with patch("commentator.commentary._get_commentary_llm") as mock_get:
        mock_get.return_value.create_chat_completion.return_value = _llm_response(
            "<think>Some reasoning.</think> Apple is surging!"
        )
        result = _generate_with_llama_cpp("prompt")
    assert "<think>" not in result
    assert "Some reasoning." not in result
    assert "Apple is surging!" in result


def test_llm_postprocess_strips_multiline_think_block() -> None:
    """Multi-line <think> blocks (requires DOTALL flag) should also be stripped."""
    with patch("commentator.commentary._get_commentary_llm") as mock_get:
        mock_get.return_value.create_chat_completion.return_value = _llm_response(
            "<think>\nLine 1\nLine 2\n</think>Market holds steady."
        )
        result = _generate_with_llama_cpp("prompt")
    assert "<think>" not in result
    assert "Line 1" not in result
    assert "Market holds steady." in result


def test_llm_postprocess_strips_folks() -> None:
    """'Folks' (a banned phrase) should be removed from the returned text."""
    with patch("commentator.commentary._get_commentary_llm") as mock_get:
        mock_get.return_value.create_chat_completion.return_value = _llm_response(
            "Folks, Apple is rallying hard today!"
        )
        result = _generate_with_llama_cpp("prompt")
    assert "Folks" not in result
    assert "Apple is rallying hard today!" in result


def test_llm_postprocess_strips_ladies_and_gentlemen() -> None:
    """'Ladies and gentlemen' (a banned phrase) should be removed from the returned text."""
    with patch("commentator.commentary._get_commentary_llm") as mock_get:
        mock_get.return_value.create_chat_completion.return_value = _llm_response(
            "Ladies and gentlemen, the bulls are charging!"
        )
        result = _generate_with_llama_cpp("prompt")
    assert "Ladies and gentlemen" not in result
    assert "bulls are charging!" in result


# ── _PAUSE_RE: decimal-number guard ──────────────────────────────────


def test_decimal_price_not_treated_as_pause() -> None:
    """A period inside a decimal price (e.g. '150.50') must not be an injection point.

    _PAUSE_RE uses `\\.(?!\\d)` to skip periods followed by a digit, so a price
    like '$150.50' contains no valid pause and the tag should be prepended.
    """
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.side_effect = [0.0, 1.0]  # first injection fires, second skipped
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags(
            "Apple sitting at 150.50 looks strong",
            {"trend": "bullish", "price_change_pct": 1.5, "volatility": "low"},
        )
    assert result.startswith("<"), (
        "tag should be prepended when text has no valid punctuation pause"
    )
    assert "150.50" in result, "price must be preserved intact"
