"""Tests for commentator.commentary — emotion tag injection.

These tests exercise all sentiment paths and edge cases of
_inject_emotion_tags without requiring the LLM.
"""

import re
from unittest.mock import patch

from commentator.commentary import _inject_emotion_tags


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
    assert len(tags) >= 1
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
    assert len(tags) >= 1
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
    assert len(tags) >= 1
    assert all(t in ("chuckle", "sniffle", "yawn") for t in tags)


def test_high_volatility_adds_gasp() -> None:
    """High volatility should add gasp to the pool."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        # Force selection of the last element (gasp, added by high vol)
        mock_rng.choice.side_effect = lambda pool: pool[-1]
        result = _inject_emotion_tags(
            "Wild swings here!",
            {"trend": "bullish", "price_change_pct": 5.0, "volatility": "high"},
        )
    assert "gasp" in _get_tags(result)


def test_big_move_adds_gasp() -> None:
    """Price change > 3% should add gasp regardless of volatility."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[-1]
        result = _inject_emotion_tags(
            "Massive rally!",
            {"trend": "bullish", "price_change_pct": 4.0, "volatility": "low"},
        )
    assert "gasp" in _get_tags(result)


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


# ── Default analysis values ─────────────────────────────────────────


def test_missing_analysis_keys_uses_defaults() -> None:
    """An empty analysis dict should not crash — defaults applied."""
    with patch("commentator.commentary.random") as mock_rng:
        mock_rng.random.return_value = 0.0
        mock_rng.choice.side_effect = lambda pool: pool[0]
        result = _inject_emotion_tags("Some commentary.", {})
    # Should not crash; result should contain original text
    assert "Some commentary" in result
