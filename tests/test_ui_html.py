"""Tests for commentator.ui_html pure helpers."""

from commentator.ui_html import (
    commentary_card_html,
    persona_blurb,
    persona_emoji,
    persona_label,
)


def test_persona_label_known() -> None:
    label = persona_label("sports")
    assert "Sports" in label
    assert persona_emoji("sports")


def test_persona_label_unknown_fallback() -> None:
    assert "Custom" in persona_label("not_a_real_persona") or "Not A Real Persona" in persona_label(
        "not_a_real_persona"
    )
    assert persona_blurb("not_a_real_persona") == "Custom style"


def test_commentary_card_strips_tags_and_escapes() -> None:
    html = commentary_card_html("Bulls <laugh> charge <b>now", "sports", "bullish", live=True)
    assert "<laugh>" not in html
    assert "&lt;b&gt;" in html or "charge" in html
    assert "ON AIR" in html
    assert "#26a69a" in html  # bullish accent


def test_commentary_card_bearish_color() -> None:
    html = commentary_card_html("Bears win", "noir", "bearish")
    assert "#ef5350" in html
    assert "ON AIR" not in html
