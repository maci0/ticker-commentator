"""Tests for commentator.live — the pure decision/format helpers extracted from
the Streamlit app so they can actually be tested."""

import re

from hypothesis import given
from hypothesis import strategies as st

from commentator.live import (
    clean_commentary,
    is_significant_move,
    prefetch_key,
    refresh_bounds,
    should_comment,
    should_refresh,
)

# ── clean_commentary ─────────────────────────────────────────────────


def test_clean_commentary_strips_tags() -> None:
    assert clean_commentary("Bulls charge! <chuckle>") == "Bulls charge!"


def test_clean_commentary_collapses_whitespace() -> None:
    assert clean_commentary("Bulls   charge  \n ahead") == "Bulls charge ahead"


def test_clean_commentary_plain_unchanged() -> None:
    assert clean_commentary("Bulls charge ahead") == "Bulls charge ahead"


@given(text=st.text(max_size=200))
def test_clean_commentary_never_crashes_no_tags(text: str) -> None:
    out = clean_commentary(text)
    assert isinstance(out, str)
    assert not re.search(r"<[^>]+>", out)  # no well-formed tag span remains
    assert clean_commentary(out) == out  # idempotent


# ── refresh_bounds (the slider min<max invariant) ───────────────────


@given(period=st.text(max_size=10))
def test_refresh_bounds_min_lt_max_always(period: str) -> None:
    """The crash that shipped was min==max; assert min < max for any period."""
    lo, hi, default = refresh_bounds(period)
    assert lo < hi
    assert lo <= default <= hi


def test_refresh_bounds_15m() -> None:
    assert refresh_bounds("15m") == (1, 5, 5)


def test_refresh_bounds_other() -> None:
    assert refresh_bounds("1d") == (5, 120, 15)


# ── is_significant_move ──────────────────────────────────────────────


def test_significant_move_true_above_tolerance() -> None:
    assert is_significant_move(True, 0.5, 0.05) is True


def test_significant_move_false_below_tolerance() -> None:
    assert is_significant_move(True, 0.01, 0.05) is False


def test_significant_move_false_when_unchanged() -> None:
    assert is_significant_move(False, 1.0, 0.05) is False


def test_significant_move_false_when_none() -> None:
    assert is_significant_move(True, None, 0.05) is False


@given(
    changed=st.booleans(),
    pct=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    tol=st.floats(min_value=0, max_value=100, allow_nan=False),
)
def test_significant_move_returns_bool(changed: bool, pct, tol: float) -> None:
    assert isinstance(is_significant_move(changed, pct, tol), bool)


# ── prefetch_key ─────────────────────────────────────────────────────


def test_prefetch_key_excludes_price_includes_settings() -> None:
    k = prefetch_key("AAPL", "1d", "1m", "leo", 1.3, "sports")
    assert k == ("AAPL", "1d", "1m", "leo", 1.3, "sports")


def test_prefetch_key_differs_by_personality() -> None:
    a = prefetch_key("AAPL", "1d", "1m", "leo", 1.3, "sports")
    b = prefetch_key("AAPL", "1d", "1m", "leo", 1.3, "noir")
    assert a != b


# ── should_refresh / should_comment ──────────────────────────────────


def test_should_refresh_force() -> None:
    assert should_refresh(
        force=True,
        live=False,
        live_just_started=False,
        interval_elapsed=False,
        data_params_changed=False,
    )


def test_should_refresh_static_only_on_data_change() -> None:
    assert not should_refresh(
        force=False,
        live=False,
        live_just_started=False,
        interval_elapsed=False,
        data_params_changed=False,
    )
    assert should_refresh(
        force=False,
        live=False,
        live_just_started=False,
        interval_elapsed=False,
        data_params_changed=True,
    )


def test_should_comment_needs_live_or_force() -> None:
    # price change alone (not live, not forced) does not comment
    assert not should_comment(
        force=False,
        live=False,
        price_changed=True,
        live_just_started=False,
        interval_elapsed=False,
    )
    assert should_comment(
        force=False,
        live=True,
        price_changed=True,
        live_just_started=False,
        interval_elapsed=False,
    )
    assert should_comment(
        force=True,
        live=False,
        price_changed=False,
        live_just_started=False,
        interval_elapsed=False,
    )


@given(a=st.booleans(), b=st.booleans(), c=st.booleans(), d=st.booleans(), e=st.booleans())
def test_decision_helpers_return_bool(a, b, c, d, e) -> None:
    assert isinstance(
        should_refresh(
            force=a, live=b, live_just_started=c, interval_elapsed=d, data_params_changed=e
        ),
        bool,
    )
    assert isinstance(
        should_comment(force=a, live=b, price_changed=c, live_just_started=d, interval_elapsed=e),
        bool,
    )
