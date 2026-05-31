"""Pure decision and formatting helpers for the live UI.

These were inline in app.py (a Streamlit script, which can't be unit-tested), so
the update-cadence logic, prefetch key, slider bounds, and display cleanup had no
tests — the kind of gap that let the refresh-slider min==max crash ship. Keeping
them here, free of Streamlit, makes them testable and reusable.
"""

import re

# Emotion-tag (and any stray angle-bracket directive) stripper for display: the
# tags steer the TTS voice but shouldn't be shown as literal "<chuckle>".
_TAG_RE = re.compile(r"<[^>]+>")


def clean_commentary(text: str) -> str:
    """Commentary text with emotion tags removed and whitespace collapsed."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def refresh_bounds(period: str) -> tuple[int, int, int]:
    """(min, max, default) seconds for the live refresh slider.

    The 15m window caps the max at 5s, so the min must be below it — otherwise
    Streamlit's slider raises "min_value must be less than max_value".
    """
    max_s = 5 if period == "15m" else 120
    min_s = 1 if period == "15m" else 5
    default = max(min_s, min(15, max_s))
    return min_s, max_s, default


def prefetch_key(
    ticker: str, period: str, interval: str, voice: str, speed: float, personality: str
) -> tuple:
    """Identity of a settings snapshot for the speculative no-move line. Price is
    intentionally excluded — a prefetched line is reused when the next tick's move
    is below the tolerance, not only when the price is byte-identical."""
    return (ticker, period, interval, voice, round(speed, 2), personality)


def is_significant_move(
    price_changed: bool, move_pct: "float | None", tolerance_pct: float
) -> bool:
    """True when this tick is a real move (>= tolerance) deserving fresh,
    move-aware commentary rather than the prefetched no-move line."""
    return bool(price_changed and move_pct is not None and abs(move_pct) >= tolerance_pct)


def should_refresh(
    *,
    force: bool,
    live: bool,
    live_just_started: bool,
    interval_elapsed: bool,
    data_params_changed: bool,
) -> bool:
    """Whether to refetch data this run."""
    return bool(
        force
        or live_just_started
        or interval_elapsed
        or (not live and data_params_changed)
    )


def should_comment(
    *,
    force: bool,
    live: bool,
    price_changed: bool,
    live_just_started: bool,
    interval_elapsed: bool,
) -> bool:
    """Whether to (re)generate commentary this run."""
    return bool(force or (live and (price_changed or live_just_started or interval_elapsed)))
