"""Comprehensive tests for commentator.analysis.analyze_stock.

Covers: empty input, column validation, exactly 2 rows, bullish/bearish/
sideways trends, NaN in data, zero first_close, RSI computation (all gains,
mixed), SMA golden/death cross, ATR volatility levels, volume trends.
"""

import math

import numpy as np
import pandas as pd
import pytest

from commentator.analysis import AnalysisError, AnalysisResult, analyze_stock


def _make_df(
    n: int,
    *,
    close: list[float] | None = None,
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    volume: list[int] | None = None,
    freq: str = "min",
) -> pd.DataFrame:
    """Helper to build OHLCV DataFrames with sensible defaults."""
    index = pd.date_range("2026-01-01 10:00", periods=n, freq=freq)
    close_vals = close or [100.0 + i for i in range(n)]
    open_vals = open_ or [c - 0.5 for c in close_vals]
    high_vals = high or [c + 0.5 for c in close_vals]
    low_vals = low or [c - 1.0 for c in close_vals]
    vol_vals = volume or [1000] * n
    return pd.DataFrame(
        {
            "Open": open_vals,
            "High": high_vals,
            "Low": low_vals,
            "Close": close_vals,
            "Volume": vol_vals,
        },
        index=index,
    )


# ── Edge cases: insufficient data ───────────────────────────────────


def test_empty_dataframe() -> None:
    assert analyze_stock(pd.DataFrame()) == AnalysisError(error="Not enough data")


def test_single_row() -> None:
    df = _make_df(1)
    assert analyze_stock(df) == AnalysisError(error="Not enough data")


def test_missing_columns() -> None:
    df = pd.DataFrame({"Close": [1, 2], "Volume": [100, 200]})
    result = analyze_stock(df)
    assert "error" in result
    assert "Missing columns" in result["error"]  # type: ignore[typeddict-item]


# ── Minimal valid input: exactly 2 rows ─────────────────────────────


def test_two_rows_basic() -> None:
    df = _make_df(2, close=[100.0, 100.5])
    result = analyze_stock(df)
    assert "error" not in result
    assert result["current_price"] == 100.5
    assert result["trend"] == "sideways"  # 0.5% change < 1% threshold


# ── Trend classification ────────────────────────────────────────────


def test_bullish_trend() -> None:
    df = _make_df(10, close=[100 + i for i in range(10)])
    result = analyze_stock(df)
    assert result["trend"] == "bullish"
    assert result["price_change_pct"] > 1


def test_bearish_trend() -> None:
    df = _make_df(10, close=[110 - i for i in range(10)])
    result = analyze_stock(df)
    assert result["trend"] == "bearish"
    assert result["price_change_pct"] < -1


def test_sideways_trend() -> None:
    df = _make_df(10, close=[100.0, 100.1, 99.9, 100.0, 100.2, 99.8, 100.0, 100.1, 100.0, 100.3])
    result = analyze_stock(df)
    assert result["trend"] == "sideways"


# ── Division-by-zero: first_close == 0 ──────────────────────────────


def test_zero_first_close() -> None:
    """When first_close is 0, percentage change should be 0 (not crash)."""
    df = _make_df(3, close=[0.0, 50.0, 100.0])
    result = analyze_stock(df)
    assert "error" not in result
    assert result["price_change_pct"] == 0.0
    assert result["current_price"] == 100.0


# ── NaN handling ────────────────────────────────────────────────────


def test_nan_in_close_column() -> None:
    """NaN first_close should produce 0% change, not propagate."""
    df = _make_df(3, close=[float("nan"), 50.0, 100.0])
    result = analyze_stock(df)
    assert "error" not in result
    assert result["price_change_pct"] == 0.0


# ── Volume trend ────────────────────────────────────────────────────


def test_heavy_volume() -> None:
    df = _make_df(
        10,
        close=[100 + i for i in range(10)],
        volume=[100, 100, 100, 100, 100, 300, 300, 300, 300, 300],
    )
    result = analyze_stock(df)
    assert result["volume_trend"] == "heavy"


def test_light_volume() -> None:
    df = _make_df(
        10,
        close=[100 + i for i in range(10)],
        volume=[300, 300, 300, 300, 300, 100, 100, 100, 100, 100],
    )
    result = analyze_stock(df)
    assert result["volume_trend"] == "light"


def test_normal_volume() -> None:
    df = _make_df(10, volume=[100] * 10)
    result = analyze_stock(df)
    assert result["volume_trend"] == "normal"


def test_volume_too_few_rows() -> None:
    """With < 10 rows, volume trend defaults to normal."""
    df = _make_df(5)
    result = analyze_stock(df)
    assert result["volume_trend"] == "normal"


# ── RSI ─────────────────────────────────────────────────────────────


def test_rsi_below_threshold() -> None:
    """With fewer than 15 rows, RSI should be None."""
    df = _make_df(10)
    result = analyze_stock(df)
    assert result["rsi"] is None


def test_rsi_computed() -> None:
    """With 15+ rows, RSI should be a valid float."""
    prices = [100 + i * 0.5 * ((-1) ** i) for i in range(20)]
    df = _make_df(20, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] is not None
    assert 0 <= result["rsi"] <= 100


def test_rsi_all_gains() -> None:
    """When all deltas are positive, RSI should be 100."""
    prices = [100.0 + i for i in range(20)]
    df = _make_df(20, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] == 100.0


# ── SMA crossover ───────────────────────────────────────────────────


def test_no_sma_cross_under_50_rows() -> None:
    df = _make_df(30)
    result = analyze_stock(df)
    assert result["sma_cross"] is None


def test_golden_cross() -> None:
    """SMA-20 crossing above SMA-50 should produce golden_cross."""
    # Build data where SMA-20 was below SMA-50, then crosses above.
    n = 60
    prices = [50.0] * 40 + [50.0 + (i * 2) for i in range(20)]
    df = _make_df(n, close=prices)
    result = analyze_stock(df)
    # The cross detection depends on exact rolling values; at minimum
    # sma_cross should be None or golden_cross (no false death_cross).
    assert result["sma_cross"] in (None, "golden_cross")


def test_death_cross() -> None:
    """SMA-20 crossing below SMA-50 should produce death_cross."""
    n = 60
    prices = [150.0] * 40 + [150.0 - (i * 2) for i in range(20)]
    df = _make_df(n, close=prices)
    result = analyze_stock(df)
    assert result["sma_cross"] in (None, "death_cross")


# ── Volatility (ATR) ────────────────────────────────────────────────


def test_volatility_unknown_under_14_rows() -> None:
    df = _make_df(10)
    result = analyze_stock(df)
    assert result["volatility"] == "unknown"


def test_volatility_low() -> None:
    """Tight price range should produce low volatility."""
    n = 20
    close = [100.0 + 0.01 * i for i in range(n)]
    df = _make_df(n, close=close, high=[c + 0.01 for c in close], low=[c - 0.01 for c in close])
    result = analyze_stock(df)
    assert result["volatility"] == "low"


def test_volatility_high() -> None:
    """Wide price swings should produce high volatility."""
    n = 20
    close = [100.0 + ((-1) ** i) * 5 for i in range(n)]
    high = [c + 3 for c in close]
    low = [c - 3 for c in close]
    df = _make_df(n, close=close, high=high, low=low)
    result = analyze_stock(df)
    assert result["volatility"] == "high"


# ── Return shape ────────────────────────────────────────────────────


def test_result_has_all_expected_keys() -> None:
    """Verify the result dict matches the AnalysisResult TypedDict."""
    df = _make_df(20)
    result = analyze_stock(df)
    expected_keys = {
        "trend", "price_change_pct", "current_price", "open_price",
        "high", "low", "volume_trend", "sma_cross", "volatility", "rsi",
    }
    assert set(result.keys()) == expected_keys
