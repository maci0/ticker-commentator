"""Comprehensive tests for commentator.analysis.analyze_stock.

Covers: empty input, column validation, exactly 2 rows, bullish/bearish/
sideways trends, NaN in data, zero first_close, RSI computation (all gains,
mixed), SMA golden/death cross, ATR volatility levels, volume trends.
"""


import pandas as pd

from commentator.analysis import AnalysisError, analyze_stock


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
    """With 15+ rows, RSI should match the expected computed value.

    For the alternating price series, the 14-period window ending at the last
    bar has avg_gain=5.75 and avg_loss=6.25, giving RS=0.92 and RSI≈47.9.
    """
    prices = [100 + i * 0.5 * ((-1) ** i) for i in range(20)]
    df = _make_df(20, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] is not None
    assert result["rsi"] == 47.9


def test_rsi_all_gains() -> None:
    """When all deltas are positive, RSI should be 100."""
    prices = [100.0 + i for i in range(20)]
    df = _make_df(20, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] == 100.0


def test_rsi_all_losses() -> None:
    """When all deltas are negative, RSI should be 0."""
    prices = [100.0 - i for i in range(20)]
    df = _make_df(20, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] == 0.0


def test_rsi_flat_prices_returns_none() -> None:
    """When all prices are identical (no change), RSI should be None.

    avg_gain == 0 and avg_loss == 0 — no directional information exists,
    so returning 100 (the all-gains path) would be incorrect.
    """
    prices = [100.0] * 20
    df = _make_df(20, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] is None


# ── SMA crossover ───────────────────────────────────────────────────


def test_no_sma_cross_under_50_rows() -> None:
    df = _make_df(30)
    result = analyze_stock(df)
    assert result["sma_cross"] is None


def test_golden_cross() -> None:
    """SMA-20 crossing above SMA-50 should produce golden_cross.

    Design: 51 rows all at 100.0, then a final row at 200.0.
    At the second-to-last row: SMA-20 = SMA-50 = 100.0 (prev_20 <= prev_50).
    At the last row: SMA-20 = (19*100 + 200)/20 = 105.0,
                     SMA-50 = (49*100 + 200)/50 = 102.0  (cur_20 > cur_50).
    """
    prices = [100.0] * 50 + [200.0]
    df = _make_df(51, close=prices)
    result = analyze_stock(df)
    assert result["sma_cross"] == "golden_cross"


def test_death_cross() -> None:
    """SMA-20 crossing below SMA-50 should produce death_cross.

    Design: 51 rows all at 100.0, then a final row at 10.0.
    At the second-to-last row: SMA-20 = SMA-50 = 100.0 (prev_20 >= prev_50).
    At the last row: SMA-20 = (19*100 + 10)/20 = 95.5,
                     SMA-50 = (49*100 + 10)/50 = 98.2  (cur_20 < cur_50).
    """
    prices = [100.0] * 50 + [10.0]
    df = _make_df(51, close=prices)
    result = analyze_stock(df)
    assert result["sma_cross"] == "death_cross"


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


def test_volatility_medium() -> None:
    """Moderate price range should produce medium volatility.

    Flat close at 100.0, high=100.5, low=99.0 gives H-L=1.5.
    ATR ≈ 1.5, ATR/price = 1.5% — between the 0.8% and 2% thresholds.
    """
    n = 20
    close = [100.0] * n
    df = _make_df(n, close=close, high=[100.5] * n, low=[99.0] * n)
    result = analyze_stock(df)
    assert result["volatility"] == "medium"


def test_volatility_high() -> None:
    """Wide price swings should produce high volatility."""
    n = 20
    close = [100.0 + ((-1) ** i) * 5 for i in range(n)]
    high = [c + 3 for c in close]
    low = [c - 3 for c in close]
    df = _make_df(n, close=close, high=high, low=low)
    result = analyze_stock(df)
    assert result["volatility"] == "high"


# ── Volume boundary ─────────────────────────────────────────────────


def test_volume_at_9_rows_is_normal() -> None:
    """9 rows is below the 10-row threshold, so volume trend is always normal."""
    df = _make_df(9, volume=[300, 300, 300, 300, 300, 100, 100, 100, 100])
    result = analyze_stock(df)
    assert result["volume_trend"] == "normal"


# ── RSI boundary ─────────────────────────────────────────────────────


def test_rsi_at_14_rows_is_none() -> None:
    """14 rows is below the 15-row threshold; RSI must be None."""
    df = _make_df(14)
    result = analyze_stock(df)
    assert result["rsi"] is None


def test_rsi_at_15_rows_is_computed() -> None:
    """15 rows is the minimum for RSI computation; result must be numeric."""
    prices = [100.0 + i for i in range(15)]  # all gains → RSI = 100
    df = _make_df(15, close=prices)
    result = analyze_stock(df)
    assert result["rsi"] == 100.0


# ── ATR / volatility boundary ────────────────────────────────────────


def test_volatility_at_13_rows_is_unknown() -> None:
    """13 rows is below the 14-row ATR threshold; volatility must be unknown."""
    df = _make_df(13)
    result = analyze_stock(df)
    assert result["volatility"] == "unknown"


def test_volatility_at_14_rows_is_unknown() -> None:
    """14 rows enters the ATR branch but the first TR is NaN (no prior close),
    making the 14-period rolling mean NaN.  The practical minimum is 15 rows.
    """
    n = 14
    close = [100.0] * n
    df = _make_df(n, close=close, high=[100.5] * n, low=[99.0] * n)
    result = analyze_stock(df)
    assert result["volatility"] == "unknown"


def test_volatility_at_15_rows_is_computed() -> None:
    """15 rows provides 14 non-NaN true-range values; volatility must be valid."""
    n = 15
    close = [100.0] * n
    df = _make_df(n, close=close, high=[100.5] * n, low=[99.0] * n)
    result = analyze_stock(df)
    assert result["volatility"] != "unknown"


# ── SMA cross boundary ───────────────────────────────────────────────


def test_no_sma_cross_at_49_rows() -> None:
    """49 rows is one below the 50-row SMA threshold; sma_cross must be None."""
    df = _make_df(49)
    result = analyze_stock(df)
    assert result["sma_cross"] is None


def test_no_sma_cross_when_no_crossover_occurred() -> None:
    """52 rows of monotonically rising prices: SMA-20 > SMA-50 throughout, no cross."""
    df = _make_df(52)
    result = analyze_stock(df)
    assert result["sma_cross"] is None


def test_sma_cross_at_exactly_50_rows_is_none() -> None:
    """With exactly 50 rows the second-to-last SMA-50 is NaN; sma_cross must be None."""
    df = _make_df(50)
    result = analyze_stock(df)
    assert result["sma_cross"] is None


# ── NaN current price ────────────────────────────────────────────────


def test_nan_last_close_gives_unknown_volatility() -> None:
    """NaN in the last Close row should default current_price to 0 → unknown volatility,
    zero price_change_pct (not a false -100% bearish reading), and sideways trend."""
    n = 20
    close = [100.0] * n
    close[-1] = float("nan")
    df = _make_df(n, close=close, high=[100.5] * n, low=[99.0] * n)
    result = analyze_stock(df)
    assert "error" not in result
    assert result["current_price"] == 0.0
    assert result["volatility"] == "unknown"
    assert result["price_change_pct"] == 0.0
    assert result["trend"] == "sideways"


# ── Zero earlier volume ──────────────────────────────────────────────


def test_zero_earlier_volume_gives_normal_trend() -> None:
    """When the first-5 volume average is zero, the division-by-zero guard must
    return 'normal' rather than crash."""
    df = _make_df(
        10,
        close=[100 + i for i in range(10)],
        volume=[0, 0, 0, 0, 0, 100, 100, 100, 100, 100],
    )
    result = analyze_stock(df)
    assert result["volume_trend"] == "normal"


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
