import pandas as pd

from commentator.analysis import analyze_stock


def test_analyze_stock_requires_at_least_two_rows() -> None:
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1000],
        }
    )

    result = analyze_stock(df)

    assert result == {"error": "Not enough data"}


def test_analyze_stock_detects_bullish_move_and_heavy_volume() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="min")
    df = pd.DataFrame(
        {
            "Open": [100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5],
            "High": [101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5, 105, 106],
            "Low": [99.5, 100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104],
            "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 110],
            "Volume": [100, 100, 100, 100, 100, 300, 300, 300, 300, 300],
        },
        index=index,
    )

    result = analyze_stock(df)

    assert result["trend"] == "bullish"
    assert result["price_change_pct"] == 10.0
    assert result["volume_trend"] == "heavy"
    assert result["current_price"] == 110.0
    assert result["high"] == 106.0
    assert result["low"] == 99.5
    assert result["volatility"] == "unknown"
