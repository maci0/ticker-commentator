"""Tests for commentator.charts pure builders."""

import json

import pandas as pd

from commentator.charts import build_candlestick_figure, tradingview_embed_html, tv_interval


def _sample_df(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


def test_tv_interval_known_and_fallback() -> None:
    assert tv_interval("15m") == "15"
    assert tv_interval("1d") == "D"
    assert tv_interval("unknown") == "D"


def test_build_candlestick_figure_has_price_and_volume() -> None:
    df = _sample_df(60)
    fig = build_candlestick_figure(
        df, sma_20=df["Close"].rolling(20).mean(), sma_50=df["Close"].rolling(50).mean()
    )
    names = [t.name for t in fig.data]
    assert "Price" in names
    assert "Volume" in names
    assert "SMA 20" in names
    assert "SMA 50" in names


def test_build_candlestick_figure_skips_sma_when_short() -> None:
    df = _sample_df(10)
    fig = build_candlestick_figure(df)
    names = [t.name for t in fig.data]
    assert "SMA 20" not in names
    assert "SMA 50" not in names


def test_tradingview_embed_html_contains_symbol_and_interval() -> None:
    html = tradingview_embed_html("AAPL", "15m")
    assert "AAPL" in html
    assert "tradingview" in html
    # Config JSON is embedded inside the script tag.
    assert '"symbol": "AAPL"' in html or '"symbol":"AAPL"' in html
    assert '"interval": "15"' in html or '"interval":"15"' in html
    # Ensure the payload is valid JSON when extracted.
    start = html.index("{")
    end = html.rindex("}") + 1
    cfg = json.loads(html[start:end])
    assert cfg["symbol"] == "AAPL"
    assert cfg["interval"] == "15"
