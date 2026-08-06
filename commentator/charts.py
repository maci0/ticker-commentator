"""Chart builders extracted from the Streamlit app for testability."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Yahoo Finance interval → TradingView advanced-chart interval code.
_TV_INTERVAL_MAP: dict[str, str] = {
    "1m": "1",
    "2m": "2",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "1d": "D",
    "1wk": "W",
}


def tv_interval(interval: str) -> str:
    """Map a yfinance interval string to a TradingView interval code."""
    return _TV_INTERVAL_MAP.get(interval, "D")


def build_candlestick_figure(
    df: pd.DataFrame,
    sma_20: pd.Series | None = None,
    sma_50: pd.Series | None = None,
) -> go.Figure:
    """Build the dark-theme candlestick + volume chart used in the main UI."""
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    if sma_20 is not None and len(df) >= 20:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_20,
                name="SMA 20",
                line=dict(color="#FFA500", width=1),
                opacity=0.8,
            )
        )
    if sma_50 is not None and len(df) >= 50:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_50,
                name="SMA 50",
                line=dict(color="#1E90FF", width=1),
                opacity=0.8,
            )
        )

    colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350").tolist()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.3,
            yaxis="y2",
            showlegend=False,
        )
    )

    fig.update_layout(
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Volume", side="right", overlaying="y", showgrid=False),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=50, t=30, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangebreaks=[
            dict(bounds=["sat", "mon"]),
        ],
    )
    return fig


def tradingview_embed_html(safe_ticker: str, interval: str, *, height: int = 740) -> str:
    """Return the TradingView advanced-chart embed HTML for a sanitized ticker.

    ``safe_ticker`` must already be stripped of characters that could break the
    surrounding ``<script>`` block (see app.py ``_TICKER_SAFE_RE``).
    """
    tv_config: dict[str, Any] = {
        "allow_symbol_change": True,
        "calendar": False,
        "details": False,
        "hide_side_toolbar": True,
        "hide_top_toolbar": False,
        "hide_legend": False,
        "hide_volume": False,
        "hotlist": False,
        "interval": tv_interval(interval),
        "locale": "en",
        "save_image": True,
        "style": "1",
        "symbol": safe_ticker,
        "theme": "dark",
        "timezone": "Etc/UTC",
        "backgroundColor": "#0F0F0F",
        "gridColor": "rgba(242, 242, 242, 0.06)",
        "watchlist": [],
        "withdateranges": False,
        "compareSymbols": [],
        "studies": [],
        "width": "100%",
        "height": 700,
    }
    widget_div = (
        '  <div class="tradingview-widget-container__widget"'
        ' style="height:calc(100% - 32px);width:100%"></div>'
    )
    copyright_div = (
        f'  <div class="tradingview-widget-copyright">'
        f'<a href="https://www.tradingview.com/symbols/{safe_ticker}/"'
        f' rel="noopener nofollow" target="_blank">'
        f'<span class="blue-text">{safe_ticker} chart</span></a>'
        '<span class="trademark"> by TradingView</span></div>'
    )
    script_src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
    return "\n".join(
        [
            f'<div class="tradingview-widget-container" style="height:{height}px;width:100%">',
            widget_div,
            copyright_div,
            f'  <script type="text/javascript" src="{script_src}" async>',
            f"  {json.dumps(tv_config)}",
            "  </script>",
            "</div>",
        ]
    )
