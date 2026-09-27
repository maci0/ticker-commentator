"""Fetch OHLCV and company info from Yahoo Finance."""

import logging
import re
import time
from typing import TypedDict

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

__all__ = ["fetch_stock_data", "fetch_stock_info", "StockInfo"]


class StockInfo(TypedDict):
    """Company metadata returned by fetch_stock_info."""

    name: str


# Matches standard tickers: AAPL, BRK.B, ^GSPC, ES=F, BTC-USD, etc.
_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,19}$")

# Accepted period values for fetch_stock_data. "15m" is a synthetic window handled
# internally (fetched as "1d" at "1m" resolution, then sliced); all others are passed
# to yfinance. Rejects arbitrary strings that could confuse the library or downstream
# consumers.
_VALID_PERIODS = frozenset(
    {"15m", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
)
# Allowed interval values passed to yfinance.
_VALID_INTERVALS = frozenset(
    {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
)


def _validate_ticker(ticker: str) -> str:
    """Normalize and validate a ticker symbol.

    Raises ValueError for empty, whitespace-only, or malformed tickers.
    Tickers are limited to 20 characters.
    This also prevents injection of arbitrary strings into yfinance or
    downstream HTML embeds.
    """
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker symbol must not be empty")
    if not _TICKER_RE.match(cleaned):
        raise ValueError(
            f"Invalid ticker symbol: {cleaned!r} — expected letters, digits, '.', '^', '=', or '-'"
        )
    return cleaned


def fetch_stock_data(ticker: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. Returns empty DataFrame on failure.

    Raises ValueError for invalid ticker, period, or interval values (does not
    swallow those — other exceptions are logged and return an empty DataFrame).

    The "15m" period is a synthetic window: data is fetched as "1d" at "1m"
    resolution, then sliced to the last 15 minutes of available data.
    """
    if period not in _VALID_PERIODS:
        raise ValueError(f"Invalid period: {period!r}")
    if interval not in _VALID_INTERVALS:
        raise ValueError(f"Invalid interval: {interval!r}")
    try:
        ticker = _validate_ticker(ticker)
        t = yf.Ticker(ticker)
        request_period = "1d" if period == "15m" else period
        request_interval = "1m" if period == "15m" else interval
        t_fetch = time.time()
        df = t.history(period=request_period, interval=request_interval)
        if df.empty:
            logger.warning(
                "no_data ticker=%s period=%s interval=%s — market may be closed or symbol invalid",
                ticker,
                period,
                interval,
            )
            return pd.DataFrame()
        if period == "15m":
            last_ts = df.index.max()
            window_start = last_ts - pd.Timedelta(minutes=15)
            df = df[df.index >= window_start]
        logger.info(
            "fetch_complete ticker=%s rows=%d period=%s interval=%s elapsed=%.2fs",
            ticker,
            len(df),
            period,
            interval,
            time.time() - t_fetch,
        )
        return df
    except ValueError:
        raise
    except Exception:
        logger.warning("Failed to fetch stock data for %s", ticker, exc_info=True)
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> StockInfo:
    """Fetch basic company info. Returns ticker as name if the company name is
    unavailable (shortName missing, None, or empty) or on network failure.

    Raises ValueError for invalid ticker symbols (does not swallow those).
    The returned name has control characters replaced with spaces and is
    truncated to 100 characters to prevent prompt injection when embedded
    in LLM inputs.
    """
    try:
        ticker = _validate_ticker(ticker)
        t = yf.Ticker(ticker)
        t_fetch = time.time()
        name = str(t.info.get("shortName") or ticker)
        logger.info("fetch_info_complete ticker=%s elapsed=%.2fs", ticker, time.time() - t_fetch)
        # Strip ASCII control characters AND Unicode newline-like characters
        # (U+0085 NEL, U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR) to
        # prevent indirect prompt injection when this value is embedded in LLM prompts.
        name = re.sub(r"[\x00-\x1f\x7f\u0085\u2028\u2029]", " ", name).strip()[:100]
        return StockInfo(name=name or ticker)
    except ValueError:
        raise
    except Exception:
        logger.warning("Failed to fetch stock info for %s", ticker, exc_info=True)
        return StockInfo(name=ticker)
