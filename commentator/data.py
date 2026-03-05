"""Fetch OHLCV and company info from Yahoo Finance."""

import logging
import re

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Matches standard tickers: AAPL, BRK.B, ^GSPC, ES=F, BTC-USD, etc.
_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,19}$")


def _validate_ticker(ticker: str) -> str:
    """Normalize and validate a ticker symbol.

    Raises ValueError for empty, whitespace-only, or malformed tickers.
    This also prevents injection of arbitrary strings into yfinance or
    downstream HTML embeds.
    """
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker symbol must not be empty")
    if not _TICKER_RE.match(cleaned):
        raise ValueError(
            f"Invalid ticker symbol: {cleaned!r} — "
            "expected letters, digits, '.', '^', '=', or '-'"
        )
    return cleaned


def fetch_stock_data(
    ticker: str, period: str = "1d", interval: str = "1m"
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. Returns empty DataFrame on failure."""
    try:
        ticker = _validate_ticker(ticker)
        t = yf.Ticker(ticker)
        request_period = "1d" if period == "15m" else period
        request_interval = "1m" if period == "15m" else interval
        df = t.history(period=request_period, interval=request_interval)
        if df.empty:
            return pd.DataFrame()
        if period == "15m":
            last_ts = df.index.max()
            window_start = last_ts - pd.Timedelta(minutes=15)
            df = df[df.index >= window_start]
        return df
    except ValueError:
        raise  # Re-raise validation errors without swallowing them.
    except Exception:
        logger.warning("Failed to fetch stock data for %s", ticker, exc_info=True)
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> dict[str, str]:
    """Fetch basic company info. Returns ticker as name on failure."""
    try:
        ticker = _validate_ticker(ticker)
        t = yf.Ticker(ticker)
        return {"name": t.info.get("shortName", ticker)}
    except ValueError:
        raise
    except Exception:
        logger.warning("Failed to fetch stock info for %s", ticker, exc_info=True)
        return {"name": ticker}
