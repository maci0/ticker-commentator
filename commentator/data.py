import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str, period: str = "1d", interval: str = "1m"
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. Returns empty DataFrame on failure."""
    try:
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
    except Exception:
        logger.warning("Failed to fetch stock data for %s", ticker, exc_info=True)
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> dict:
    """Fetch basic company info."""
    try:
        t = yf.Ticker(ticker)
        return {"name": t.info.get("shortName", ticker)}
    except Exception:
        logger.warning("Failed to fetch stock info for %s", ticker, exc_info=True)
        return {"name": ticker}
