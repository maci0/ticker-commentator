"""Technical analysis of OHLCV stock data."""

import math
from typing import TypedDict

import pandas as pd


_REQUIRED_COLUMNS = frozenset({"Open", "High", "Low", "Close", "Volume"})


class AnalysisError(TypedDict):
    error: str


class AnalysisResult(TypedDict):
    trend: str
    price_change_pct: float
    current_price: float
    open_price: float
    high: float
    low: float
    volume_trend: str
    sma_cross: str | None
    volatility: str
    rsi: float | None


def analyze_stock(df: pd.DataFrame) -> AnalysisResult | AnalysisError:
    """Analyze stock DataFrame and return technical indicators.

    Returns an AnalysisError dict if the data is insufficient or malformed.
    Every numeric path guards against NaN and division-by-zero so callers
    never receive silent garbage.
    """
    if df.empty or len(df) < 2:
        return AnalysisError(error="Not enough data")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return AnalysisError(error=f"Missing columns: {', '.join(sorted(missing))}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    current_price = float(close.iloc[-1])
    first_close = float(close.iloc[0])

    # Guard: zero or NaN opening price makes percentage meaningless.
    if first_close == 0 or math.isnan(first_close) or math.isnan(current_price):
        price_change = 0.0
        price_change_pct = 0.0
    else:
        price_change = current_price - first_close
        price_change_pct = (price_change / first_close) * 100

    # Trend classification
    if price_change_pct > 1:
        trend = "bullish"
    elif price_change_pct < -1:
        trend = "bearish"
    else:
        trend = "sideways"

    # Volume trend: compare last-5 average to first-5 average.
    if len(volume) >= 10:
        recent_vol = float(volume.iloc[-5:].mean())
        earlier_vol = float(volume.iloc[:5].mean())
        if math.isnan(recent_vol) or math.isnan(earlier_vol) or earlier_vol <= 0:
            volume_trend = "normal"
        else:
            vol_ratio = recent_vol / earlier_vol
            if vol_ratio > 1.5:
                volume_trend = "heavy"
            elif vol_ratio < 0.5:
                volume_trend = "light"
            else:
                volume_trend = "normal"
    else:
        volume_trend = "normal"

    # SMA crossover (20/50) — requires 50+ data points.
    sma_cross: str | None = None
    if len(close) >= 50:
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        cur_20 = float(sma_20.iloc[-1])
        cur_50 = float(sma_50.iloc[-1])
        prev_20 = float(sma_20.iloc[-2])
        prev_50 = float(sma_50.iloc[-2])
        if not any(math.isnan(v) for v in (cur_20, cur_50, prev_20, prev_50)):
            if cur_20 > cur_50 and prev_20 <= prev_50:
                sma_cross = "golden_cross"
            elif cur_20 < cur_50 and prev_20 >= prev_50:
                sma_cross = "death_cross"

    # RSI (14-period) — requires 15+ data points.
    rsi: float | None = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        last_gain = float(gain.iloc[-1])
        last_loss = float(loss.iloc[-1])
        if math.isnan(last_gain) or math.isnan(last_loss):
            rsi = None
        elif last_loss == 0:
            rsi = 100.0
        else:
            rs = last_gain / last_loss
            rsi = 100 - (100 / (1 + rs))

    # Volatility via ATR (14-period).
    if len(df) >= 14:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_val = float(tr.rolling(14).mean().iloc[-1])
        if math.isnan(atr_val) or current_price <= 0:
            volatility = "unknown"
        else:
            atr_pct = (atr_val / current_price) * 100
            if atr_pct > 2:
                volatility = "high"
            elif atr_pct > 0.8:
                volatility = "medium"
            else:
                volatility = "low"
    else:
        volatility = "unknown"

    return AnalysisResult(
        trend=trend,
        price_change_pct=round(price_change_pct, 2),
        current_price=round(current_price, 2),
        open_price=round(first_close, 2),
        high=round(float(high.max()), 2),
        low=round(float(low.min()), 2),
        volume_trend=volume_trend,
        sma_cross=sma_cross,
        volatility=volatility,
        rsi=round(rsi, 1) if rsi is not None else None,
    )
