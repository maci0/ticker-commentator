"""Technical analysis of OHLCV stock data."""

import logging
import math
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["analyze_stock", "AnalysisResult", "AnalysisError"]

_REQUIRED_COLUMNS = frozenset({"Open", "High", "Low", "Close", "Volume"})

_TREND_THRESHOLD_PCT = 1.0  # % change to classify as bullish/bearish
_VOLUME_HEAVY_RATIO = 1.5  # recent/earlier vol ratio to be "heavy"
_VOLUME_LIGHT_RATIO = 0.5  # recent/earlier vol ratio to be "light"
_ATR_HIGH_PCT = 2.0  # ATR/price % above which volatility is "high"
_ATR_MEDIUM_PCT = 0.8  # ATR/price % above which volatility is "medium"


def _safe_round(value: float, ndigits: int) -> float:
    """Round value, substituting 0.0 for any non-finite input (NaN or ±inf) to
    prevent silent garbage propagating into results."""
    return round(value if math.isfinite(value) else 0.0, ndigits)


class AnalysisError(TypedDict):
    """Returned by analyze_stock when input data is insufficient or malformed."""

    error: str


class AnalysisResult(TypedDict):
    """Technical indicators computed from OHLCV data.

    trend: 'bullish', 'bearish', or 'sideways'
    price_change_pct: period change from first close to last close as a percentage
    current_price: latest close price in the window
    open_price: first open price in the window (period open); 0.0 if NaN
    high: highest high across the window; 0.0 if NaN
    low: lowest low across the window; 0.0 if NaN
    volume_trend: 'heavy', 'light', or 'normal'
    volatility: 'high', 'medium', 'low', or 'unknown'
    sma_cross: 'golden_cross', 'death_cross', or None if no recent crossover detected
    rsi: RSI value in [0.0, 100.0], or None if fewer than 15 data points,
         or when the price series is flat (zero gain and zero loss)
    """

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
        logger.warning("analyze_stock: insufficient data (rows=%d)", len(df))
        return AnalysisError(error="Not enough data")

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.warning("analyze_stock: missing columns %s", sorted(missing))
        return AnalysisError(error=f"Missing columns: {', '.join(sorted(missing))}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    current_price = float(close.iloc[-1])
    # Neutralize NaN current_price early so downstream numeric paths (ATR, return
    # fields) never receive NaN.  Zero is an impossible real price and will cause
    # the ATR guard (`current_price <= 0`) to classify volatility as "unknown",
    # which is the correct behaviour when the latest close is missing.
    if math.isnan(current_price):
        logger.warning("analyze_stock: NaN current price; defaulting to 0.0")
        current_price = 0.0
    first_close = float(close.iloc[0])

    # Guard: zero or NaN first close, or zero current_price (substituted from NaN),
    # makes price-change percentage meaningless.  Without the current_price guard,
    # a missing last close would produce a false -100% bearish reading.
    if first_close == 0 or math.isnan(first_close) or current_price == 0.0:
        price_change = 0.0
        price_change_pct = 0.0
    else:
        price_change = current_price - first_close
        price_change_pct = (price_change / first_close) * 100
        # A denormal/near-zero first_close (e.g. 5e-324) slips past the `== 0`
        # guard above and makes the ratio non-finite; treat that as no change.
        if not math.isfinite(price_change_pct):
            price_change = 0.0
            price_change_pct = 0.0

    if price_change_pct > _TREND_THRESHOLD_PCT:
        trend = "bullish"
    elif price_change_pct < -_TREND_THRESHOLD_PCT:
        trend = "bearish"
    else:
        trend = "sideways"

    if len(volume) >= 10:
        recent_vol = float(volume.iloc[-5:].mean())
        earlier_vol = float(volume.iloc[:5].mean())
        if math.isnan(recent_vol) or math.isnan(earlier_vol) or earlier_vol <= 0:
            volume_trend = "normal"
        else:
            vol_ratio = recent_vol / earlier_vol
            if vol_ratio > _VOLUME_HEAVY_RATIO:
                volume_trend = "heavy"
            elif vol_ratio < _VOLUME_LIGHT_RATIO:
                volume_trend = "light"
            else:
                volume_trend = "normal"
    else:
        volume_trend = "normal"

    # SMA crossover (20/50) — requires 51+ data points to detect a cross.
    # With exactly 50 rows, prev_50 is NaN and the NaN check below skips detection.
    # Slice to the last 51 rows for efficiency; rolling(50) on the full series is wasteful
    # since we only need the last two SMA values.
    sma_cross: str | None = None
    if len(close) >= 50:
        tail = close.iloc[-51:]
        sma_20 = tail.rolling(20).mean()
        sma_50 = tail.rolling(50).mean()
        cur_20 = float(sma_20.iloc[-1])
        cur_50 = float(sma_50.iloc[-1])
        prev_20 = float(sma_20.iloc[-2])
        prev_50 = float(sma_50.iloc[-2])
        if not any(math.isnan(v) for v in (cur_20, cur_50, prev_20, prev_50)):
            if cur_20 > cur_50 and prev_20 <= prev_50:
                sma_cross = "golden_cross"
            elif cur_20 < cur_50 and prev_20 >= prev_50:
                sma_cross = "death_cross"

    # RSI (14-period, SMA-smoothed) — requires 15+ data points.
    # Uses a simple rolling mean rather than Wilder's EMA; sufficient for
    # trend commentary but diverges from the traditional RSI formula.
    # Slice to last 15 rows: exactly enough for one rolling(14) window.
    rsi: float | None = None
    if len(close) >= 15:
        tail_rsi = close.iloc[-15:]
        delta = tail_rsi.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        last_gain = float(gain.iloc[-1])
        last_loss = float(loss.iloc[-1])
        if math.isnan(last_gain) or math.isnan(last_loss):
            rsi = None
        elif last_loss == 0:
            rsi = 100.0 if last_gain > 0 else None  # 100 for all-gain periods, None for flat prices
        else:
            rs = last_gain / last_loss
            rsi = 100 - (100 / (1 + rs))

    # Volatility via ATR (14-period) — requires 15+ data points in practice;
    # 14 rows enters this branch but the first TR is NaN (no prior close), so
    # np.mean(tr_vals[-14:]) is NaN and volatility falls back to 'unknown'.
    if len(df) >= 14:
        # Slice to the last 15 rows: each TR uses current + prior close, so 15
        # rows yield 14 usable TR values. Computing on the full DataFrame wastes
        # work on rows discarded when taking tr_vals[-14:].
        _atr_tail = df.iloc[-15:]
        _atr_close = _atr_tail["Close"]
        _atr_high = _atr_tail["High"]
        _atr_low = _atr_tail["Low"]
        prev_close = _atr_close.shift()
        tr_vals = np.maximum(
            np.maximum(
                (_atr_high - _atr_low).to_numpy(),
                np.abs((_atr_high - prev_close).to_numpy()),
            ),
            np.abs((_atr_low - prev_close).to_numpy()),
        )
        # np.mean of the last 14 true ranges is equivalent to rolling(14).mean().iloc[-1]
        # and avoids constructing a temporary pd.Series.
        atr_val = float(np.mean(tr_vals[-14:]))
        if math.isnan(atr_val) or current_price <= 0:
            volatility = "unknown"
        else:
            atr_pct = (atr_val / current_price) * 100
            if atr_pct > _ATR_HIGH_PCT:
                volatility = "high"
            elif atr_pct > _ATR_MEDIUM_PCT:
                volatility = "medium"
            else:
                volatility = "low"
    else:
        volatility = "unknown"

    result = AnalysisResult(
        trend=trend,
        price_change_pct=round(price_change_pct, 2),
        current_price=round(current_price, 2),
        open_price=_safe_round(float(df["Open"].iloc[0]), 2),
        high=_safe_round(float(high.max()), 2),
        low=_safe_round(float(low.min()), 2),
        volume_trend=volume_trend,
        sma_cross=sma_cross,
        volatility=volatility,
        rsi=round(rsi, 1) if rsi is not None else None,
    )
    logger.debug(
        "analysis_result trend=%s price=%.2f change_pct=%+.2f rsi=%s "
        "volatility=%s sma_cross=%s rows=%d",
        result["trend"],
        result["current_price"],
        result["price_change_pct"],
        result["rsi"],
        result["volatility"],
        result["sma_cross"],
        len(df),
    )
    return result
