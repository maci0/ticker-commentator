import pandas as pd


def analyze_stock(df: pd.DataFrame) -> dict:
    """Analyze stock DataFrame and return a dict of technical indicators."""
    if df.empty or len(df) < 2:
        return {"error": "Not enough data"}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    current_price = float(close.iloc[-1])
    first_close = float(close.iloc[0])
    price_change = current_price - first_close
    price_change_pct = (price_change / first_close) * 100

    # Trend
    if price_change_pct > 1:
        trend = "bullish"
    elif price_change_pct < -1:
        trend = "bearish"
    else:
        trend = "sideways"

    # Volume trend
    if len(volume) >= 10:
        recent_vol = volume.iloc[-5:].mean()
        earlier_vol = volume.iloc[:5].mean()
        if earlier_vol > 0:
            vol_ratio = recent_vol / earlier_vol
            if vol_ratio > 1.5:
                volume_trend = "heavy"
            elif vol_ratio < 0.5:
                volume_trend = "light"
            else:
                volume_trend = "normal"
        else:
            volume_trend = "normal"
    else:
        volume_trend = "normal"

    # SMA crossover (20/50 if enough data)
    sma_cross = None
    if len(close) >= 50:
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
            sma_cross = "golden_cross"
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
            sma_cross = "death_cross"

    # RSI (14-period)
    rsi = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        last_loss = float(loss.iloc[-1])
        if last_loss != 0:
            rs = float(gain.iloc[-1]) / last_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100.0

    # Volatility via ATR
    if len(df) >= 14:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr / current_price) * 100
        if atr_pct > 2:
            volatility = "high"
        elif atr_pct > 0.8:
            volatility = "medium"
        else:
            volatility = "low"
    else:
        volatility = "unknown"

    return {
        "trend": trend,
        "price_change_pct": round(price_change_pct, 2),
        "current_price": round(current_price, 2),
        "open_price": round(first_close, 2),
        "high": round(float(high.max()), 2),
        "low": round(float(low.min()), 2),
        "volume_trend": volume_trend,
        "sma_cross": sma_cross,
        "volatility": volatility,
        "rsi": round(rsi, 1) if rsi is not None else None,
    }
