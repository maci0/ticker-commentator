"""Tests for commentator.data — fetch helpers and ticker validation."""

import pandas as pd
import pytest

from commentator import data


class _FakeTicker:
    def __init__(self, history_df: pd.DataFrame, info: dict | None = None):
        self._history_df = history_df
        self.info = info or {}
        self.history_calls: list[tuple[str, str]] = []

    def history(self, period: str, interval: str) -> pd.DataFrame:
        self.history_calls.append((period, interval))
        return self._history_df


# ── Ticker validation ───────────────────────────────────────────────


def test_empty_ticker_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        data.fetch_stock_data("")


def test_whitespace_ticker_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        data.fetch_stock_data("   ")


def test_invalid_ticker_raises() -> None:
    with pytest.raises(ValueError, match="Invalid ticker"):
        data.fetch_stock_data("<script>alert(1)</script>")


def test_valid_ticker_formats() -> None:
    """Various valid ticker formats should not raise during validation."""
    # We test the internal validator via fetch_stock_data with a monkeypatched yf
    # These would normally hit the network, so we just test they pass validation.
    for ticker in ["AAPL", "BRK.B", "^GSPC", "BTC-USD"]:
        cleaned = data._validate_ticker(ticker)
        assert cleaned == ticker.upper()


# ── 15-minute window ────────────────────────────────────────────────


def test_fetch_stock_data_15m_uses_1d_1m_and_trims_window(monkeypatch) -> None:
    index = pd.date_range("2026-01-01 10:00:00", periods=20, freq="min")
    history_df = pd.DataFrame(
        {
            "Open": range(20),
            "High": range(20),
            "Low": range(20),
            "Close": range(20),
            "Volume": [100] * 20,
        },
        index=index,
    )
    fake_ticker = _FakeTicker(history_df)

    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_data("AAPL", period="15m", interval="15m")

    assert fake_ticker.history_calls == [("1d", "1m")]
    assert len(result) == 16
    assert result.index.min() == index[-1] - pd.Timedelta(minutes=15)
    assert result.index.max() == index[-1]


# ── Non-15m period pass-through ─────────────────────────────────────


def test_fetch_stock_data_non_15m_passes_through(monkeypatch) -> None:
    """Non-15m periods should pass period and interval unchanged."""
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    history_df = pd.DataFrame(
        {
            "Open": range(5),
            "High": range(5),
            "Low": range(5),
            "Close": range(5),
            "Volume": [100] * 5,
        },
        index=index,
    )
    fake_ticker = _FakeTicker(history_df)
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_data("AAPL", period="5d", interval="15m")

    assert fake_ticker.history_calls == [("5d", "15m")]
    assert len(result) == 5


# ── Empty history ───────────────────────────────────────────────────


def test_fetch_stock_data_empty_history(monkeypatch) -> None:
    fake_ticker = _FakeTicker(pd.DataFrame())
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_data("AAPL")
    assert result.empty


# ── fetch_stock_info ────────────────────────────────────────────────


def test_fetch_stock_info_returns_name(monkeypatch) -> None:
    fake_ticker = _FakeTicker(pd.DataFrame(), info={"shortName": "Apple Inc."})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    assert data.fetch_stock_info("AAPL") == {"name": "Apple Inc."}


def test_fetch_stock_info_returns_ticker_on_failure(monkeypatch) -> None:
    def _boom(_ticker: str):
        raise RuntimeError("network down")

    monkeypatch.setattr(data.yf, "Ticker", _boom)

    assert data.fetch_stock_info("MSFT") == {"name": "MSFT"}


def test_fetch_stock_info_falls_back_to_ticker(monkeypatch) -> None:
    """When shortName is missing, should use ticker as name."""
    fake_ticker = _FakeTicker(pd.DataFrame(), info={})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    assert data.fetch_stock_info("GOOG") == {"name": "GOOG"}
