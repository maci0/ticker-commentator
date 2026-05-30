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
    for ticker in ["AAPL", "BRK.B", "^GSPC", "BTC-USD"]:
        cleaned = data._validate_ticker(ticker)
        assert cleaned == ticker.upper()


def test_valid_ticker_max_length() -> None:
    """A 20-character ticker is at the regex boundary and must be accepted."""
    ticker = "A" * 20
    assert data._validate_ticker(ticker) == ticker


def test_invalid_ticker_too_long() -> None:
    """A 21-character ticker exceeds the regex limit and must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        data._validate_ticker("A" * 21)


def test_valid_ticker_single_char() -> None:
    """A single uppercase letter is a valid minimal ticker."""
    assert data._validate_ticker("S") == "S"


def test_valid_ticker_lowercase_normalizes_to_uppercase() -> None:
    """Lowercase input must be normalized to uppercase before regex validation."""
    assert data._validate_ticker("aapl") == "AAPL"
    assert data._validate_ticker("brk.b") == "BRK.B"


def test_valid_ticker_futures_format() -> None:
    """Futures tickers (e.g. ES=F) contain '=' and must be accepted."""
    assert data._validate_ticker("ES=F") == "ES=F"


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


def test_fetch_stock_info_invalid_ticker_raises() -> None:
    with pytest.raises(ValueError, match="Invalid ticker"):
        data.fetch_stock_info("<script>")


def test_fetch_stock_info_returns_name(monkeypatch) -> None:
    fake_ticker = _FakeTicker(pd.DataFrame(), info={"shortName": "Apple Inc."})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    assert data.fetch_stock_info("AAPL") == {"name": "Apple Inc."}


def test_fetch_stock_info_returns_ticker_on_failure(monkeypatch) -> None:
    def _boom(_ticker: str):
        raise RuntimeError("network down")

    monkeypatch.setattr(data.yf, "Ticker", _boom)

    assert data.fetch_stock_info("MSFT") == {"name": "MSFT"}


def test_fetch_stock_data_network_error_returns_empty(monkeypatch) -> None:
    """Non-ValueError exceptions from yfinance should be swallowed and return an empty DataFrame."""
    def _boom(_ticker: str):
        raise ConnectionError("network down")

    monkeypatch.setattr(data.yf, "Ticker", _boom)

    result = data.fetch_stock_data("AAPL")
    assert result.empty


def test_fetch_stock_info_falls_back_to_ticker(monkeypatch) -> None:
    """When shortName is missing, should use ticker as name."""
    fake_ticker = _FakeTicker(pd.DataFrame(), info={})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    assert data.fetch_stock_info("GOOG") == {"name": "GOOG"}


def test_fetch_stock_data_15m_empty_1d_returns_empty(monkeypatch) -> None:
    """When the underlying 1d@1m fetch returns empty, period='15m' must also return empty."""
    fake_ticker = _FakeTicker(pd.DataFrame())
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_data("AAPL", period="15m")
    assert result.empty
    assert fake_ticker.history_calls == [("1d", "1m")]


def test_fetch_stock_data_invalid_period_raises() -> None:
    """An unrecognised period string must raise ValueError before hitting the network."""
    with pytest.raises(ValueError, match="Invalid period"):
        data.fetch_stock_data("AAPL", period="invalid")


def test_fetch_stock_data_invalid_interval_raises() -> None:
    """An unrecognised interval string must raise ValueError before hitting the network."""
    with pytest.raises(ValueError, match="Invalid interval"):
        data.fetch_stock_data("AAPL", interval="invalid")


def test_fetch_stock_info_short_name_none_falls_back_to_ticker(monkeypatch) -> None:
    """When shortName key exists but its value is None, should fall back to ticker."""
    fake_ticker = _FakeTicker(pd.DataFrame(), info={"shortName": None})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    assert data.fetch_stock_info("TSLA") == {"name": "TSLA"}


def test_fetch_stock_info_sanitizes_control_chars(monkeypatch) -> None:
    """Control characters in company names must be replaced with spaces."""
    fake_ticker = _FakeTicker(pd.DataFrame(), info={"shortName": "Apple\x00Inc\n"})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_info("AAPL")
    assert "\x00" not in result["name"]
    assert "\n" not in result["name"]
    assert "Apple" in result["name"]
    assert "Inc" in result["name"]


def test_fetch_stock_info_truncates_long_name(monkeypatch) -> None:
    """Company names longer than 100 characters must be truncated to 100."""
    fake_ticker = _FakeTicker(pd.DataFrame(), info={"shortName": "A" * 150})
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_info("AAPL")
    assert len(result["name"]) == 100


def test_fetch_stock_info_sanitizes_unicode_newlines(monkeypatch) -> None:
    """Unicode newline-like characters must be stripped to prevent prompt injection."""
    # U+0085 NEL, U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR
    fake_ticker = _FakeTicker(
        pd.DataFrame(), info={"shortName": "Apple\u0085Inc\u2028Corp\u2029"}
    )
    monkeypatch.setattr(data.yf, "Ticker", lambda _: fake_ticker)

    result = data.fetch_stock_info("AAPL")
    assert "\u0085" not in result["name"]
    assert "\u2028" not in result["name"]
    assert "\u2029" not in result["name"]
    assert "Apple" in result["name"]
    assert "Inc" in result["name"]
    assert "Corp" in result["name"]
