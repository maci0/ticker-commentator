import pandas as pd

from commentator import data


class _FakeTicker:
    def __init__(self, history_df: pd.DataFrame, info: dict | None = None):
        self._history_df = history_df
        self.info = info or {}
        self.history_calls: list[tuple[str, str]] = []

    def history(self, period: str, interval: str) -> pd.DataFrame:
        self.history_calls.append((period, interval))
        return self._history_df


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


def test_fetch_stock_info_returns_ticker_on_failure(monkeypatch) -> None:
    def _boom(_ticker: str):
        raise RuntimeError("network down")

    monkeypatch.setattr(data.yf, "Ticker", _boom)

    assert data.fetch_stock_info("MSFT") == {"name": "MSFT"}
