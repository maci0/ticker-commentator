"""Tests for the live-mode background prefetch store."""

import time

from commentator import prefetch


def _wait_idle(timeout: float = 2.0) -> None:
    """Wait until no prefetch thread is running (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = prefetch._THREAD
        if t is None or not t.is_alive():
            return
        time.sleep(0.01)


def setup_function() -> None:
    _wait_idle()
    prefetch.reset()


def test_take_returns_none_when_empty() -> None:
    assert prefetch.take(("AAPL", 100.0)) is None


def test_start_then_take_returns_result() -> None:
    prefetch.start(("AAPL", 100.0), lambda: "audio-result")
    _wait_idle()
    assert prefetch.take(("AAPL", 100.0)) == "audio-result"


def test_take_consumes_result() -> None:
    prefetch.start(("AAPL", 100.0), lambda: "x")
    _wait_idle()
    assert prefetch.take(("AAPL", 100.0)) == "x"
    # second take of same key returns None (consumed)
    assert prefetch.take(("AAPL", 100.0)) is None


def test_take_wrong_key_returns_none() -> None:
    prefetch.start(("AAPL", 100.0), lambda: "x")
    _wait_idle()
    assert prefetch.take(("AAPL", 101.0)) is None  # key mismatch (price moved)


def test_worker_exception_yields_none_not_crash() -> None:
    def boom() -> str:
        raise RuntimeError("worker blew up")

    prefetch.start(("AAPL", 100.0), boom)
    _wait_idle()
    # key matches but result is None (worker failed) -> take returns None safely
    assert prefetch.take(("AAPL", 100.0)) is None


def test_does_not_pile_up_threads() -> None:
    """A second start() while one is running must not spawn a competing thread."""

    def slow() -> str:
        time.sleep(0.2)
        return "first"

    prefetch.start(("AAPL", 100.0), slow)
    first_thread = prefetch._THREAD
    prefetch.start(("AAPL", 100.0), lambda: "second")  # should be ignored
    assert prefetch._THREAD is first_thread
    _wait_idle()
    assert prefetch.take(("AAPL", 100.0)) == "first"


def test_reset_clears_ready_result() -> None:
    prefetch.start(("AAPL", 100.0), lambda: "x")
    _wait_idle()
    prefetch.reset()
    assert prefetch.take(("AAPL", 100.0)) is None
