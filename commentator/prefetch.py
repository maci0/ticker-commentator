"""Background speculative prefetch for live mode.

In live mode the Python process is idle while the current clip plays. With
LIVE_PREFETCH=1 this module spends that idle time generating the *next* update's
commentary + audio in a daemon thread, keyed by a snapshot of the data. When the
next refresh fires and the data is unchanged (a common interval-driven tick),
the result is served instantly instead of blocking ~4s on regeneration.

Thread-safety: results live in this module's globals, NOT Streamlit
session_state (which is not thread-safe and lacks a ScriptRunContext off the main
thread). The worker calls only pure commentator functions, which already
serialize llama.cpp access on LLAMA_CPP_LOCK. Disabled by default — when off the
app path is completely unchanged.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# key: identity of the data snapshot the result was generated for.
# result: whatever the worker returns (None on failure).
# busy_key: key currently being computed (None when idle).
# thread: the live worker thread, if any.
_KEY: Any = None
_RESULT: Any = None
_BUSY_KEY: Any = None
_THREAD: "threading.Thread | None" = None


def start(key: Any, worker: Callable[[], Any]) -> None:
    """Compute worker() in a daemon thread under `key`, unless a result for that
    key is already ready or a prefetch is already running. Never raises."""
    global _KEY, _RESULT, _BUSY_KEY, _THREAD
    with _LOCK:
        already_ready = _KEY == key and _RESULT is not None
        running = _THREAD is not None and _THREAD.is_alive()
        if already_ready or running:
            return
        _KEY = key
        _RESULT = None
        _BUSY_KEY = key

    def _run() -> None:
        global _RESULT, _BUSY_KEY
        result = None
        try:
            result = worker()
        except Exception:
            logger.exception("prefetch worker failed for key=%r", key)
        with _LOCK:
            # Only publish if this key is still the one in flight (not superseded).
            if _BUSY_KEY == key:
                _RESULT = result
                _BUSY_KEY = None

    t = threading.Thread(target=_run, name="live-prefetch", daemon=True)
    with _LOCK:
        _THREAD = t
    t.start()


def take(key: Any) -> Any:
    """Return and consume the prefetched result for `key` if ready, else None."""
    global _KEY, _RESULT
    with _LOCK:
        if _KEY == key and _RESULT is not None:
            result = _RESULT
            _RESULT = None
            _KEY = None
            return result
        return None


def reset() -> None:
    """Clear all prefetch state (does not stop an in-flight daemon thread; its
    result will simply be discarded by the busy_key check)."""
    global _KEY, _RESULT, _BUSY_KEY
    with _LOCK:
        _KEY = None
        _RESULT = None
        _BUSY_KEY = None
