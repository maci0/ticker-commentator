"""Shared helpers for reading typed environment variables."""

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def env_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to default on invalid input."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("%s has an invalid value; using default %d", name, default)
        return default


def env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to default on invalid input."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("%s has an invalid value; using default %g", name, default)
        return default


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var.

    Truthy: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Falsy: ``0``, ``false``, ``no``, ``off``, empty string.
    Unset uses ``default``; any other value logs a warning and uses ``default``.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    logger.warning("%s has an invalid boolean value %r; using default %s", name, raw, default)
    return default


def env_abs_float(name: str, default: float) -> float:
    """Parse a float env var and return its absolute value (falls back on invalid)."""
    return abs(env_float(name, default))


def parse_tensor_split(env_var: str) -> list[float]:
    """Parse a comma-separated float list from an env var (e.g. '0.6,0.4').

    Returns [1.0] when the variable is not set.
    Invalid items are skipped with a warning; empty string returns [].
    """
    result: list[float] = []
    for s in os.getenv(env_var, "1.0").split(","):
        if s.strip():
            try:
                result.append(float(s.strip()))
            except ValueError:
                logger.warning("%s has invalid item %r; skipping", env_var, s.strip())
    return result
