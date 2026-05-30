"""Shared helpers for reading typed environment variables."""

import logging
import os

logger = logging.getLogger(__name__)


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
