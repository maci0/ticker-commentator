"""Tests for commentator._config — typed environment variable helpers."""

import pytest

from commentator._config import env_float, env_int, parse_tensor_split

# ── env_int ─────────────────────────────────────────────────────────


def test_env_int_returns_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("_TEST_ENV_INT", raising=False)
    assert env_int("_TEST_ENV_INT", 42) == 42


def test_env_int_returns_parsed_value(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_INT", "7")
    assert env_int("_TEST_ENV_INT", 42) == 7


def test_env_int_falls_back_on_non_integer_string(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_INT", "not_a_number")
    assert env_int("_TEST_ENV_INT", 42) == 42


def test_env_int_falls_back_on_float_string(monkeypatch) -> None:
    """int('1.5') raises ValueError; the default must be used."""
    monkeypatch.setenv("_TEST_ENV_INT", "1.5")
    assert env_int("_TEST_ENV_INT", 42) == 42


def test_env_int_negative_value(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_INT", "-5")
    assert env_int("_TEST_ENV_INT", 0) == -5


def test_env_int_zero(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_INT", "0")
    assert env_int("_TEST_ENV_INT", 99) == 0


# ── env_float ────────────────────────────────────────────────────────


def test_env_float_returns_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("_TEST_ENV_FLOAT", raising=False)
    assert env_float("_TEST_ENV_FLOAT", 3.14) == pytest.approx(3.14)


def test_env_float_returns_parsed_value(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_FLOAT", "2.71")
    assert env_float("_TEST_ENV_FLOAT", 0.0) == pytest.approx(2.71)


def test_env_float_falls_back_on_invalid(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_FLOAT", "bad")
    assert env_float("_TEST_ENV_FLOAT", 1.0) == pytest.approx(1.0)


def test_env_float_integer_string_is_valid(monkeypatch) -> None:
    """float('42') is valid; must return 42.0, not fall back to default."""
    monkeypatch.setenv("_TEST_ENV_FLOAT", "42")
    assert env_float("_TEST_ENV_FLOAT", 0.0) == pytest.approx(42.0)


def test_env_float_negative_value(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_ENV_FLOAT", "-0.5")
    assert env_float("_TEST_ENV_FLOAT", 0.0) == pytest.approx(-0.5)


# ── parse_tensor_split ───────────────────────────────────────────────


def test_parse_tensor_split_returns_default_when_unset(monkeypatch) -> None:
    """Unset variable uses the hardcoded default '1.0', producing [1.0]."""
    monkeypatch.delenv("_TEST_TENSOR", raising=False)
    assert parse_tensor_split("_TEST_TENSOR") == pytest.approx([1.0])


def test_parse_tensor_split_parses_two_values(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_TENSOR", "0.6,0.4")
    assert parse_tensor_split("_TEST_TENSOR") == pytest.approx([0.6, 0.4])


def test_parse_tensor_split_empty_string_returns_empty_list(monkeypatch) -> None:
    """Empty string is a valid env var value; must return [] not [1.0]."""
    monkeypatch.setenv("_TEST_TENSOR", "")
    assert parse_tensor_split("_TEST_TENSOR") == []


def test_parse_tensor_split_skips_invalid_items(monkeypatch) -> None:
    """Invalid entries are silently skipped; valid neighbours are kept."""
    monkeypatch.setenv("_TEST_TENSOR", "1.0,bad,2.0")
    assert parse_tensor_split("_TEST_TENSOR") == pytest.approx([1.0, 2.0])


def test_parse_tensor_split_single_value(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_TENSOR", "0.5")
    assert parse_tensor_split("_TEST_TENSOR") == pytest.approx([0.5])


def test_parse_tensor_split_strips_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("_TEST_TENSOR", " 0.3 , 0.7 ")
    assert parse_tensor_split("_TEST_TENSOR") == pytest.approx([0.3, 0.7])
