"""Tests for commentator.llama_loader — shared GGUF load helper."""

from unittest.mock import MagicMock, patch

import pytest

from commentator.llama_loader import LazyLlama, create_llama


def _kwargs(**overrides):
    base = {
        "repo_id": "org/model",
        "filename": "model.gguf",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_batch": 512,
        "n_ubatch": 512,
        "main_gpu": 0,
        "tensor_split": [1.0],
        "flash_attn": True,
        "verbose": False,
        "label": "Test GGUF",
    }
    base.update(overrides)
    return base


def test_create_llama_downloads_and_constructs() -> None:
    fake = MagicMock(name="Llama")
    with (
        patch("commentator.llama_loader.Llama", fake),
        patch("commentator.llama_loader.hf_hub_download", return_value="/tmp/m.gguf") as dl,
    ):
        model = create_llama(**_kwargs())
    dl.assert_called_once_with(repo_id="org/model", filename="model.gguf")
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "/tmp/m.gguf"
    assert kwargs["flash_attn"] is True
    assert model is fake.return_value


def test_create_llama_retries_without_flash_attn() -> None:
    fake = MagicMock(name="Llama")
    loaded = MagicMock(name="loaded")
    fake.side_effect = [RuntimeError("no FA"), loaded]
    with (
        patch("commentator.llama_loader.Llama", fake),
        patch("commentator.llama_loader.hf_hub_download", return_value="/tmp/m.gguf"),
    ):
        model = create_llama(**_kwargs(flash_attn=True))
    assert fake.call_count == 2
    assert fake.call_args_list[0].kwargs["flash_attn"] is True
    assert fake.call_args_list[1].kwargs["flash_attn"] is False
    assert model is loaded


def test_create_llama_no_retry_when_flash_attn_disabled() -> None:
    fake = MagicMock(name="Llama", side_effect=RuntimeError("boom"))
    with (
        patch("commentator.llama_loader.Llama", fake),
        patch("commentator.llama_loader.hf_hub_download", return_value="/tmp/m.gguf"),
        pytest.raises(RuntimeError, match="boom"),
    ):
        create_llama(**_kwargs(flash_attn=False))
    assert fake.call_count == 1


def test_create_llama_missing_llama_cpp_raises() -> None:
    with (
        patch("commentator.llama_loader.Llama", None),
        pytest.raises(RuntimeError, match="llama_cpp is not installed"),
    ):
        create_llama(**_kwargs())


def test_lazy_llama_loads_once_and_warms() -> None:
    loaded = MagicMock(name="model")
    warmup = MagicMock()
    with patch("commentator.llama_loader.create_llama", return_value=loaded) as create:
        lazy = LazyLlama(warmup=warmup, **_kwargs())
        assert lazy.get() is loaded
        assert lazy.get() is loaded  # second call hits cache
    create.assert_called_once()
    warmup.assert_called_once_with(loaded)


def test_lazy_llama_warmup_failure_is_nonfatal() -> None:
    loaded = MagicMock(name="model")
    warmup = MagicMock(side_effect=RuntimeError("warmup boom"))
    with patch("commentator.llama_loader.create_llama", return_value=loaded):
        lazy = LazyLlama(warmup=warmup, **_kwargs())
        assert lazy.get() is loaded


def test_lazy_llama_reset_allows_reload() -> None:
    m1, m2 = MagicMock(name="m1"), MagicMock(name="m2")
    with patch("commentator.llama_loader.create_llama", side_effect=[m1, m2]) as create:
        lazy = LazyLlama(**_kwargs())
        assert lazy.get() is m1
        lazy.reset()
        assert lazy.get() is m2
    assert create.call_count == 2


def test_lazy_llama_missing_llama_cpp_raises() -> None:
    with (
        patch("commentator.llama_loader.Llama", None),
        pytest.raises(RuntimeError, match="llama_cpp is not installed"),
    ):
        LazyLlama(**_kwargs()).get()
