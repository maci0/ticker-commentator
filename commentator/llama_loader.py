"""Shared lazy GGUF loading via llama-cpp-python.

Commentary and Orpheus TTS each hold a separate Llama instance but share the
same load path (download, flash-attn fallback, optional warmup) and the
process-wide LLAMA_CPP_LOCK so only one model runs inference at a time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from huggingface_hub import hf_hub_download

from commentator.llama_lock import LLAMA_CPP_LOCK

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - exercised via tests with Llama=None
    Llama = None

logger = logging.getLogger(__name__)

WarmupFn = Callable[[Any], None]


def create_llama(
    *,
    repo_id: str,
    filename: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_batch: int,
    n_ubatch: int,
    main_gpu: int,
    tensor_split: list[float],
    flash_attn: bool,
    verbose: bool,
    label: str = "GGUF",
) -> Any:
    """Download (if needed) and construct a Llama instance.

    Retries once without flash attention when the first load fails and
    ``flash_attn`` was requested — some llama.cpp builds lack FA kernels.
    """
    if Llama is None:
        raise RuntimeError("llama_cpp is not installed")

    gguf_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(
        "Loading %s: %s (ctx=%d, gpu_layers=%d, main_gpu=%d)",
        label,
        gguf_path,
        n_ctx,
        n_gpu_layers,
        main_gpu,
    )

    def _load(use_flash_attn: bool) -> Any:
        return Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attn=use_flash_attn,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
            verbose=verbose,
        )

    try:
        return _load(flash_attn)
    except Exception:
        if not flash_attn:
            raise
        logger.warning(
            "%s load with flash_attn failed; retrying without it",
            label,
            exc_info=True,
        )
        return _load(False)


class LazyLlama:
    """Thread-safe lazy singleton around :func:`create_llama`.

    Double-checked locking uses the shared ``LLAMA_CPP_LOCK`` so concurrent
    first-callers serialize with other llama.cpp work in the process.
    """

    def __init__(
        self,
        *,
        label: str,
        repo_id: str,
        filename: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_batch: int,
        n_ubatch: int,
        main_gpu: int,
        tensor_split: list[float],
        flash_attn: bool,
        verbose: bool,
        warmup: WarmupFn | None = None,
    ) -> None:
        self._label = label
        self._warmup = warmup
        self._load_kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "main_gpu": main_gpu,
            "tensor_split": tensor_split,
            "flash_attn": flash_attn,
            "verbose": verbose,
            "label": label,
        }
        self._model: Any | None = None

    def get(self) -> Any:
        """Return the cached model, loading it on first call."""
        if self._model is not None:
            return self._model
        if Llama is None:
            raise RuntimeError("llama_cpp is not installed")
        with LLAMA_CPP_LOCK:
            if self._model is not None:
                return self._model
            t0 = time.time()
            model = create_llama(**self._load_kwargs)
            logger.info("%s loaded in %.1fs", self._label, time.time() - t0)
            if self._warmup is not None:
                try:
                    tw = time.time()
                    self._warmup(model)
                    logger.info("%s warmup done in %.2fs", self._label, time.time() - tw)
                except Exception:
                    logger.warning(
                        "%s warmup failed (non-fatal)",
                        self._label,
                        exc_info=True,
                    )
            self._model = model
        return self._model

    def reset(self) -> None:
        """Drop the cached model. Intended for tests."""
        self._model = None
