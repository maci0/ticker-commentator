"""Optional alternative TTS engines, selected via TTS_ENGINE.

These are NOT installed by default: each pins torch/transformers versions that
conflict with the project's (and with each other), so install the one you want
into its own environment (see docs/tts_engines.md) and run the app from there.

Every adapter yields 16-bit mono PCM at SAMPLE_RATE (24 kHz), matching the
Orpheus path, so commentator.tts.pcm_chunks_to_wav consumes the output
unchanged. Unlike Orpheus these models synthesize the whole clip at once, so the
adapter yields a single chunk (no token-level streaming).

Benchmarked on the project's RX 7900 XTX (see tts_eval/): naturalness UTMOS —
chatterbox 4.37, kokoro 4.40, orpheus 4.26; speed (warm RTF, lower = faster) —
orpheus 0.62, kokoro 0.20 (CPU), chatterbox 1.51. Chatterbox is the most
expressive but slower than real time; kokoro is fastest but emotionally flat.
"""

import logging
import os
from collections.abc import Generator

import numpy as np

logger = logging.getLogger(__name__)

# All supported engines output 24 kHz; kept equal to commentator.tts.SAMPLE_RATE.
SAMPLE_RATE = 24000

_CHATTERBOX_MODEL = None
_KOKORO_PIPELINE = None


def _float_to_pcm16(audio: object) -> bytes:
    """Convert a float waveform (numpy array or torch tensor, [-1, 1]) to
    little-endian 16-bit mono PCM bytes."""
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).ravel()
    # nan_to_num before clip: clip leaves NaN as NaN, which casts to garbage int16.
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def iter_audio_chunks(
    engine: str, text: str, voice: str = "", speed: float = 1.3
) -> Generator[bytes, None, None]:
    """Route to the selected alternative engine. Raises RuntimeError with an
    install hint if the engine's package is not available."""
    if engine == "chatterbox":
        return _chatterbox_chunks(text)
    if engine == "kokoro":
        return _kokoro_chunks(text)
    raise ValueError(
        f"Unknown TTS_ENGINE {engine!r}; expected 'orpheus', 'chatterbox', or 'kokoro'"
    )


def _get_chatterbox() -> object:
    global _CHATTERBOX_MODEL
    if _CHATTERBOX_MODEL is not None:
        return _CHATTERBOX_MODEL
    try:
        import torch
        from chatterbox.tts import ChatterboxTTS
    except ImportError as exc:
        raise RuntimeError(
            "TTS_ENGINE=chatterbox needs the 'chatterbox-tts' package; its torch "
            "pin conflicts with the project, so install it in a separate env "
            "(see docs/tts_engines.md)."
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Chatterbox TTS on %s", device)
    _CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(device=device)
    return _CHATTERBOX_MODEL


def _chatterbox_chunks(text: str) -> Generator[bytes, None, None]:
    # exaggeration 0.8 / cfg 0.5 scored best in tts_eval; >1.0 hurt naturalness.
    exaggeration = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.8"))
    cfg = float(os.getenv("CHATTERBOX_CFG", "0.5"))
    model = _get_chatterbox()
    wav = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg)
    yield _float_to_pcm16(wav)


def _get_kokoro() -> object:
    global _KOKORO_PIPELINE
    if _KOKORO_PIPELINE is not None:
        return _KOKORO_PIPELINE
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise RuntimeError(
            "TTS_ENGINE=kokoro needs the 'kokoro' package; its transformers pin "
            "conflicts with the project, so install it in a separate env "
            "(see docs/tts_engines.md)."
        ) from exc
    logger.info("Loading Kokoro pipeline")
    _KOKORO_PIPELINE = KPipeline(lang_code=os.getenv("KOKORO_LANG", "a"))
    return _KOKORO_PIPELINE


def _kokoro_chunks(text: str) -> Generator[bytes, None, None]:
    voice = os.getenv("KOKORO_VOICE", "am_michael")
    pipe = _get_kokoro()
    for _gs, _ps, audio in pipe(text, voice=voice):
        yield _float_to_pcm16(audio)
