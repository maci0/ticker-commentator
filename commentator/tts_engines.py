"""Optional alternative TTS engines, selected via TTS_ENGINE.

These are NOT installed by default. `qwen` coexists in the main env (its
transformers pin doesn't clash with the project, which uses none): enable with
`uv sync --extra qwen`. `chatterbox` and `kokoro` pin conflicting torch builds,
so install those in their own environment (see docs/tts_engines.md).

Every adapter yields 16-bit mono PCM at SAMPLE_RATE (24 kHz), matching the
Orpheus path, so commentator.tts.pcm_chunks_to_wav consumes the output
unchanged. Unlike Orpheus these models synthesize the whole clip at once, so the
adapter yields a single chunk (no token-level streaming).

Supported alternatives: chatterbox, kokoro, qwen (Qwen3-TTS, multilingual with
named speakers and a natural-language `instruct` style control).

Benchmarked on the project's RX 7900 XTX (see tts_eval/): naturalness UTMOS —
chatterbox 4.37, kokoro 4.40, orpheus 4.26; speed (RTF, lower = faster) —
orpheus 0.62, kokoro 0.20 (CPU), chatterbox 1.51, qwen ~5 (1.7B, cold; slowest).
Chatterbox is the most expressive but slower than real time; kokoro is fastest
but emotionally flat; qwen is multilingual but heavy.
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
_QWEN_MODEL = None


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
    if engine == "qwen":
        return _qwen_chunks(text)
    raise ValueError(
        f"Unknown TTS_ENGINE {engine!r}; expected 'orpheus', 'chatterbox', "
        "'kokoro', or 'qwen'"
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


def _get_qwen() -> object:
    global _QWEN_MODEL
    if _QWEN_MODEL is not None:
        return _QWEN_MODEL
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "TTS_ENGINE=qwen needs the 'qwen-tts' package. It coexists in the main "
            "env (no separate venv): run `uv sync --extra qwen`."
        ) from exc
    repo = os.getenv("QWEN_TTS_REPO", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    use_cuda = torch.cuda.is_available()
    device_map = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    logger.info("Loading Qwen3-TTS %s on %s", repo, device_map)
    _QWEN_MODEL = Qwen3TTSModel.from_pretrained(repo, device_map=device_map, dtype=dtype)
    return _QWEN_MODEL


def _qwen_chunks(text: str) -> Generator[bytes, None, None]:
    # Qwen3-TTS outputs 24 kHz float audio (matches SAMPLE_RATE). The optional
    # natural-language `instruct` steers delivery style (e.g. "speak excitedly").
    speaker = os.getenv("QWEN_TTS_SPEAKER", "ryan")
    language = os.getenv("QWEN_TTS_LANGUAGE", "english")
    instruct = os.getenv("QWEN_TTS_INSTRUCT") or None
    model = _get_qwen()
    wavs, _sr = model.generate_custom_voice(
        text=text, speaker=speaker, language=language, instruct=instruct
    )
    yield _float_to_pcm16(wavs[0])
