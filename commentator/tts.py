"""Orpheus TTS: generate audio tokens via llama.cpp and decode with SNAC."""

import io
import logging
import os
import queue
import re
import threading
import time
import wave
from collections.abc import Generator, Iterable
from typing import Any, cast

import numpy as np
import torch
from num2words import num2words
from snac import SNAC

from commentator._config import env_bool, env_float, env_int, parse_tensor_split
from commentator.llama_loader import LazyLlama
from commentator.llama_lock import LLAMA_CPP_LOCK

logger = logging.getLogger(__name__)

__all__ = [
    "iter_audio_chunks",
    "pcm_chunks_to_wav",
    "text_to_speech",
    "SAMPLE_RATE",
    "VALID_VOICES",
]


VALID_VOICES: frozenset[str] = frozenset(
    {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}
)

_TEMPERATURE = env_float("ORPHEUS_TEMPERATURE", 0.6)
_TOP_P = env_float("ORPHEUS_TOP_P", 0.9)
# SAMPLE_RATE must be >= 1 to prevent division-by-zero in callers that compute
# duration as len(chunk) / (SAMPLE_RATE * 2).
SAMPLE_RATE = max(1, env_int("ORPHEUS_SAMPLE_RATE", 24000))
# _MAX_DECODE_SECONDS must be >= 1 so the timeout check is not immediately triggered.
_MAX_DECODE_SECONDS = max(1.0, env_float("ORPHEUS_TTS_MAX_SECONDS", 60.0))
# SNAC frames per decode batch (1 frame = 7 tokens).
# Must be >= 1; 0 would make the accumulation condition always true.
_CHUNK_FRAMES = max(1, env_int("ORPHEUS_CHUNK_FRAMES", 24))
# Frames in the FIRST decoded chunk only. A smaller first chunk cuts
# time-to-first-audio when streaming (e.g. 8 frames ≈ 0.4s vs 24 ≈ 1.3s) while
# the rest decode at _CHUNK_FRAMES, so only one extra batch boundary is added
# (SNAC decodes each batch independently). Clamped to [4, _CHUNK_FRAMES]: SNAC
# needs ≥4 frames, and it never exceeds the steady-state batch size.
_FIRST_CHUNK_FRAMES = min(_CHUNK_FRAMES, max(4, env_int("ORPHEUS_FIRST_CHUNK_FRAMES", 8)))

_CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

# Sentinel pushed onto the token queue to signal the producer thread is done.
_TOKEN_STREAM_END = object()

# Number→speech: the model mangles digit strings ($312.07 spoken as "31.07"), so
# convert prices/percentages/bare numbers to spoken words before synthesis. The
# commentary keeps digits for display; only the TTS input is converted.
# The comma-grouped branch requires a comma (+) so a plain run like 1000 falls to
# the \d+ branch and is matched whole, not truncated to its first three digits.
_MONEY_RE = re.compile(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")
_PERCENT_RE = re.compile(r"([+-]?)(\d+(?:\.\d+)?)\s*%")
_BARE_NUM_RE = re.compile(r"(?<![\w$])([+-]?\d+(?:\.\d+)?)(?![\w%])")


def _spoken_money(m: "re.Match[str]") -> str:
    dollars = int(m.group(1).replace(",", ""))
    cents_s = m.group(2)
    words = f"{num2words(dollars)} {'dollar' if dollars == 1 else 'dollars'}"
    if cents_s:
        cents = int(cents_s.ljust(2, "0")[:2])
        if cents:
            words += f" and {num2words(cents)} {'cent' if cents == 1 else 'cents'}"
    return words


def _spoken_percent(m: "re.Match[str]") -> str:
    sign, num = m.group(1), m.group(2)
    value = float(num) if "." in num else int(num)
    prefix = "negative " if sign == "-" else ""
    return f"{prefix}{num2words(value)} percent"


def _spoken_number(m: "re.Match[str]") -> str:
    num = m.group(1)
    value = float(num) if "." in num else int(num.lstrip("+"))
    return num2words(value)


def numbers_to_speech(text: str) -> str:
    """Replace prices, percentages, and bare numbers with spoken words so the TTS
    model pronounces them correctly. Falls back to the original text on error."""
    try:
        text = _MONEY_RE.sub(_spoken_money, text)
        text = _PERCENT_RE.sub(_spoken_percent, text)
        text = _BARE_NUM_RE.sub(_spoken_number, text)
    except (ValueError, OverflowError):
        logger.warning("numbers_to_speech conversion failed; leaving text as-is")
    return text


# Valid SNAC codebook range: 0 inclusive to 4096 exclusive (4096 entries).
_SNAC_CODE_MAX = 4096

_SNAC_MODEL: SNAC | None = None
_SNAC_DEVICE: torch.device | None = None
# guards SNAC singleton init only; separate from LLAMA_CPP_LOCK (inference)
_SNAC_INIT_LOCK = threading.Lock()

_ORPHEUS_TTS_DEBUG = env_bool("ORPHEUS_TTS_DEBUG", False)
# Warm up the LLM kernels and the SNAC decoder when the Orpheus model loads,
# instead of paying that cost (HIP kernel JIT + SNAC load/download) during the
# first user-visible TTS generation — which otherwise stalls the audio stream
# right as the first chunk should be decoded. Set ORPHEUS_TTS_WARMUP=0 to skip.
_ORPHEUS_TTS_WARMUP = env_bool("ORPHEUS_TTS_WARMUP", True)

# TTS engine selector. "orpheus" (default) is the built-in llama.cpp + SNAC
# pipeline. "chatterbox" and "kokoro" are optional alternative engines (see
# commentator/tts_engines.py) that must be installed separately — their deps
# conflict with the project's, so they live in their own venv/extras.
TTS_ENGINE = os.getenv("TTS_ENGINE", "orpheus").strip().lower()

_GGUF_REPO = os.getenv("ORPHEUS_GGUF_REPO", "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF")
_GGUF_FILE = os.getenv("ORPHEUS_GGUF_FILE", "orpheus-3b-0.1-ft-q4_k_m.gguf")
_LLAMA_CTX = env_int("ORPHEUS_LLAMA_CTX", 4096)
# n_batch/n_ubatch govern prompt prefill throughput. llama.cpp's own default is
# 512; the previous 64 throttled prefill for no benefit on GPU.
_LLAMA_BATCH = env_int("ORPHEUS_LLAMA_BATCH", 512)
_LLAMA_UBATCH = env_int("ORPHEUS_LLAMA_UBATCH", 512)
_LLAMA_GPU_LAYERS = env_int("ORPHEUS_LLAMA_GPU_LAYERS", -1)
_LLAMA_MAIN_GPU = env_int("ORPHEUS_LLAMA_MAIN_GPU", 0)
_LLAMA_TENSOR_SPLIT = parse_tensor_split("ORPHEUS_LLAMA_TENSOR_SPLIT")
# Flash attention speeds the long (up to 2048-token) audio-token decode and
# halves KV-cache memory. Supported on the target gfx1100 GPU; disable
# (ORPHEUS_LLAMA_FLASH_ATTN=0) if a llama.cpp build lacks FA kernels.
_LLAMA_FLASH_ATTN = env_bool("ORPHEUS_LLAMA_FLASH_ATTN", True)


def _format_prompt(text: str, voice: str) -> str:
    """Build the Orpheus prompt from text and voice name."""
    return f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"


def _speed_to_generation(speed: float) -> dict[str, float]:
    """Map a human-readable speed factor to LLM generation parameters.

    Speed is normalized to [0.0, 1.0] over the range [0.8, 1.4] and then
    clamped, so values outside that range are treated as the nearest endpoint.
    Higher speed increases temperature and repeat penalty; the higher repeat
    penalty prevents the repetition loops Orpheus is prone to, reducing total
    token count.
    """
    if not (0.8 <= speed <= 1.4):
        logger.warning("_speed_to_generation: speed %.2f is outside [0.8, 1.4]; clamping", speed)
    speed_norm = (speed - 0.8) / 0.6
    speed_norm = max(0.0, min(1.0, speed_norm))
    return {
        "temperature": _TEMPERATURE + 0.4 * speed_norm,
        "top_p": _TOP_P,
        "repeat_penalty": 1.1 + 0.2 * speed_norm,
    }


def _turn_token_into_id(raw_token: int, index: int) -> int:
    """Convert a raw Orpheus custom token to a SNAC codebook index.

    The result may fall outside the valid codebook range [0, 4096). Negative
    results indicate a token that is too low for its codebook position and must
    be discarded by the caller (skip if result < 0). Results >= 4096 are also
    invalid but are caught by the bounds check in _decode_frames_to_pcm.
    """
    return raw_token - 10 - ((index % 7) * 4096)


def _get_snac_model() -> tuple[SNAC, torch.device]:
    """Lazy-init the SNAC decoder singleton (thread-safe double-checked lock)."""
    global _SNAC_MODEL, _SNAC_DEVICE
    if _SNAC_MODEL is not None and _SNAC_DEVICE is not None:
        return _SNAC_MODEL, _SNAC_DEVICE
    with _SNAC_INIT_LOCK:
        if _SNAC_MODEL is not None and _SNAC_DEVICE is not None:
            return _SNAC_MODEL, _SNAC_DEVICE
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info("Loading SNAC model on %s (cuda_available=%s)", device, use_cuda)
        t0 = time.time()
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        model.eval()
        _SNAC_MODEL = model
        _SNAC_DEVICE = device
        logger.info("SNAC model loaded on %s in %.1fs", device, time.time() - t0)
    return _SNAC_MODEL, _SNAC_DEVICE


def _warmup_orpheus_llm(model: Any) -> None:
    # Warm the LLM decode kernels with a 1-token generation.
    model(_format_prompt("warm up", "tara"), max_tokens=1)
    # Load + warm SNAC now so its load (and possible first-run download) doesn't
    # stall the audio stream mid-generation. Four zero-valued frames form a valid
    # in-range batch the decoder accepts (28 tokens = 4 * 7).
    _decode_frames_to_pcm([0] * (4 * 7))


_LLAMA_MODEL = LazyLlama(
    label="Orpheus GGUF",
    repo_id=_GGUF_REPO,
    filename=_GGUF_FILE,
    n_ctx=_LLAMA_CTX,
    n_gpu_layers=_LLAMA_GPU_LAYERS,
    n_batch=_LLAMA_BATCH,
    n_ubatch=_LLAMA_UBATCH,
    main_gpu=_LLAMA_MAIN_GPU,
    tensor_split=_LLAMA_TENSOR_SPLIT,
    flash_attn=_LLAMA_FLASH_ATTN,
    verbose=_ORPHEUS_TTS_DEBUG,
    warmup=_warmup_orpheus_llm if _ORPHEUS_TTS_WARMUP else None,
)


def _get_llama_model() -> Any:
    """Lazy-init the Orpheus TTS LLM singleton (thread-safe)."""
    return _LLAMA_MODEL.get()


@torch.inference_mode()
def _decode_frames_to_pcm(frame_tokens: list[int]) -> bytes | None:
    """Decode SNAC frame tokens to 16-bit PCM bytes.

    Returns None if there are fewer than 4 complete frames (28 tokens)
    or if any token falls outside the valid codebook range [0, 4096).
    """
    num_frames = len(frame_tokens) // 7
    if num_frames < 4:
        logger.debug("_decode_frames_to_pcm: too few frames (%d < 4), skipping", num_frames)
        return None

    frame = frame_tokens[: num_frames * 7]
    arr = np.array(frame, dtype=np.int64).reshape(num_frames, 7)
    # SNAC uses 3 hierarchical codebooks. Each frame has 7 tokens laid out as:
    #   col 0       → codebook 0 (coarsest, 1 code/frame)
    #   cols 1, 4   → codebook 1 (2 codes/frame, interleaved)
    #   cols 2,3,5,6 → codebook 2 (finest, 4 codes/frame, interleaved)
    codes_0 = np.ascontiguousarray(arr[:, 0])
    codes_1 = np.empty(num_frames * 2, dtype=np.int64)
    codes_1[0::2] = arr[:, 1]
    codes_1[1::2] = arr[:, 4]
    codes_2 = np.empty(num_frames * 4, dtype=np.int64)
    codes_2[0::4] = arr[:, 2]
    codes_2[1::4] = arr[:, 3]
    codes_2[2::4] = arr[:, 5]
    codes_2[3::4] = arr[:, 6]

    # Negative tokens are normally filtered out in _iter_audio_chunks_gen before
    # being added to pending_tokens, but guard the full range [0, _SNAC_CODE_MAX)
    # here too: an out-of-range index passed straight to SNAC raises IndexError
    # (CPU) or reads garbage (GPU), so this function must be self-defensive.
    for codes in (codes_0, codes_1, codes_2):
        if np.any(codes < 0) or np.any(codes >= _SNAC_CODE_MAX):
            logger.debug(
                "_decode_frames_to_pcm: token out of codebook range [0, %d), dropping %d frames",
                _SNAC_CODE_MAX,
                num_frames,
            )
            return None

    model, device = _get_snac_model()
    codes = [
        torch.from_numpy(codes_0.reshape(1, -1)).to(device),
        torch.from_numpy(codes_1.reshape(1, -1)).to(device),
        torch.from_numpy(codes_2.reshape(1, -1)).to(device),
    ]
    audio_hat = model.decode(codes).cpu().numpy().ravel()
    audio_int16 = (audio_hat * 32767).astype(np.int16)
    return audio_int16.tobytes()


def _iter_custom_tokens_from_text_stream(
    text_stream: Iterable[str],
) -> Generator[int, None, None]:
    """Extract integer custom_token IDs from a stream of text chunks."""
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        while True:
            match = _CUSTOM_TOKEN_RE.search(buffer)
            if not match:
                # Keep only the tail to avoid unbounded buffer growth.
                if len(buffer) > 128:
                    buffer = buffer[-128:]
                break
            yield int(match.group(1))
            buffer = buffer[match.end() :]


def _stream_custom_tokens(
    prompt: str,
    options: dict[str, float],
    stop_event: threading.Event,
) -> Generator[int, None, None]:
    """Yield Orpheus custom tokens as llama.cpp produces them.

    Generation runs in a background thread that holds LLAMA_CPP_LOCK only for the
    duration of inference. The consumer decodes each frame batch through SNAC
    while the producer keeps generating, so audio decode overlaps token
    generation instead of running serially after it. The lock is still released
    the instant the last token is produced, so the commentary LLM is not blocked
    during decode.

    Setting stop_event makes the producer stop at the next token boundary; the
    consumer sets it (and closes this generator) on the decode timeout.
    """
    llm = _get_llama_model()
    # maxsize comfortably exceeds the 2048 max_tokens cap, so put() never blocks
    # and the producer always observes stop_event between tokens (no deadlock).
    token_q: "queue.Queue[Any]" = queue.Queue(maxsize=4096)
    error_box: dict[str, BaseException] = {}

    def _produce() -> None:
        try:
            with LLAMA_CPP_LOCK:
                stream = llm(
                    prompt,
                    max_tokens=2048,
                    temperature=options["temperature"],
                    top_p=options["top_p"],
                    repeat_penalty=options["repeat_penalty"],
                    stream=True,
                )
                text_stream = (
                    str(cast(dict[str, Any], item)["choices"][0]["text"]) for item in stream
                )
                for tok in _iter_custom_tokens_from_text_stream(text_stream):
                    if stop_event.is_set():
                        break
                    token_q.put(tok)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            error_box["error"] = exc
        finally:
            token_q.put(_TOKEN_STREAM_END)

    producer = threading.Thread(target=_produce, name="orpheus-tokens", daemon=True)
    producer.start()
    try:
        while True:
            item = token_q.get()
            if item is _TOKEN_STREAM_END:
                break
            yield cast(int, item)
    finally:
        stop_event.set()
        producer.join(timeout=5.0)
    if "error" in error_box:
        raise error_box["error"]


def iter_audio_chunks(
    text: str, voice: str = "zac", speed: float = 1.3
) -> Generator[bytes, None, None]:
    """Generate PCM audio chunks from text via Orpheus TTS.

    Yields 16-bit mono PCM byte chunks as they are decoded. Stops early
    if decoding exceeds the ORPHEUS_TTS_MAX_SECONDS timeout (default 60s)
    to prevent runaway generation.

    Raises ValueError for unknown voice names at call time (eagerly).
    Valid voices: VALID_VOICES.

    speed is clamped to [0.8, 1.4]; values outside that range are treated as
    the nearest endpoint. Higher speed increases temperature and repeat penalty.

    When TTS_ENGINE is set to a non-default engine ("chatterbox", "kokoro"), the
    call is routed to that engine's adapter instead (see commentator.tts_engines).
    All engines yield 16-bit mono PCM at SAMPLE_RATE (24 kHz).
    """
    # Convert digit numbers to spoken words for every engine (the displayed
    # commentary keeps the concise digit form; only synthesis sees the words).
    text = numbers_to_speech(text)
    if TTS_ENGINE != "orpheus":
        from commentator.tts_engines import iter_audio_chunks as _engine_chunks

        return _engine_chunks(TTS_ENGINE, text, voice, speed)
    if voice not in VALID_VOICES:
        raise ValueError(f"Unknown voice {voice!r}. Valid voices: {sorted(VALID_VOICES)}")
    return _iter_audio_chunks_gen(text, voice, speed)


def _iter_audio_chunks_gen(text: str, voice: str, speed: float) -> Generator[bytes, None, None]:
    """Internal generator that yields PCM chunks. Voice must be pre-validated."""
    logger.info("tts_start voice=%s speed=%.2f text_len=%d", voice, speed, len(text))
    start_time = time.time()
    options = _speed_to_generation(speed)
    prompt = _format_prompt(text, voice)

    stop_event = threading.Event()
    # Token generation runs in a background thread; SNAC decode below overlaps it.
    token_gen = _stream_custom_tokens(prompt, options, stop_event)

    decode_start = time.time()
    pending_tokens: list[int] = []
    count = 0  # valid-token counter; determines codebook position in _turn_token_into_id
    total_tokens = 0
    dropped_chunks = 0
    first_chunk_done = False  # the first emitted chunk uses _FIRST_CHUNK_FRAMES

    try:
        for i, raw_token in enumerate(token_gen):
            total_tokens += 1
            if i % 64 == 0:  # amortize time.time() syscall over every 64 tokens
                elapsed = time.time() - decode_start
                if elapsed > _MAX_DECODE_SECONDS:
                    logger.warning(
                        "TTS decode timeout hit after %.1fs; audio may be truncated",
                        elapsed,
                    )
                    break

            token = _turn_token_into_id(raw_token, count)
            if token < 0:
                continue

            pending_tokens.append(token)
            count += 1

            # First emitted chunk is smaller (faster first audio when streaming);
            # subsequent chunks use the full batch size.
            chunk_frames = _CHUNK_FRAMES if first_chunk_done else _FIRST_CHUNK_FRAMES
            if len(pending_tokens) >= chunk_frames * 7:
                pcm = _decode_frames_to_pcm(pending_tokens[: chunk_frames * 7])
                if pcm:
                    yield pcm
                    first_chunk_done = True
                else:
                    dropped_chunks += 1
                del pending_tokens[: chunk_frames * 7]

        # Drain remaining pending tokens in frame-aligned chunks
        # (min 4 frames required by SNAC decoder).
        while len(pending_tokens) >= 4 * 7:
            take = min(len(pending_tokens) // 7, _CHUNK_FRAMES) * 7
            pcm = _decode_frames_to_pcm(pending_tokens[:take])
            if pcm:
                yield pcm
            else:
                dropped_chunks += 1
            del pending_tokens[:take]
    finally:
        # Stop and tear down the producer thread (raises any generation error).
        stop_event.set()
        token_gen.close()

    if dropped_chunks:
        logger.warning(
            "TTS dropped %d frame batch(es) due to invalid token ranges voice=%s",
            dropped_chunks,
            voice,
        )
    logger.info(
        "TTS complete voice=%s elapsed=%.2fs tokens=%d dropped_batches=%d",
        voice,
        time.time() - start_time,
        total_tokens,
        dropped_chunks,
    )


def text_to_speech(text: str, voice: str = "zac", speed: float = 1.3) -> bytes | None:
    """Convert text to a WAV audio bytes object. Returns None if no audio was generated.

    Raises ValueError for unknown voice names. Valid voices: VALID_VOICES.
    speed is clamped to [0.8, 1.4]; see iter_audio_chunks for details.
    """
    chunks = list(iter_audio_chunks(text, voice=voice, speed=speed))
    if not chunks:
        return None
    return pcm_chunks_to_wav(chunks)


def pcm_chunks_to_wav(chunks: Iterable[bytes], sample_rate: int = SAMPLE_RATE) -> bytes:
    """Assemble raw 16-bit mono PCM chunks into a WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in chunks:
            wf.writeframes(chunk)
    return buf.getvalue()
