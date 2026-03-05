import io
import os
import re
import time
import wave
from collections.abc import Generator, Iterable
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from snac import SNAC
from commentator.llama_lock import LLAMA_CPP_LOCK

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

_TEMPERATURE = float(os.getenv("ORPHEUS_TEMPERATURE", "0.6"))
_TOP_P = float(os.getenv("ORPHEUS_TOP_P", "0.9"))
_SAMPLE_RATE = int(os.getenv("ORPHEUS_SAMPLE_RATE", "24000"))
_MAX_DECODE_SECONDS = float(os.getenv("ORPHEUS_TTS_MAX_SECONDS", "60"))
_CHUNK_FRAMES = int(os.getenv("ORPHEUS_CHUNK_FRAMES", "24"))
_DEBUG = os.getenv("ORPHEUS_TTS_DEBUG", "0") == "1"

_CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

_SNAC_MODEL: SNAC | None = None
_SNAC_DEVICE: torch.device | None = None
_LLAMA_MODEL = None

_GGUF_REPO = os.getenv("ORPHEUS_GGUF_REPO", "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF")
_GGUF_FILE = os.getenv("ORPHEUS_GGUF_FILE", "orpheus-3b-0.1-ft-q4_k_m.gguf")
_LLAMA_CTX = int(os.getenv("ORPHEUS_LLAMA_CTX", "4096"))
_LLAMA_BATCH = int(os.getenv("ORPHEUS_LLAMA_BATCH", "64"))
_LLAMA_GPU_LAYERS = int(os.getenv("ORPHEUS_LLAMA_GPU_LAYERS", "-1"))
_LLAMA_MAIN_GPU = int(os.getenv("ORPHEUS_LLAMA_MAIN_GPU", "0"))
_LLAMA_TENSOR_SPLIT = [
    float(x)
    for x in os.getenv("ORPHEUS_LLAMA_TENSOR_SPLIT", "1.0").split(",")
    if x.strip()
]


def _format_prompt(text: str, voice: str) -> str:
    return f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"


def _speed_to_generation(speed: float) -> dict[str, float]:
    speed_norm = (speed - 0.8) / 0.6
    speed_norm = max(0.0, min(1.0, speed_norm))
    return {
        "temperature": _TEMPERATURE + 0.4 * speed_norm,
        "top_p": _TOP_P,
        "repeat_penalty": 1.1 + 0.2 * speed_norm,
    }


def _turn_token_into_id(raw_token: int, index: int) -> int:
    return raw_token - 10 - ((index % 7) * 4096)


def _get_snac_model() -> tuple[SNAC, torch.device]:
    global _SNAC_MODEL, _SNAC_DEVICE
    if _SNAC_MODEL is not None and _SNAC_DEVICE is not None:
        return _SNAC_MODEL, _SNAC_DEVICE

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if _DEBUG:
        backend = "GPU (CUDA)" if use_cuda else "CPU"
        print(f"TTS debug: compute backend={backend} (torch.cuda_available={use_cuda})")
        print(f"TTS debug: loading SNAC model on {device}")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    model.eval()

    _SNAC_MODEL = model
    _SNAC_DEVICE = device
    return model, device


def _get_llama_model():
    global _LLAMA_MODEL
    if _LLAMA_MODEL is not None:
        return _LLAMA_MODEL
    if Llama is None:
        raise RuntimeError("llama_cpp is not installed")
    with LLAMA_CPP_LOCK:
        if _LLAMA_MODEL is not None:
            return _LLAMA_MODEL
        if _DEBUG:
            print("TTS debug: downloading/locating GGUF")
        gguf_path = hf_hub_download(repo_id=_GGUF_REPO, filename=_GGUF_FILE)

        if _DEBUG:
            print("TTS debug: loading llama.cpp model")
            if _LLAMA_GPU_LAYERS == 0:
                print("TTS debug: llama.cpp compute backend=CPU only (n_gpu_layers=0)")
            else:
                print(
                    "TTS debug: llama.cpp compute backend=GPU offload enabled "
                    f"(n_gpu_layers={_LLAMA_GPU_LAYERS}, main_gpu={_LLAMA_MAIN_GPU}, "
                    f"tensor_split={_LLAMA_TENSOR_SPLIT})"
                )
        _LLAMA_MODEL = Llama(
            model_path=gguf_path,
            n_ctx=_LLAMA_CTX,
            n_gpu_layers=_LLAMA_GPU_LAYERS,
            n_batch=_LLAMA_BATCH,
            main_gpu=_LLAMA_MAIN_GPU,
            tensor_split=_LLAMA_TENSOR_SPLIT,
            verbose=False,
        )
    return _LLAMA_MODEL


@torch.no_grad()
def _decode_frames_to_pcm(frame_tokens: list[int]) -> bytes | None:
    num_frames = len(frame_tokens) // 7
    if num_frames < 4:
        return None

    frame = frame_tokens[: num_frames * 7]
    codes_0 = np.empty(num_frames, dtype=np.int64)
    codes_1 = np.empty(num_frames * 2, dtype=np.int64)
    codes_2 = np.empty(num_frames * 4, dtype=np.int64)

    for j in range(num_frames):
        idx = j * 7
        codes_0[j] = frame[idx]
        codes_1[j * 2] = frame[idx + 1]
        codes_1[j * 2 + 1] = frame[idx + 4]
        codes_2[j * 4] = frame[idx + 2]
        codes_2[j * 4 + 1] = frame[idx + 3]
        codes_2[j * 4 + 2] = frame[idx + 5]
        codes_2[j * 4 + 3] = frame[idx + 6]

    if (
        np.any(codes_0 < 0)
        or np.any(codes_0 > 4096)
        or np.any(codes_1 < 0)
        or np.any(codes_1 > 4096)
        or np.any(codes_2 < 0)
        or np.any(codes_2 > 4096)
    ):
        return None

    model, device = _get_snac_model()
    codes = [
        torch.from_numpy(codes_0.reshape(1, -1)).long().to(device),
        torch.from_numpy(codes_1.reshape(1, -1)).long().to(device),
        torch.from_numpy(codes_2.reshape(1, -1)).long().to(device),
    ]
    audio_hat = model.decode(codes).cpu().numpy()
    audio_int16 = (audio_hat.flatten() * 32767).astype(np.int16)
    return audio_int16.tobytes()


def _iter_custom_tokens_from_text_stream(
    text_stream: Generator[str, None, None],
) -> Generator[int, None, None]:
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        while True:
            match = _CUSTOM_TOKEN_RE.search(buffer)
            if not match:
                if len(buffer) > 128:
                    buffer = buffer[-128:]
                break
            yield int(match.group(1))
            buffer = buffer[match.end() :]


def _iter_custom_tokens_from_llama_cpp(
    prompt: str,
    options: dict[str, float],
) -> Generator[int, None, None]:
    llm = _get_llama_model()
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
        yield from _iter_custom_tokens_from_text_stream(text_stream)


def iter_audio_chunks(
    text: str, voice: str = "zac", speed: float = 1.3
) -> Generator[bytes, None, None]:
    start_time = time.time()
    options = _speed_to_generation(speed)
    prompt = _format_prompt(text, voice)
    pending_tokens: list[int] = []
    count = 0

    token_stream = _iter_custom_tokens_from_llama_cpp(prompt, options)

    for raw_token in token_stream:
        if time.time() - start_time > _MAX_DECODE_SECONDS:
            if _DEBUG:
                print("TTS debug: decode timeout hit")
            break

        token = _turn_token_into_id(raw_token, count)
        if token <= 0:
            continue

        pending_tokens.append(token)
        count += 1

        if len(pending_tokens) >= _CHUNK_FRAMES * 7:
            pcm = _decode_frames_to_pcm(pending_tokens[: _CHUNK_FRAMES * 7])
            if pcm:
                yield pcm
            pending_tokens = pending_tokens[_CHUNK_FRAMES * 7 :]

    while len(pending_tokens) >= 4 * 7:
        take = min(len(pending_tokens) // 7, _CHUNK_FRAMES) * 7
        pcm = _decode_frames_to_pcm(pending_tokens[:take])
        if pcm:
            yield pcm
        pending_tokens = pending_tokens[take:]


def pcm_chunks_to_wav(
    chunks: Iterable[bytes], sample_rate: int = _SAMPLE_RATE
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in chunks:
            wf.writeframes(chunk)
    return buf.getvalue()


def _resample_pcm(pcm: bytes, speed: float) -> bytes:
    if speed == 1.0:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size < 2:
        return pcm
    new_len = max(1, int(samples.size / speed))
    x_old = np.arange(samples.size, dtype=np.float32)
    x_new = np.linspace(0, samples.size - 1, new_len, dtype=np.float32)
    resampled = np.interp(x_new, x_old, samples.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def text_to_speech(text: str, voice: str = "zac", speed: float = 1.3) -> bytes | None:
    try:
        chunks = list(iter_audio_chunks(text, voice=voice, speed=speed))
        if not chunks:
            return None
        pcm = b"".join(chunks)
        pcm = _resample_pcm(pcm, speed)
        return pcm_chunks_to_wav([pcm], sample_rate=_SAMPLE_RATE)
    except Exception as exc:
        print(f"TTS error: {exc}")
        return None
