#!/usr/bin/env python3
"""Benchmark: SNAC decode — PyTorch vs ONNX CPU.

Streams Orpheus tokens from Ollama for 10 sentences, then decodes each
sentence's token list with both backends and compares wall-clock times.

Fully self-contained — no imports from commentator/.
"""

import http.client
import json
import os
import time
import urllib.parse

import numpy as np
import onnxruntime
import torch
from huggingface_hub import hf_hub_download
from snac import SNAC

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("ORPHEUS_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("ORPHEUS_MODEL_NAME", "legraphista/Orpheus:3b-ft-q8")
VOICE = "zac"
SENTENCES = [
    "The market is absolutely surging right now, what a rally!",
    "We're seeing heavy selling pressure across the board today.",
    "This stock just broke through a key resistance level.",
    "Volume is picking up significantly in the final hour of trading.",
    "A surprising earnings beat is sending shares higher after hours.",
    "The bears are in full control as we approach the session low.",
    "What a comeback! Buyers stepped in right at support.",
    "Volatility is through the roof with the VIX spiking hard.",
    "The moving averages just crossed, signaling a potential trend change.",
    "And that's a new all-time high, folks! History in the making!",
]

CUSTOM_TOKEN_PREFIX = "<custom_token_"


# ---------------------------------------------------------------------------
# Ollama streaming
# ---------------------------------------------------------------------------
def _ollama_stream(prompt: str) -> list[str]:
    """Stream tokens from Ollama and return list of raw token strings."""
    parsed = urllib.parse.urlparse(OLLAMA_HOST)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname, parsed.port, timeout=120)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1},
    }
    conn.request(
        "POST",
        "/api/generate",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Ollama error: {resp.status} {resp.reason} {detail}")

    tokens: list[str] = []
    buffer = ""
    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        data = json.loads(line)
        if data.get("response"):
            buffer += data["response"]
            # Extract complete custom tokens from buffer
            while True:
                start = buffer.find(CUSTOM_TOKEN_PREFIX)
                if start == -1:
                    if len(buffer) > len(CUSTOM_TOKEN_PREFIX):
                        buffer = buffer[-len(CUSTOM_TOKEN_PREFIX) :]
                    break
                end = buffer.find(">", start)
                if end == -1:
                    buffer = buffer[start:]
                    break
                tokens.append(buffer[start : end + 1])
                buffer = buffer[end + 1 :]
        if data.get("done"):
            break
    conn.close()
    return tokens


def _format_prompt(text: str) -> str:
    return f"<|audio|>{VOICE}: {text}<|eot_id|><custom_token_4>"


# ---------------------------------------------------------------------------
# Token parsing (shared)
# ---------------------------------------------------------------------------
def _parse_token_ids(raw_tokens: list[str]) -> list[int]:
    """Convert raw token strings to integer IDs, dropping invalid ones."""
    ids: list[int] = []
    count = 0
    for tok in raw_tokens:
        tok = tok.strip()
        if not (tok.startswith(CUSTOM_TOKEN_PREFIX) and tok.endswith(">")):
            continue
        try:
            num = int(tok[14:-1]) - 10 - ((count % 7) * 4096)
        except ValueError:
            continue
        if num <= 0:
            continue
        ids.append(num)
        count += 1
    return ids


# ---------------------------------------------------------------------------
# Reshape: 7 tokens/frame → 3 codebooks
# ---------------------------------------------------------------------------
def _reshape_codes_numpy(token_ids: list[int]):
    """Return (codes_0, codes_1, codes_2) as numpy arrays."""
    num_frames = len(token_ids) // 7
    if num_frames == 0:
        return None
    frame = token_ids[: num_frames * 7]

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

    return (
        codes_0.reshape(1, -1),
        codes_1.reshape(1, -1),
        codes_2.reshape(1, -1),
    )


def _reshape_codes_torch(token_ids: list[int]):
    """Return (codes_0, codes_1, codes_2) as torch tensors."""
    result = _reshape_codes_numpy(token_ids)
    if result is None:
        return None
    c0, c1, c2 = result
    return (
        torch.from_numpy(c0),
        torch.from_numpy(c1),
        torch.from_numpy(c2),
    )


# ---------------------------------------------------------------------------
# ONNX decode
# ---------------------------------------------------------------------------
def _load_onnx_session() -> onnxruntime.InferenceSession:
    path = hf_hub_download(
        "onnx-community/snac_24khz-ONNX",
        subfolder="onnx",
        filename="decoder_model.onnx",
    )
    return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])


def _decode_onnx(
    session: onnxruntime.InferenceSession, codes: tuple[np.ndarray, ...]
) -> np.ndarray:
    input_names = [x.name for x in session.get_inputs()]
    input_dict = dict(zip(input_names, codes))
    audio_hat = session.run(None, input_dict)[0]
    return audio_hat


# ---------------------------------------------------------------------------
# PyTorch decode
# ---------------------------------------------------------------------------
def _load_pytorch_model() -> SNAC:
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model.eval()
    return model


@torch.no_grad()
def _decode_pytorch(model: SNAC, codes: tuple[torch.Tensor, ...]) -> np.ndarray:
    codes_list = [c.long() for c in codes]
    audio_hat = model.decode(codes_list)
    return audio_hat.numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SNAC Decode Benchmark: PyTorch vs ONNX (CPU)")
    print("=" * 60)

    # Phase 1: collect tokens from Ollama
    print(f"\nPhase 1: Streaming tokens for {len(SENTENCES)} sentences...")
    all_token_ids: list[list[int]] = []
    for i, sentence in enumerate(SENTENCES):
        print(f"  [{i + 1}/{len(SENTENCES)}] {sentence[:50]}...")
        prompt = _format_prompt(sentence)
        raw_tokens = _ollama_stream(prompt)
        token_ids = _parse_token_ids(raw_tokens)
        all_token_ids.append(token_ids)
        print(f"           → {len(raw_tokens)} raw tokens → {len(token_ids)} valid IDs")

    # Check we have enough tokens
    valid = [(i, ids) for i, ids in enumerate(all_token_ids) if len(ids) >= 7]
    if not valid:
        print("\nERROR: No sentences produced enough tokens to decode.")
        return

    print(f"\n{len(valid)}/{len(SENTENCES)} sentences have decodable tokens.\n")

    # Phase 2: ONNX decode
    print("Phase 2: Loading ONNX model...")
    onnx_session = _load_onnx_session()
    print("  ONNX session ready (CPUExecutionProvider)\n")

    onnx_times: list[float] = []
    for i, ids in valid:
        codes = _reshape_codes_numpy(ids)
        if codes is None:
            continue
        t0 = time.perf_counter()
        _decode_onnx(onnx_session, codes)
        dt = time.perf_counter() - t0
        onnx_times.append(dt)
        frames = len(ids) // 7
        print(f"  Sentence {i + 1:2d}: {dt:.4f}s  ({frames} frames, {len(ids)} tokens)")

    onnx_total = sum(onnx_times)
    print(
        f"  ONNX total: {onnx_total:.4f}s  avg: {onnx_total / len(onnx_times):.4f}s\n"
    )

    # Phase 3: PyTorch decode
    print("Phase 3: Loading PyTorch model...")
    pt_model = _load_pytorch_model()
    print("  PyTorch model ready (CPU)\n")

    pytorch_times: list[float] = []
    for i, ids in valid:
        codes = _reshape_codes_torch(ids)
        if codes is None:
            continue
        t0 = time.perf_counter()
        _decode_pytorch(pt_model, codes)
        dt = time.perf_counter() - t0
        pytorch_times.append(dt)
        frames = len(ids) // 7
        print(f"  Sentence {i + 1:2d}: {dt:.4f}s  ({frames} frames, {len(ids)} tokens)")

    pt_total = sum(pytorch_times)
    print(
        f"  PyTorch total: {pt_total:.4f}s  avg: {pt_total / len(pytorch_times):.4f}s\n"
    )

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  ONNX  (CPU):  total={onnx_total:.4f}s  avg={onnx_total / len(onnx_times):.4f}s"
    )
    print(
        f"  PyTorch (CPU): total={pt_total:.4f}s  avg={pt_total / len(pytorch_times):.4f}s"
    )
    if pt_total > 0:
        ratio = onnx_total / pt_total
        faster = "ONNX" if ratio < 1 else "PyTorch"
        speedup = max(ratio, 1 / ratio)
        print(f"  Winner: {faster} ({speedup:.2f}x faster)")
    print()


if __name__ == "__main__":
    main()
