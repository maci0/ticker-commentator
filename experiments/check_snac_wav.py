#!/usr/bin/env python3
"""Generate a WAV file using PyTorch SNAC decode to check output quality.

Streams tokens from Ollama/Orpheus for one sentence, decodes with both
ONNX and PyTorch SNAC, and saves WAV files for comparison.

Usage:
    uv run python scripts/check_snac_wav.py
"""

import http.client
import io
import json
import os
import urllib.parse
import wave

import numpy as np
import onnxruntime
import torch
from huggingface_hub import hf_hub_download
from snac import SNAC

OLLAMA_HOST = os.getenv("ORPHEUS_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("ORPHEUS_MODEL_NAME", "legraphista/Orpheus:3b-ft-q8")
VOICE = "zac"
SENTENCE = "What an incredible move in the market today, absolutely electric!"
SAMPLE_RATE = 24000
CUSTOM_TOKEN_PREFIX = "<custom_token_"


def _ollama_stream(prompt: str) -> list[str]:
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
        raise RuntimeError(resp.read().decode("utf-8", errors="ignore").strip())

    tokens: list[str] = []
    buffer = ""
    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        data = json.loads(line)
        if data.get("response"):
            buffer += data["response"]
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


def _parse_token_ids(raw_tokens: list[str]) -> list[int]:
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


def _reshape_numpy(token_ids: list[int]):
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


def _save_wav(path: str, audio: np.ndarray):
    # Flatten and convert to int16
    audio = audio.flatten()
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    duration = len(audio_int16) / SAMPLE_RATE
    print(f"  Saved {path} ({duration:.2f}s, {os.path.getsize(path)} bytes)")


def main():
    print(f"Sentence: {SENTENCE}\n")

    # Stream tokens
    print("Streaming tokens from Ollama...")
    prompt = f"<|audio|>{VOICE}: {SENTENCE}<|eot_id|><custom_token_4>"
    raw_tokens = _ollama_stream(prompt)
    token_ids = _parse_token_ids(raw_tokens)
    print(
        f"  {len(raw_tokens)} raw tokens → {len(token_ids)} valid IDs "
        f"({len(token_ids) // 7} frames)\n"
    )

    if len(token_ids) < 7:
        print("ERROR: Not enough tokens to decode.")
        return

    codes_np = _reshape_numpy(token_ids)

    # ONNX decode
    print("Decoding with ONNX...")
    onnx_path = hf_hub_download(
        "onnx-community/snac_24khz-ONNX",
        subfolder="onnx",
        filename="decoder_model.onnx",
    )
    session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    input_names = [x.name for x in session.get_inputs()]
    input_dict = dict(zip(input_names, codes_np))
    onnx_audio = session.run(None, input_dict)[0]
    _save_wav("onnx_output.wav", onnx_audio)

    # PyTorch decode
    print("Decoding with PyTorch...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model.eval()
    codes_torch = [torch.from_numpy(c).long() for c in codes_np]
    with torch.no_grad():
        pt_audio = model.decode(codes_torch).numpy()
    _save_wav("pytorch_output.wav", pt_audio)

    print("\nDone! Compare the two files:")
    print("  aplay onnx_output.wav")
    print("  aplay pytorch_output.wav")


if __name__ == "__main__":
    main()
