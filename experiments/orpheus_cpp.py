#!/usr/bin/env python3
"""Minimal Orpheus TTS via llama-cpp-python + PyTorch SNAC."""

import argparse
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from snac import SNAC

# --- Config ---
GGUF_REPO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
GGUF_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"
VOICE = "tara"
TEXT = "And here comes Apple Inc., folks, mired in a neutral zone, trading at $275.92!"
SAMPLE_RATE = 24000
CHUNK_FRAMES = 24
VALID_VOICES = ("dan", "jess", "leo", "leah", "mia", "tara", "zac", "zoe")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orpheus TTS with optional aplay output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    aplay_group = parser.add_mutually_exclusive_group()
    aplay_group.add_argument(
        "--aplay",
        action="store_true",
        help="Pipe generated WAV into aplay after synthesis",
    )
    aplay_group.add_argument(
        "--aplay-live",
        action="store_true",
        help="Stream decoded PCM chunks directly to aplay while generating",
    )
    parser.add_argument(
        "--voice",
        default=VOICE,
        choices=sorted(VALID_VOICES),
        help="Voice name",
    )
    parser.add_argument(
        "--text",
        default=TEXT,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--out",
        default="orpheus_cpp_output.wav",
        help="Output WAV path",
    )
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("--text must not be empty")

    # --- Load models ---
    print("Loading GGUF model...")
    gguf_path = hf_hub_download(GGUF_REPO, GGUF_FILE)
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_batch=64,
        main_gpu=0,
        tensor_split=[1.0],
        verbose=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SNAC model on {device}...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    snac_model.eval()

    # --- Format prompt ---
    prompt = f"<|audio|>{args.voice}: {args.text}<|eot_id|><custom_token_4>"

    # --- Generate tokens ---
    print("Generating tokens...")
    start = time.monotonic()

    token_ids: list[int] = []
    count = 0
    token_re = re.compile(r"<custom_token_(\d+)>")
    out_path = str(Path(args.out).expanduser().resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    live_aplay: subprocess.Popen[bytes] | None = None
    if args.aplay_live:
        if shutil.which("aplay") is None:
            print("ERROR: aplay not found on PATH", file=sys.stderr)
            return 1
        live_aplay = subprocess.Popen(
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE)],
            stdin=subprocess.PIPE,
        )

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        def decode_and_write(frame_tokens: list[int]) -> int:
            num_frames = len(frame_tokens) // 7
            if num_frames < 4:
                return 0
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
            codes = [
                torch.from_numpy(codes_0.reshape(1, -1)).long().to(device),
                torch.from_numpy(codes_1.reshape(1, -1)).long().to(device),
                torch.from_numpy(codes_2.reshape(1, -1)).long().to(device),
            ]
            with torch.no_grad():
                audio_hat = snac_model.decode(codes).cpu().numpy()
            audio_int16 = (audio_hat.flatten() * 32767).astype(np.int16)
            pcm_bytes = audio_int16.tobytes()
            wf.writeframes(pcm_bytes)
            if live_aplay is not None and live_aplay.stdin is not None:
                live_aplay.stdin.write(pcm_bytes)
                live_aplay.stdin.flush()
            return int(audio_int16.size)

        print("Streaming decode with SNAC...")
        decode_start = time.monotonic()
        pending: list[int] = []
        buffer = ""
        total_samples = 0

        for tok in llm(
            prompt,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
        ):
            tok = cast(dict[str, Any], tok)
            text = str(tok["choices"][0]["text"])
            if not text:
                continue
            buffer += text
            while True:
                match = token_re.search(buffer)
                if not match:
                    if len(buffer) > 128:
                        buffer = buffer[-128:]
                    break
                raw = int(match.group(1))
                num = raw - 10 - ((count % 7) * 4096)
                if num > 0:
                    token_ids.append(num)
                    pending.append(num)
                    count += 1
                    if len(pending) >= CHUNK_FRAMES * 7:
                        total_samples += decode_and_write(pending[: CHUNK_FRAMES * 7])
                        pending = pending[CHUNK_FRAMES * 7 :]
                buffer = buffer[match.end() :]

        while len(pending) >= 4 * 7:
            take = min(len(pending) // 7, CHUNK_FRAMES) * 7
            total_samples += decode_and_write(pending[:take])
            pending = pending[take:]

    gen_time = time.monotonic() - start
    decode_time = time.monotonic() - decode_start
    duration = total_samples / SAMPLE_RATE
    total = time.monotonic() - start
    print(f"  {count} tokens in {gen_time:.2f}s")
    print(f"  Decode: {decode_time:.2f}s")
    print(f"  Total: {total:.2f}s for {duration:.2f}s of audio")
    print(f"  Saved: {out_path}")

    if live_aplay is not None:
        if live_aplay.stdin is not None:
            live_aplay.stdin.close()
        rc = live_aplay.wait()
        if rc != 0:
            print(f"ERROR: live aplay failed with code {rc}", file=sys.stderr)
            return rc

    if args.aplay:
        if shutil.which("aplay") is None:
            print("ERROR: aplay not found on PATH", file=sys.stderr)
            return 1
        print("Playing via aplay...")
        with open(out_path, "rb") as audio_file:
            play_result = subprocess.run(["aplay", "-q"], stdin=audio_file)
        if play_result.returncode != 0:
            print(f"ERROR: aplay failed with code {play_result.returncode}", file=sys.stderr)
            return play_result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
