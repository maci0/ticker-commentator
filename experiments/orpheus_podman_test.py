#!/usr/bin/env python3
"""Minimal Orpheus GGUF load test using Podman + official llama.cpp ROCm image."""

from __future__ import annotations

import shlex
import subprocess
import time

from huggingface_hub import hf_hub_download

IMAGE = "ghcr.io/ggml-org/llama.cpp:full-rocm"
GGUF_REPO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
GGUF_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"
PROMPT = "<|audio|>tara: hello from podman rocm test<|eot_id|><custom_token_4>"


def main() -> int:
    print("Downloading/locating Orpheus GGUF...")
    model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

    model_dir = str(model_path.rsplit("/", 1)[0])
    model_file = model_path.rsplit("/", 1)[1]

    cmd = [
        "podman",
        "run",
        "--rm",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "keep-groups",
        "-v",
        f"{model_dir}:/models:ro",
        IMAGE,
        "llama-cli",
        "-m",
        f"/models/{model_file}",
        "-ngl",
        "999",
        "-n",
        "96",
        "--no-display-prompt",
        "-p",
        PROMPT,
    ]

    print("Running:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    print("\n--- llama.cpp output ---")

    start = time.monotonic()
    result = subprocess.run(cmd, text=True)
    elapsed = time.monotonic() - start

    print("--- end output ---")
    print(f"Exit code: {result.returncode}")
    print(f"Elapsed: {elapsed:.2f}s")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
