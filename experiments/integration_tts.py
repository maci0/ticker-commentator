#!/usr/bin/env python3
"""Test Orpheus TTS via llama.cpp (PyTorch SNAC decode)."""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commentator.tts import VALID_VOICES, text_to_speech


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Orpheus TTS via llama.cpp (PyTorch SNAC decode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        default="Hello there, this is a quick Orpheus TTS test.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        default="zac",
        choices=sorted(VALID_VOICES),
        help="Voice name",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.3,
        metavar="SPEED",
        help="Speech speed multiplier (0.8–1.4)",
    )
    parser.add_argument(
        "--out",
        default="orpheus_test.wav",
        help="Output WAV path",
    )
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("--text must not be empty")

    if not (0.8 <= args.speed <= 1.4):
        parser.error(f"--speed must be between 0.8 and 1.4, got {args.speed}")

    audio = text_to_speech(args.text, voice=args.voice, speed=args.speed)
    if not audio:
        print("TTS failed: no audio returned", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(audio)

    print(f"Wrote {len(audio)} bytes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
