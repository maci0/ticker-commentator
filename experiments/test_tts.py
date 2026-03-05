import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commentator.tts import text_to_speech


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Orpheus TTS via Ollama tokens (PyTorch SNAC decode)"
    )
    parser.add_argument(
        "--text",
        default="Hello there, this is a quick Orpheus TTS test.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        default="zac",
        help="Voice name (tara, leah, jess, leo, dan, mia, zac, zoe)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.3,
        help="Speech speed multiplier (0.8 to 1.4)",
    )
    parser.add_argument(
        "--out",
        default="orpheus_test.wav",
        help="Output WAV path",
    )
    args = parser.parse_args()

    audio = text_to_speech(args.text, voice=args.voice, speed=args.speed)
    if not audio:
        print("TTS failed: no audio returned")
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
