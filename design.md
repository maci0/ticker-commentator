# AI App Specification: Real-Time Stock Chart Commentator

## Overview

Local Streamlit app that fetches stock OHLCV data, runs pandas technical analysis,
generates a one-line persona commentary via llama.cpp (GGUF), injects optional Orpheus
emotion tags, then synthesizes speech (default: Orpheus tokens + SNAC decode) for
in-browser playback. No cloud LLM/TTS APIs.

## Pipeline

```
User Input → yfinance → pandas analysis → llama.cpp commentary
  → emotion tags → llama.cpp Orpheus tokens → SNAC decode
  → Streamlit / streaming WAV playback
```

## Key Features

- **Input**: ticker, period, interval; optional TradingView embed vs Plotly chart
- **Analysis**: trend, % change, volume trend, SMA 20/50 cross, RSI(14), ATR volatility
- **Commentary**: 15 selectable personas (sports, neutral, kramer, …); last 5 lines fed
  back for variety; similarity guard against live-mode repeats
- **TTS**: Orpheus (default) with optional `chatterbox` / `kokoro` / `qwen` engines
- **Live mode**: timer refresh, audio-duration wait, speculative prefetch, streaming audio
- **Local-only**: models via Hugging Face GGUF + SNAC; audio stream server binds 127.0.0.1

## Models & Tools

| Role | Default |
|---|---|
| Commentary LLM | Qwen3.5-4B GGUF via llama-cpp-python |
| TTS | Orpheus-3B GGUF tokens + SNAC 24 kHz |
| UI | Streamlit + Plotly |

## Technical Requirements

- Python 3.12; install with `uv` (`uv sync`; AMD ROCm: `./setup.sh`)
- GPU optional; `GPU_DEVICE` pins HIP/CUDA before torch/llama init
- Shared `LazyLlama` loader for commentary + Orpheus (flash-attn fallback, warmup)
- All processing local; no proprietary APIs

## Edge Cases

- Invalid tickers / empty history / analysis errors → clear UI errors, no crash
- Market closed / thin data → still narrates available bars (no session gate yet)
- Streaming bind failure → silent fallback to full-WAV autoplay
- Missing alt TTS packages → RuntimeError with install hint

## Out of scope (current)

- Multi-ticker watchlists
- Market-hours-aware prompting
- Hosted Streamlit with remote audio streaming (set `STREAM_AUDIO=0`)
