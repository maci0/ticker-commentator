# Real-Time Stock Chart Commentator — Implementation Plan

Historical checklist. For current architecture see `README.md` and `CLAUDE.md`.

## Architecture (current)

```
User Input → yfinance → pandas analysis → llama.cpp commentary (Qwen3.5 GGUF)
  → Orpheus tokens (llama.cpp) → SNAC decode → Streamlit playback
```

Optional: `TTS_ENGINE=chatterbox|kokoro|qwen` (see `docs/tts_engines.md`).

## Tech stack

| Component | Choice |
|---|---|
| Package manager | `uv` + hatchling (installable package) |
| Stock data | `yfinance` |
| Charting | Plotly + optional TradingView embed |
| Analysis | pandas on OHLCV |
| Commentary | llama-cpp-python GGUF (default Qwen3.5-4B) |
| TTS | Orpheus GGUF + SNAC; optional alt engines |
| UI | Streamlit |

## Project layout

```
ticker-commentator/
├── app.py                     # Streamlit shell (session, live loop)
├── commentator/
│   ├── data.py                # yfinance
│   ├── analysis.py            # technicals
│   ├── commentary.py          # personas + LLM
│   ├── tts.py                 # Orpheus + SNAC
│   ├── tts_engines.py         # optional alt TTS
│   ├── llama_loader.py        # shared GGUF load
│   ├── llama_lock.py
│   ├── audio_server.py        # streaming WAV
│   ├── playback.py / charts.py / ui_html.py / live.py / prefetch.py
│   └── _config.py             # typed env helpers
├── tests/
├── .env.example
├── pyproject.toml
└── docs/
```

## Done

- [x] Data + analysis modules with TypedDict results
- [x] Commentary personas, emotion tags, anti-repetition
- [x] Orpheus TTS streaming decode + alt engines
- [x] Live mode, prefetch, streaming audio
- [x] Pure UI helpers extracted from `app.py`
- [x] Shared `LazyLlama` loader
- [x] Unit + hypothesis tests (CPU-only via blanked GPU devices)
- [x] Ruff + pytest CI

## Setup / run

```bash
cp .env.example .env
uv sync                 # CPU / NVIDIA; AMD ROCm: ./setup.sh
uv run streamlit run app.py
make check              # lint + format + tests
```

No Ollama required — inference is llama-cpp-python only.
