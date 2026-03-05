# Ticker Commentator

Real-time stock chart commentary with a sports announcer voice. Uses llama.cpp GGUF models for commentary + Orpheus token generation, then decodes audio locally with SNAC.

## Quickstart

```bash
cp .env.example .env   # edit as needed
uv sync
uv run streamlit run app.py
```

Debug mode:

```bash
APP_DEBUG=1 COMMENTARY_DEBUG=1 ORPHEUS_TTS_DEBUG=1 uv run streamlit run app.py
```

## Architecture

Pipeline:

`User Input → yfinance → pandas analysis → llama.cpp commentary → emotion tag injection → llama.cpp Orpheus tokens → SNAC decode → Streamlit audio playback`

Core files:

- `app.py` — Streamlit UI with live refresh, Plotly/TradingView charts, auto-playing audio.
- `commentator/data.py` — yfinance fetch with ticker validation and logged errors.
- `commentator/analysis.py` — Technical summary: trend, RSI(14), SMA 20/50 cross, ATR volatility, volume trend. Returns typed `AnalysisResult` dict.
- `commentator/commentary.py` — Sports-style one-liner generation via llama.cpp GGUF. Injects Orpheus emotion tags based on market sentiment.
- `commentator/tts.py` — Orpheus token generation via llama.cpp + chunked SNAC decode to PCM audio.
- `commentator/llama_lock.py` — Shared threading lock for llama.cpp serialization.

## Testing

```bash
uv run python -m pytest tests/ -v
uv run python -m pytest tests/ -v --cov=commentator --cov-report=term-missing
```

## Configuration

Copy `.env.example` to `.env` and edit as needed. See `.env.example` for all variables with defaults.

| Variable | Default | Description |
|---|---|---|
| `APP_DEBUG` | `0` | Streamlit exception tracebacks |
| `COMMENTARY_DEBUG` | `0` | llama.cpp commentary logs |
| `ORPHEUS_TTS_DEBUG` | `0` | TTS token/decode debug logs |
| `COMMENTARY_GPU_LAYERS` | `-1` | GPU layers for commentary model (`-1` = all, `0` = CPU) |
| `ORPHEUS_LLAMA_GPU_LAYERS` | `-1` | GPU layers for TTS model (`-1` = all, `0` = CPU) |
| `ORPHEUS_TTS_MAX_SECONDS` | `60` | Timeout to stop Orpheus repetition |

## Runtime Notes

- First run downloads GGUF models and SNAC weights from Hugging Face.
- Use `HF_TOKEN` for faster downloads and higher rate limits.
- On ROCm systems, models are pinned to a single GPU by default.

## Troubleshooting

- **Commentary hangs/crashes**: Try `COMMENTARY_GPU_LAYERS=0` (CPU fallback) or explicit GPU pinning via `COMMENTARY_MAIN_GPU=0`.
- **No audio**: Enable `ORPHEUS_TTS_DEBUG=1` and check logs.
- **Stack traces in UI**: Run with `APP_DEBUG=1`.
- **`n_ctx_per_seq < n_ctx_train` warning**: Harmless — 4096 context is sufficient for commentary prompts.
