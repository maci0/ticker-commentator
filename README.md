# Ticker Commentator

Real-time stock chart commentary with a sports announcer voice. Uses llama.cpp GGUF models for commentary + Orpheus token generation, then decodes audio locally with SNAC.

## Quickstart

```bash
cp .env.example .env   # edit as needed
uv sync
uv run streamlit run app.py
```

Optional debug run:

```bash
APP_DEBUG=1 COMMENTARY_DEBUG=1 ORPHEUS_TTS_DEBUG=1 uv run streamlit run app.py
```

## Architecture

Pipeline:

`User Input -> yfinance -> pandas analysis -> llama.cpp commentary -> emotion tag injection -> llama.cpp Orpheus tokens -> SNAC decode -> Streamlit audio playback`

Core files:

- `app.py` - Streamlit UI, loads `.env` via `python-dotenv`, live refresh loop, chart rendering (Plotly), commentary + audio playback.
- `commentator/data.py` - yfinance fetch helpers with logged errors.
- `commentator/analysis.py` - technical summary (trend, RSI, SMA cross, volatility, volume trend).
- `commentator/commentary.py` - sports-style one-liner generation with llama.cpp GGUF. Feeds back up to 5 prior lines for variety. Injects Orpheus emotion tags deterministically based on market sentiment.
- `commentator/tts.py` - Orpheus token generation with llama.cpp + chunked SNAC decode.
- `.env.example` - all configuration variables with defaults and comments.

## Runtime Notes

- First run downloads GGUF models and SNAC weights from Hugging Face.
- Use `HF_TOKEN` to improve Hugging Face download reliability and rate limits.
- On ROCm systems, commentary/TTS are pinned to a single GPU by default to avoid multi-device instability.

## Test Script

```bash
uv run python experiments/test_tts.py
```

Custom text/voice:

```bash
uv run python experiments/test_tts.py --text "Test line" --voice tara --out /tmp/orpheus_test.wav
```

## Configuration

Copy `.env.example` to `.env` and edit as needed. The app loads it automatically via `python-dotenv`. See `.env.example` for the full list of variables with defaults and comments.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `APP_DEBUG` | `0` | Streamlit exception tracebacks |
| `COMMENTARY_DEBUG` | `0` | llama.cpp commentary logs |
| `ORPHEUS_TTS_DEBUG` | `0` | TTS token/decode debug logs |
| `COMMENTARY_GPU_LAYERS` | `-1` | GPU layers for commentary model (`-1` = all, `0` = CPU) |
| `ORPHEUS_LLAMA_GPU_LAYERS` | `-1` | GPU layers for TTS model (`-1` = all, `0` = CPU) |
| `ORPHEUS_TTS_MAX_SECONDS` | `60` | Timeout to stop Orpheus repetition |

## Troubleshooting

- **Commentary hangs/crashes after model load:**
  - Try single-GPU pinning explicitly:
    - `COMMENTARY_MAIN_GPU=0`
    - `COMMENTARY_TENSOR_SPLIT=1.0`
  - Reduce GPU pressure:
    - `COMMENTARY_GPU_LAYERS=0` (recommended on ROCm when TTS is GPU)
    - `COMMENTARY_GPU_LAYERS=20` (optional partial offload)
- **No audio output:**
  - Enable logs: `ORPHEUS_TTS_DEBUG=1`
  - Verify llama-cpp-python is ROCm-enabled and model loads without errors.
- **Need stack traces in UI:**
  - Run with `APP_DEBUG=1 COMMENTARY_DEBUG=1`.
- **`n_ctx_per_seq < n_ctx_train` warning:**
  - Harmless. The 4096 context window is more than enough for commentary prompts.
