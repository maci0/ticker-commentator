# Ticker Commentator

Real-time stock chart commentary with selectable announcer personas. Uses llama.cpp
GGUF models for commentary + Orpheus token generation, then decodes audio locally
with SNAC.

## Quickstart

```bash
cp .env.example .env   # edit as needed
uv sync                # AMD ROCm GPU: ./setup.sh instead
uv run streamlit run app.py
```

Debug mode:

```bash
APP_DEBUG=1 COMMENTARY_DEBUG=1 ORPHEUS_TTS_DEBUG=1 uv run streamlit run app.py
```

## Architecture

```
User Input → yfinance → pandas analysis → llama.cpp commentary
  → emotion tag injection → llama.cpp Orpheus tokens → SNAC decode
  → Streamlit / streaming WAV playback
```

Core modules:

- `app.py` — Streamlit shell (session state, live loop, widgets)
- `commentator/data.py` — yfinance fetch + ticker validation
- `commentator/analysis.py` — trend, RSI, SMA cross, ATR volatility, volume
- `commentator/commentary.py` — persona prompts + llama.cpp one-liners
- `commentator/tts.py` — Orpheus tokens + SNAC PCM stream
- `commentator/llama_loader.py` — shared lazy GGUF load (flash-attn fallback)
- `commentator/charts.py` / `playback.py` / `ui_html.py` — pure UI helpers
- `commentator/audio_server.py` / `prefetch.py` / `live.py` — streaming + live logic
- `commentator/_config.py` — typed env helpers (`env_int`, `env_bool`, …)

## Testing

```bash
make check             # ruff lint + format check + pytest with coverage
uv run pytest -v
```

## Configuration

Copy `.env.example` → `.env`. Full list with comments is in `.env.example`. Common knobs:

| Variable | Default | Description |
|---|---|---|
| `APP_DEBUG` | `0` | Verbose app logs / UI exception tracebacks |
| `COMMENTARY_DEBUG` | `0` | Commentary llama.cpp verbose |
| `ORPHEUS_TTS_DEBUG` | `0` | Orpheus token/decode debug |
| `COMMENTARY_PERSONALITY` | `sports` | Default persona (15 options) |
| `COMMENTARY_GGUF_REPO` / `FILE` | Qwen3.5-4B Q8 | Commentary model |
| `COMMENTARY_GPU_LAYERS` | `-1` | `-1` all GPU, `0` CPU |
| `ORPHEUS_LLAMA_GPU_LAYERS` | `-1` | TTS GPU layers |
| `ORPHEUS_TTS_MAX_SECONDS` | `60` | Stop runaway Orpheus decode |
| `TTS_ENGINE` | `orpheus` | `orpheus` \| `chatterbox` \| `kokoro` \| `qwen` |
| `LIVE_PREFETCH` | `1` | Speculative next-line generation |
| `LIVE_PREFETCH_TOLERANCE_PCT` | `0.05` | Reuse prefetch under this move % |
| `STREAM_AUDIO` | `1` | Local 127.0.0.1 streaming WAV (set `0` if remote) |
| `GPU_DEVICE` | `0` | HIP/CUDA visible device before model init |

## Runtime notes

- First run downloads GGUF models and SNAC weights from Hugging Face (`HF_TOKEN` helps).
- On ROCm, `./setup.sh` builds llama-cpp-python with HIP; plain `uv sync` is CPU llama.
- `make audit` scans non-torch deps (ROCm torch pins are not on PyPI for pip-audit).

## Troubleshooting

- **Commentary hangs/crashes**: `COMMENTARY_GPU_LAYERS=0` or `GPU_DEVICE=0`
- **No audio**: `ORPHEUS_TTS_DEBUG=1` and check logs; try `STREAM_AUDIO=0`
- **Remote Streamlit**: set `STREAM_AUDIO=0` (browser cannot reach host 127.0.0.1)
- **`n_ctx_per_seq < n_ctx_train`**: harmless for short prompts
