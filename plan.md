# Real-Time Stock Chart Commentator - Implementation Plan

## Architecture Overview

A Streamlit web app that fetches stock data, analyzes trends, generates sports-style comedy commentary via Ollama (LLM), and speaks it aloud via Orpheus TTS (tokens + SNAC decode).

```
User Input (ticker) → yfinance (data) → Chart + Analysis
    → Ollama LLM (commentary text) → Orpheus tokens → SNAC decode → Streamlit playback
```

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Package manager | `uv` | User preference |
| Stock data | `yfinance` | Free, no API key needed |
| Charting | `plotly` | Interactive candlestick charts with zoom/hover |
| Chart analysis | Direct data analysis (pandas) | More reliable than CV on charts |
| LLM | llama.cpp (Llama 3.2 3B GGUF) | Local inference via llama-cpp-python |
| TTS | Orpheus (llama.cpp + SNAC) | Local, high quality, open source |
| UI | Streamlit | Web-based, charts + audio playback |

## Key Design Decisions

### Chart Analysis
Instead of using computer vision (CLIP/YOLO) on rendered chart images — which is fragile and adds complexity — we analyze the raw data directly with pandas to extract:
- [x] Price trend (up/down/sideways)
- [x] Percentage change
- [x] Volume spikes
- [x] Moving average crossovers
- [x] Volatility (ATR)
- [x] RSI

### Python Version
- [x] Pinned to `>=3.12` for torch + snac compatibility

## Project Structure

```
ticker-commentator/
├── pyproject.toml          # uv project with dependencies
├── app.py                  # Streamlit main app
├── commentator/
│   ├── __init__.py
│   ├── data.py             # Stock data fetching (yfinance)
│   ├── analysis.py         # Technical analysis (pandas)
│   ├── commentary.py       # LLM commentary generation (llama.cpp) + emotion tag injection
│   └── tts.py              # Text-to-speech (Orpheus tokens via llama.cpp + SNAC)
├── .env.example            # All configuration variables with defaults
├── CLAUDE.md               # Claude Code guidance
├── design.md               # Original spec
└── plan.md                 # This plan
```

## Implementation Steps

### Step 1: Project Setup
- [x] Initialize uv project with `uv init`
- [x] Add dependencies: `yfinance`, `plotly`, `ollama`, `snac`, `torch`, `streamlit`, `pandas`, `requests`
- [x] Pin Python to 3.12 (`uv python pin 3.12`, `requires-python = ">=3.12,<3.14"`)
- [x] Install pip in venv for Kokoro (`uv pip install pip`)

### Step 2: Stock Data Module (`commentator/data.py`)
- [x] `fetch_stock_data(ticker, period, interval)` → DataFrame
- [x] `fetch_stock_info(ticker)` → dict with company name, sector, etc.
- [x] Handle invalid tickers and market closures gracefully

### Step 3: Technical Analysis Module (`commentator/analysis.py`)
- [x] Trend detection (bullish/bearish/sideways)
- [x] Price change percentage
- [x] Volume trend (heavy/light/normal)
- [x] SMA 20/50 crossover detection (golden cross / death cross)
- [x] RSI (14-period)
- [x] ATR-based volatility classification

### Step 4: Commentary Generation (`commentator/commentary.py`)
- [x] System prompt: sports commentator + WWE announcer style
- [x] llama.cpp GGUF inference via llama-cpp-python
- [x] Play-by-play mode: focuses on live price movement between refreshes
- [x] Opening commentary mode: for first load without movement data
- [x] No all-caps rule to avoid TTS spelling issues
- [x] Feeds back last 5 commentary lines to avoid repetition
- [x] Deterministic Orpheus emotion tag injection based on market sentiment (trend, volatility, change %)

### Step 5: TTS Module (`commentator/tts.py`)
- [x] Generate Orpheus tokens via llama.cpp GGUF
- [x] Decode tokens to WAV with SNAC (24kHz)
- [x] Configurable voice (`tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`)
- [x] Configurable speech speed

### Step 6: Streamlit App (`app.py`)
- [x] Ticker input, period/interval selectors
- [x] Interactive Plotly candlestick chart with volume overlay
- [x] Live analysis panel (price, trend, range, volume, volatility, RSI, SMA signals)
- [x] Commentary text display
- [x] Audio playback with autoplay on fresh commentary
- [x] Live mode toggle with configurable refresh (5-120s)
- [x] Single update button for manual triggers
- [x] No auto-generation on initial page load — user must start it
- [x] Refresh waits for audio to finish before rerunning
- [x] Voice and speed controls in sidebar

## Setup Prerequisites
1. `ollama serve` must be running
2. Pull a model: `ollama pull legraphista/Orpheus:3b-ft-q8`
3. First run downloads SNAC via Hugging Face (set `HF_TOKEN` for faster downloads)

## Running
```bash
uv run streamlit run app.py
```
