AI App Specification: Real-Time Stock Chart Commentator
Overview
Develop an AI application that analyzes real-time stock data and generates audio commentary in the style of a sports announcer, incorporating comedic elements (puns, exaggerated reactions). The app runs locally on CPU/GPU using open source models and Ollama for LLM inference.

Key Features

Input Handling: Fetch and display real-time stock chart data from Yahoo Finance via `yfinance`. Allow user to input stock ticker symbol.
Chart Analysis: Analyze raw OHLCV data (trend, moving averages, RSI, volatility) directly with pandas.
Commentary Generation:
Employ an LLM (Llama 3.2 3B via llama.cpp) to create narrative text based on analysis.
Style: Energetic sports commentary with comedy.
Orpheus emotion tags (`<laugh>`, `<sigh>`, `<gasp>`, etc.) are injected deterministically based on market sentiment after LLM generation.
Prior commentary (up to 5 lines) is fed back to the LLM to avoid repetition.
Output: Convert text to audio using Orpheus TTS by generating tokens via llama.cpp and decoding locally with SNAC.

Real-Time Processing: Update commentary every 1-5 minutes during market hours; handle streaming data.
User Interface: Web-based UI via Streamlit for input, chart display, and audio playback.

Technical Requirements

Hardware: Optimize for local CPU or GPU; fallback to CPU if GPU unavailable.
Models and Tools:
LLM: Open source Llama 3.2 3B (GGUF) via llama-cpp-python.
TTS: Orpheus tokens from llama.cpp + SNAC decoder in-app (PyTorch).

Dependencies: Python-based; no proprietary APIs/models. Install via `uv` and pip-compatible packages.
Performance: Keep updates responsive; decode audio locally for playback.
Security/Privacy: All processing local; no cloud uploads.

Development Steps

Set up environment with Ollama.
Implement data fetch and chart rendering.
Analyze raw data in pandas (no chart CV).
Generate commentary prompt for LLM.
Decode Orpheus tokens to audio and play.
Test with sample tickers (e.g., AAPL).

Edge Cases

Handle market closures, data errors, or invalid tickers with graceful fallbacks (e.g., "Game's postponed!").
Support multiple tickers or custom indicators if time allows.
