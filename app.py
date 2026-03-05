import time
import base64
import os
import json
import re

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from commentator.data import fetch_stock_data, fetch_stock_info
from commentator.analysis import analyze_stock
from commentator.tts import iter_audio_chunks, pcm_chunks_to_wav
from commentator.commentary import generate_commentary

_APP_DEBUG = os.getenv("APP_DEBUG", "0") == "1"
_MAX_COMMENTARY_HISTORY = 50
_TICKER_SAFE_RE = re.compile(r"[^A-Z0-9.^\-]")

st.set_page_config(page_title="Stock Commentator", layout="wide")
st.title("Real-Time Stock Chart Commentator")

# --- Sidebar controls ---
with st.sidebar:
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    period = st.selectbox("Period", ["15m", "1d", "5d", "1mo", "3mo"], index=0)
    interval_options = {
        "15m": ["1m"],
        "1d": ["1m", "2m", "5m", "15m"],
        "5d": ["5m", "15m", "30m"],
        "1mo": ["30m", "1h", "1d"],
        "3mo": ["1d", "1wk"],
    }
    intervals = interval_options.get(period, ["1m"])
    default_interval = "15m" if "15m" in intervals else intervals[0]
    interval = st.selectbox(
        "Interval", intervals, index=intervals.index(default_interval)
    )
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        5,
        120,
        15,
        key="refresh_interval",
    )
    voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    voice = st.selectbox(
        "Commentator voice",
        voices,
        index=voices.index("leo"),
    )
    speed = st.slider("Speech speed", 1.0, 3.0, 1.4, step=0.05)
    use_tradingview = st.checkbox("Use TradingView embedded chart", value=False)

    live = st.toggle("LIVE MODE", value=False)

    if st.button("Single update now"):
        st.session_state["force_update"] = True

# --- Session state init ---
for key, default in [
    ("last_price", None),
    ("commentary_history", []),
    ("last_audio", None),
    ("live_active", False),
    ("audio_duration", 0.0),
    ("last_commentary_time", 0.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if period == "15m" and refresh_interval > 5:
    refresh_interval = 5
    st.caption("Refresh interval capped at 5s for 15m window.")

# --- Determine if we need fresh data ---
force = st.session_state.pop("force_update", False)
live_just_started = live and not st.session_state["live_active"]
live_wait = max(
    float(refresh_interval),
    float(st.session_state.get("audio_duration", 0.0)) + 1.0,
)
live_interval_elapsed = live and (
    time.time() - st.session_state["last_commentary_time"] >= live_wait
)
need_refresh = force or live_just_started or live_interval_elapsed or not live

# --- Fetch data (skip API calls during countdown-only reruns) ---
if need_refresh or "cached_df" not in st.session_state:
    df = fetch_stock_data(ticker, period=period, interval=interval)
    if df.empty:
        st.error(
            f"No data for **{ticker}**. Check the symbol or try when the market is open."
        )
        st.stop()
    info = fetch_stock_info(ticker)
    analysis = analyze_stock(df)
    if "error" in analysis:
        st.error(analysis["error"])
        st.stop()
    st.session_state["cached_df"] = df
    st.session_state["cached_analysis"] = analysis
    st.session_state["cached_company_name"] = info["name"]
else:
    df = st.session_state["cached_df"]
    analysis = st.session_state["cached_analysis"]

company_name = st.session_state["cached_company_name"]
current_price = analysis["current_price"]
last_price = st.session_state["last_price"]
price_changed = last_price is not None and last_price != current_price
need_commentary = force or (
    live and (price_changed or live_just_started or live_interval_elapsed)
)

# Track live movement for the commentator
if price_changed:
    move = current_price - last_price
    move_pct = (move / last_price) * 100
    analysis["live_move"] = round(move, 2)
    analysis["live_move_pct"] = round(move_pct, 3)
    analysis["live_direction"] = "up" if move > 0 else "down"

# --- Layout ---
col_chart, col_info = st.columns([2, 1])

with col_chart:
    st.subheader(f"{company_name} ({ticker})")
    if use_tradingview:
        interval_map = {
            "1m": "1",
            "2m": "2",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "1d": "D",
            "1wk": "W",
        }
        tv_config = {
            "allow_symbol_change": True,
            "calendar": False,
            "details": False,
            "hide_side_toolbar": True,
            "hide_top_toolbar": False,
            "hide_legend": False,
            "hide_volume": False,
            "hotlist": False,
            "interval": interval_map.get(interval, "D"),
            "locale": "en",
            "save_image": True,
            "style": "1",
            "symbol": ticker,
            "theme": "dark",
            "timezone": "Etc/UTC",
            "backgroundColor": "#0F0F0F",
            "gridColor": "rgba(242, 242, 242, 0.06)",
            "watchlist": [],
            "withdateranges": False,
            "compareSymbols": [],
            "studies": [],
            "width": "100%",
            "height": 700,
        }
        # Sanitize ticker to prevent XSS in the raw HTML embed.
        safe_ticker = _TICKER_SAFE_RE.sub("", ticker)
        components.html(
            f"""
<div class="tradingview-widget-container" style="height:740px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/{safe_ticker}/" rel="noopener nofollow" target="_blank"><span class="blue-text">{safe_ticker} chart</span></a><span class="trademark"> by TradingView</span></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {json.dumps(tv_config)}
  </script>
</div>
""",
            height=780,
            scrolling=False,
        )
    else:
        fig = go.Figure()

        # Candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

        # Volume as bar chart on secondary y-axis
        colors = [
            "#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.3,
                yaxis="y2",
            )
        )

        fig.update_layout(
            yaxis=dict(title="Price", side="left"),
            yaxis2=dict(title="Volume", side="right", overlaying="y", showgrid=False),
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date",
            ),
            template="plotly_dark",
            height=500,
            margin=dict(l=50, r=50, t=30, b=30),
            showlegend=False,
            xaxis_rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
            ],
        )

        st.plotly_chart(fig, width="stretch")

with col_info:
    st.subheader("Live Analysis")
    st.metric(
        "Price",
        f"${analysis['current_price']:.2f}",
        f"{analysis['price_change_pct']:+.2f}%",
    )
    st.markdown(f"**Trend:** {analysis['trend'].upper()}", unsafe_allow_html=True)
    st.markdown(f"**Range:** \\${analysis['low']:.2f} – \\${analysis['high']:.2f}", unsafe_allow_html=True)
    st.markdown(f"**Volume:** {analysis['volume_trend']}", unsafe_allow_html=True)
    st.markdown(f"**Volatility:** {analysis['volatility']}", unsafe_allow_html=True)
    if analysis["rsi"] is not None:
        st.markdown(f"**RSI:** {analysis['rsi']:.1f}", unsafe_allow_html=True)
    if analysis["sma_cross"]:
        st.markdown(f"**Signal:** {analysis['sma_cross'].replace('_', ' ').title()}", unsafe_allow_html=True)

# --- Commentary ---
st.divider()
st.subheader("Commentary")

if need_commentary:
    # Claim this cycle up front so failures do not trigger 1s retry loops.
    st.session_state["last_commentary_time"] = time.time()
    try:
        with st.spinner("The commentator is calling the action..."):
            commentary = generate_commentary(
                analysis,
                ticker,
                company_name,
                previous_commentary=st.session_state.get("commentary_history", []),
            )
    except Exception as exc:
        commentary = "The commentator is having technical difficulties!"
        st.error("Commentary generation failed.")
        if _APP_DEBUG:
            st.exception(exc)
    st.write(commentary)
    audio = None
    try:
        with st.spinner("Generating speech..."):
            progress = st.empty()
            audio_chunks: list[bytes] = []
            sample_rate = int(os.getenv("ORPHEUS_SAMPLE_RATE", "24000"))
            for chunk in iter_audio_chunks(commentary, voice=voice, speed=speed):
                audio_chunks.append(chunk)
                pcm_bytes_so_far = sum(len(part) for part in audio_chunks)
                seconds_so_far = pcm_bytes_so_far / (sample_rate * 2)
                progress.caption(f"Synthesizing... {seconds_so_far:.1f}s buffered")

            pcm_bytes = sum(len(chunk) for chunk in audio_chunks)
            audio_duration = max(pcm_bytes / (sample_rate * 2), 1)
            audio_duration += 0.75
            audio = (
                pcm_chunks_to_wav(audio_chunks, sample_rate=sample_rate)
                if audio_chunks
                else None
            )
            st.session_state["audio_duration"] = (
                audio_duration if audio else float(refresh_interval)
            )
            progress.empty()
    except Exception as exc:
        st.warning("TTS generation failed.")
        st.session_state["audio_duration"] = float(refresh_interval)
        if _APP_DEBUG:
            st.exception(exc)

    st.session_state["commentary_history"].append(commentary)
    # Cap history to prevent unbounded memory growth in long live sessions.
    if len(st.session_state["commentary_history"]) > _MAX_COMMENTARY_HISTORY:
        st.session_state["commentary_history"] = st.session_state["commentary_history"][-_MAX_COMMENTARY_HISTORY:]
    st.session_state["last_audio"] = audio
    st.session_state["last_price"] = current_price
    # Re-anchor timer at cycle end so long generations do not trigger rapid retries.
    st.session_state["last_commentary_time"] = time.time()

    if audio:
        b64 = base64.b64encode(audio).decode()
        uid = int(time.time() * 1000)
        st.html(
            f'<audio id="a{uid}" autoplay controls src="data:audio/wav;base64,{b64}"></audio>'
        )
    else:
        st.warning("TTS unavailable.")
elif st.session_state["commentary_history"]:
    st.write(st.session_state["commentary_history"][-1])
    audio = st.session_state["last_audio"]
    if audio:
        st.audio(audio, format="audio/wav")
else:
    st.info(
        "Toggle **LIVE MODE** or click **Single update now** to start the commentator."
    )

# --- Live mode auto-refresh ---
if live:
    st.session_state["live_active"] = True
    if st.session_state["last_price"] is None:
        st.session_state["last_price"] = current_price

    audio_dur = st.session_state.get("audio_duration", 0.0)
    wait = max(refresh_interval, audio_dur + 1)
    elapsed = time.time() - st.session_state["last_commentary_time"]
    remaining = max(0, wait - elapsed)
    st.sidebar.success(f"LIVE — next update in {int(remaining)}s")
    # Keep the app process running in live mode and trigger a timed rerun.
    time.sleep(remaining)
    st.rerun()
else:
    st.session_state["live_active"] = False
