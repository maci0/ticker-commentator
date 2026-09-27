"""Streamlit UI for the real-time stock chart commentator.

Manages Streamlit session state, renders charts, triggers commentary and TTS,
and drives the live-mode auto-refresh loop. .env is loaded by
commentator/__init__.py on first import.
"""

import logging
import re
import time

import streamlit as st
import streamlit.components.v1 as components

from commentator import (
    SAMPLE_RATE,
    VALID_VOICES,
    analyze_stock,
    audio_server,
    available_personalities,
    default_speed_for,
    default_voice_for,
    fetch_stock_data,
    fetch_stock_info,
    generate_commentary,
    iter_audio_chunks,
    pcm_chunks_to_wav,
    prefetch,
)
from commentator._config import env_abs_float, env_bool
from commentator.charts import build_candlestick_figure, tradingview_embed_html
from commentator.live import (
    clean_commentary,
    is_significant_move,
    prefetch_key,
    refresh_bounds,
    should_comment,
    should_refresh,
)
from commentator.playback import (
    data_uri_audio_html,
    streaming_audio_html,
    synthesize_wav,
)
from commentator.ui_html import (
    commentary_card_html,
    persona_blurb,
    persona_label,
)

_APP_DEBUG = env_bool("APP_DEBUG", False)
# Speculative prefetch: during live-mode audio playback, generate the next
# (unchanged-data) update in a background thread so an interval refresh that
# finds the same price serves instantly instead of blocking on regeneration.
# On by default; set LIVE_PREFETCH=0 to restore plain regenerate-each-cycle.
_LIVE_PREFETCH = env_bool("LIVE_PREFETCH", True)
# A live tick whose move is smaller than this (percent) reuses the prefetched
# no-move line instead of regenerating. Exact-price matching would almost never
# hit on a liquid stock, so the prefetch only pays off with a small tolerance.
# Moves at/above it still get fresh, move-aware commentary.
_LIVE_PREFETCH_TOL = env_abs_float("LIVE_PREFETCH_TOLERANCE_PCT", 0.05)
# Stream audio to the player as it decodes (time-to-first-audio ~1s vs ~4s) via a
# local HTTP server. On by default for local use; the server binds 127.0.0.1, so
# set STREAM_AUDIO=0 for remote/hosted Streamlit (the browser can't reach it).
# Falls back to full-WAV autoplay automatically if the port can't bind.
_STREAM_AUDIO = env_bool("STREAM_AUDIO", True)

logging.basicConfig(
    level=logging.DEBUG if _APP_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("app startup debug=%s", _APP_DEBUG)

# Session state history cap; only the last 5 entries are sent to the LLM (see commentary.py).
_MAX_COMMENTARY_HISTORY = 50
# Strips any character not in a standard ticker before HTML embedding; json.dumps does not
# escape < or >, so a raw ticker could break the TradingView <script> block (XSS).
_TICKER_SAFE_RE = re.compile(r"[^A-Z0-9.^=\-]")

st.set_page_config(page_title="Stock Commentator", page_icon="📈", layout="wide")


def _render_commentary_card(text: str, personality: str, trend: str, *, live: bool = False) -> None:
    st.markdown(
        commentary_card_html(text, personality, trend, live=live),
        unsafe_allow_html=True,
    )


def _render_audio(audio: bytes, *, autoplay: bool = False) -> None:
    st.html(data_uri_audio_html(audio, safe_ticker, autoplay=autoplay))


def _render_streaming_audio(stream_id: str, port: int) -> None:
    st.html(streaming_audio_html(stream_id, port, safe_ticker))


# --- Sidebar controls ---
with st.sidebar:
    st.subheader("📊 Chart")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    _period_labels = {
        "15m": "Last 15 min",
        "1d": "1 Day",
        "5d": "5 Days",
        "1mo": "1 Month",
        "3mo": "3 Months",
    }
    period = st.selectbox(
        "Period",
        ["15m", "1d", "5d", "1mo", "3mo"],
        index=0,
        format_func=lambda x: _period_labels[x],
    )
    interval_options = {
        "15m": ["1m"],
        "1d": ["1m", "2m", "5m", "15m"],
        "5d": ["5m", "15m", "30m"],
        "1mo": ["30m", "1h", "1d"],
        "3mo": ["1d", "1wk"],
    }
    intervals = interval_options.get(period, ["1m"])
    default_interval = "15m" if "15m" in intervals else intervals[0]
    interval = st.selectbox("Interval", intervals, index=intervals.index(default_interval))
    use_tradingview = st.checkbox("Use TradingView embedded chart", value=False)

    st.divider()
    st.subheader("🎙️ Commentator")

    voices = sorted(VALID_VOICES)
    _personalities = available_personalities()
    personality = st.selectbox(
        "Style",
        _personalities,
        index=_personalities.index("sports") if "sports" in _personalities else 0,
        format_func=persona_label,
    )
    st.caption(persona_blurb(personality))
    # Voice defaults to the chosen personality's voice. The per-personality key
    # makes the selector reset to that default when the style changes, while a
    # manual override still sticks within the same style.
    _default_voice = default_voice_for(personality)
    voice = st.selectbox(
        "Voice",
        voices,
        index=voices.index(_default_voice) if _default_voice in voices else 0,
        key=f"voice_{personality}",
        help=f"Default for this style: {_default_voice}",
    )
    # Speed also defaults per style (keyed per personality so it resets on switch).
    speed = st.slider(
        "Speech speed",
        0.8,
        1.4,
        default_speed_for(personality),
        step=0.05,
        key=f"speed_{personality}",
    )

    st.divider()
    st.subheader("🔴 Live Mode")

    _refresh_min, _refresh_max, _refresh_default = refresh_bounds(period)
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        _refresh_min,
        _refresh_max,
        _refresh_default,
        key="refresh_interval",
        help=(
            "Minimum time between updates (Live Mode only). Actual wait"
            " may be longer while audio is still playing."
        ),
    )
    live = st.toggle("Live Mode", value=False)

    if st.button("▶ Single update now", use_container_width=True):
        st.session_state["force_update"] = True

safe_ticker = _TICKER_SAFE_RE.sub("", ticker)

st.title(f"{ticker} — Stock Commentator")
st.caption(f"{persona_label(personality)} · voice: {voice}" + (" · 🔴 LIVE" if live else ""))


# --- Session state init ---
for key, default in [
    ("last_price", None),
    ("commentary_history", []),
    ("last_audio", None),
    ("live_active", False),
    ("audio_duration", 0.0),
    ("last_commentary_time", 0.0),
    ("cached_ticker", None),
    ("cached_company_name", ticker),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Determine if we need fresh data ---
force = st.session_state.pop("force_update", False)
live_just_started = live and not st.session_state["live_active"]
live_wait = max(refresh_interval, st.session_state["audio_duration"] + 1.0)
live_interval_elapsed = live and (
    time.time() - st.session_state["last_commentary_time"] >= live_wait
)
# In static mode, only reload when data-affecting settings change so that
# adjusting voice, speed, or chart type doesn't trigger an unnecessary spinner.
_data_params_changed = (
    st.session_state.get("cached_ticker") != ticker
    or st.session_state.get("cached_period") != period
    or st.session_state.get("cached_interval") != interval
)
need_refresh = should_refresh(
    force=force,
    live=live,
    live_just_started=live_just_started,
    interval_elapsed=live_interval_elapsed,
    data_params_changed=_data_params_changed,
)

# --- Fetch data (skip API calls during countdown-only reruns) ---
if need_refresh or "cached_df" not in st.session_state:
    with st.spinner(f"Loading {ticker}…"):
        try:
            df = fetch_stock_data(ticker, period=period, interval=interval)
        except ValueError:
            st.error(
                f"Invalid ticker symbol: {ticker!r}."
                " Enter a valid symbol in the sidebar (e.g. AAPL, TSLA, MSFT)."
            )
            st.stop()
        if df.empty:
            logger.error("No data for ticker=%s period=%s interval=%s", ticker, period, interval)
            st.error(f"No data for **{ticker}**. Check the symbol or try when the market is open.")
            st.stop()
        analysis = analyze_stock(df)
        if "error" in analysis:
            logger.error("Analysis failed ticker=%s error=%r", ticker, analysis["error"])
            st.error(analysis["error"])
            st.stop()
        # Only fetch company info when the ticker changes — it's a slow HTTP call
        # and the name never changes between refreshes for the same symbol.
        if st.session_state["cached_ticker"] != ticker:
            info = fetch_stock_info(ticker)
            st.session_state["cached_company_name"] = info["name"]
            st.session_state["cached_ticker"] = ticker
        st.session_state["cached_period"] = period
        st.session_state["cached_interval"] = interval
        st.session_state["cached_df"] = df
        st.session_state["cached_analysis"] = analysis
        # Cache SMA series so the chart doesn't recompute them on every rerun.
        st.session_state["cached_sma_20"] = (
            df["Close"].rolling(20).mean() if len(df) >= 20 else None
        )
        st.session_state["cached_sma_50"] = (
            df["Close"].rolling(50).mean() if len(df) >= 50 else None
        )
else:
    df = st.session_state["cached_df"]
    analysis = st.session_state["cached_analysis"]
    logger.debug("Using cached data ticker=%s period=%s interval=%s", ticker, period, interval)

company_name = st.session_state["cached_company_name"]
current_price = analysis["current_price"]
last_price = st.session_state["last_price"]
price_changed = last_price is not None and last_price != current_price
need_commentary = should_comment(
    force=force,
    live=live,
    price_changed=price_changed,
    live_just_started=live_just_started,
    interval_elapsed=live_interval_elapsed,
)

live_move: float | None = None
live_move_pct: float | None = None
live_direction: str | None = None
if price_changed:
    _move = current_price - last_price
    live_move = round(_move, 2)
    live_move_pct = round((_move / last_price) * 100, 3) if last_price != 0 else 0.0
    live_direction = "up" if _move > 0 else "down"
    logger.info(
        "price_change ticker=%s from=%.2f to=%.2f move=%+.2f pct=%+.3f",
        ticker,
        last_price,
        current_price,
        live_move,
        live_move_pct,
    )

# --- Layout ---
col_chart, col_info = st.columns([2, 1])

with col_chart:
    _chart_title = f"{company_name} ({ticker})"
    if live:
        _chart_title += " — LIVE"
    st.subheader(_chart_title)
    if use_tradingview:
        components.html(
            tradingview_embed_html(safe_ticker, interval),
            height=780,
            scrolling=False,
        )
    else:
        if need_refresh or "cached_fig" not in st.session_state:
            st.session_state["cached_fig"] = build_candlestick_figure(
                df,
                sma_20=st.session_state.get("cached_sma_20"),
                sma_50=st.session_state.get("cached_sma_50"),
            )
        st.plotly_chart(st.session_state["cached_fig"], use_container_width=True)

with col_info:
    st.subheader("📡 Live Analysis" if live else "📈 Analysis")
    st.metric(
        "Price",
        f"${analysis['current_price']:.2f}",
        f"{analysis['price_change_pct']:+.2f}%",
    )
    _trend = analysis["trend"]
    _trend_icon = {"bullish": "📈", "bearish": "📉"}.get(_trend, "➡️")
    st.markdown(f"**Trend** &nbsp; {_trend_icon} {_trend.capitalize()}")
    st.markdown(f"**Range** &nbsp; \\${analysis['low']:.2f} – \\${analysis['high']:.2f}")
    _vol_icon = {"heavy": "🔊", "light": "🔈"}.get(analysis["volume_trend"], "🔉")
    st.markdown(f"**Volume** &nbsp; {_vol_icon} {analysis['volume_trend'].capitalize()}")
    _volat_icon = {"high": "⚡", "medium": "〰️", "low": "😴"}.get(analysis["volatility"], "❔")
    st.markdown(f"**Volatility** &nbsp; {_volat_icon} {analysis['volatility'].capitalize()}")
    if analysis["rsi"] is not None:
        rsi_val = float(analysis["rsi"])
        rsi_ctx = " · Overbought" if rsi_val > 70 else (" · Oversold" if rsi_val < 30 else "")
        st.progress(min(max(rsi_val / 100.0, 0.0), 1.0), text=f"RSI {rsi_val:.0f}{rsi_ctx}")
    if analysis["sma_cross"]:
        if analysis["sma_cross"] == "golden_cross":
            st.success("⚡ Golden Cross — bullish signal")
        else:
            st.error("💀 Death Cross — bearish signal")

# --- Commentary ---
st.divider()
st.subheader("🎙️ Commentary")
st.caption(f"{persona_label(personality)} · voice: {voice}")

if need_commentary:
    # Claim this cycle up front so fast failures do not trigger immediate retries.
    st.session_state["last_commentary_time"] = time.time()
    if force:
        _trigger = "force"
    elif live_just_started:
        _trigger = "live_start"
    elif live_interval_elapsed:
        _trigger = "interval"
    else:
        _trigger = "price_change"
    logger.info(
        "commentary_start ticker=%s price=%.2f trigger=%s",
        ticker,
        current_price,
        _trigger,
    )
    audio = None
    audio_duration = None
    _tts_error = False
    _streamed = False  # True once a streaming <audio> element has been rendered

    # Speculative-prefetch fast path: an earlier live cycle may have generated
    # the no-move update during playback. Reuse it unless this tick is a
    # *significant* move (>= tolerance), which deserves fresh move-aware
    # commentary. Tiny ticks and interval refreshes reuse the prefetched line.
    _significant_move = is_significant_move(price_changed, live_move_pct, _LIVE_PREFETCH_TOL)
    _pf_key = prefetch_key(ticker, period, interval, voice, speed, personality)
    _pf = prefetch.take(_pf_key) if (_LIVE_PREFETCH and live and not _significant_move) else None

    if _pf is not None:
        commentary, audio, audio_duration = _pf
        logger.info("commentary_served_from_prefetch ticker=%s", ticker)
        _render_commentary_card(commentary, personality, analysis["trend"], live=live)
        st.session_state["audio_duration"] = audio_duration or float(refresh_interval)
    else:
        # generate_commentary handles LLM failures internally; this guard catches
        # unexpected exceptions from the prompt-building path or future changes.
        try:
            with st.spinner("Generating commentary..."):
                commentary = generate_commentary(
                    analysis,
                    ticker,
                    company_name,
                    previous_commentary=st.session_state.get("commentary_history", []),
                    live_move=live_move,
                    live_move_pct=live_move_pct,
                    live_direction=live_direction,
                    personality=personality,
                )
        except Exception as exc:
            logger.exception("Commentary generation failed ticker=%s", ticker)
            commentary = "The commentator is having technical difficulties!"
            st.error("Commentary unavailable — please try again.")
            if _APP_DEBUG:
                st.exception(exc)
        _render_commentary_card(commentary, personality, analysis["trend"], live=live)
        _stream = None
        try:
            with st.spinner("Synthesizing audio..."):
                progress = st.empty()
                audio_chunks: list[bytes] = []
                seconds_so_far = 0.0
                # Streaming: render the <audio> element up front and feed chunks
                # to the local server so playback starts at the first chunk
                # (~1s) instead of after the whole clip. Falls back silently to
                # full-WAV playback if the server can't bind.
                _stream = audio_server.open_stream() if _STREAM_AUDIO else None
                if _stream is not None:
                    _render_streaming_audio(_stream[0], _stream[1])
                    _streamed = True
                for chunk in iter_audio_chunks(commentary, voice=voice, speed=speed):
                    audio_chunks.append(chunk)
                    seconds_so_far += len(chunk) / (SAMPLE_RATE * 2)
                    if _stream is not None:
                        audio_server.feed(_stream[0], chunk)
                    progress.caption(f"Generating audio... {seconds_so_far:.1f}s")
                if _stream is not None:
                    audio_server.finish(_stream[0])

                audio_duration = max(seconds_so_far, 1)  # at least 1s so the timer is never zero
                # Small tail buffer so the next refresh doesn't cut off the final words.
                audio_duration += 0.75
                if not audio_chunks:
                    logger.warning("TTS produced no audio ticker=%s voice=%s", ticker, voice)
                audio = pcm_chunks_to_wav(audio_chunks) if audio_chunks else None
                logger.info(
                    "audio_synthesized ticker=%s duration=%.1fs chunks=%d streamed=%s",
                    ticker,
                    audio_duration,
                    len(audio_chunks),
                    _streamed,
                )
                st.session_state["audio_duration"] = (
                    audio_duration if audio else float(refresh_interval)
                )
                progress.empty()
        except Exception as exc:
            logger.exception("TTS generation failed ticker=%s", ticker)
            _tts_error = True
            # Unblock the streaming handler so it closes the connection.
            if _stream is not None:
                audio_server.finish(_stream[0])
            st.error("Audio unavailable — please try again.")
            st.session_state["audio_duration"] = float(refresh_interval)
            if _APP_DEBUG:
                st.exception(exc)

    history = st.session_state["commentary_history"]
    history.append(commentary)
    # Cap history to prevent unbounded memory growth in long live sessions.
    st.session_state["commentary_history"] = history[-_MAX_COMMENTARY_HISTORY:]
    st.session_state["last_audio"] = audio
    st.session_state["last_price"] = current_price
    # Re-anchor timer at cycle end so long generations do not trigger rapid retries.
    st.session_state["last_commentary_time"] = time.time()

    if audio and not _streamed:
        # Non-streaming path (or prefetch hit): autoplay the complete WAV.
        # When streamed, the element was already rendered and is playing.
        _render_audio(audio, autoplay=True)
    elif not audio and not _tts_error and not _streamed:
        st.error("Audio unavailable — no speech was generated.")
elif st.session_state["commentary_history"]:
    _render_commentary_card(
        st.session_state["commentary_history"][-1], personality, analysis["trend"]
    )
    audio = st.session_state["last_audio"]
    if audio:
        _render_audio(audio)
else:
    st.info("Toggle **LIVE MODE** or click **Single update now** to start the commentator.")

_history = st.session_state["commentary_history"]
if len(_history) > 1:
    with st.expander(f"📜 Commentary history ({len(_history) - 1} previous)"):
        for _line in reversed(_history[:-1]):
            st.markdown(f"› {clean_commentary(_line)}")

# --- Live mode auto-refresh ---
if live:
    st.session_state["live_active"] = True
    if st.session_state["last_price"] is None:
        st.session_state["last_price"] = current_price

    elapsed = time.time() - st.session_state["last_commentary_time"]
    remaining = max(0, live_wait - elapsed)
    if remaining > 0:
        st.sidebar.success(f"LIVE — next update in {int(remaining)}s")
    else:
        st.sidebar.success("LIVE — updating now...")

    # Use the idle playback window to speculatively generate the next update,
    # assuming the price is unchanged at the next tick (interval-driven). If it
    # is, the need_commentary block above serves it instantly next rerun. The
    # worker is a daemon thread calling only pure commentator functions (no
    # Streamlit/session_state access), so it is safe off the main thread.
    if _LIVE_PREFETCH and remaining > 0:
        _next_key = prefetch_key(ticker, period, interval, voice, speed, personality)
        _hist = list(st.session_state["commentary_history"])
        _analysis, _name, _persona = analysis, company_name, personality

        def _prefetch_worker() -> tuple[str, bytes | None, float | None]:
            text = generate_commentary(
                _analysis, ticker, _name, previous_commentary=_hist, personality=_persona
            )
            wav, dur = synthesize_wav(text, voice, speed)
            return text, wav, dur

        prefetch.start(_next_key, _prefetch_worker)

    # Sleep for the remaining wait period, then rerun; without the sleep,
    # st.rerun() would spin immediately.
    time.sleep(remaining)
    st.rerun()
else:
    st.session_state["live_active"] = False
