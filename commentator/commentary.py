"""Generate sports-style stock commentary via llama.cpp."""

import logging
import os
import random
import re
from typing import Any

from huggingface_hub import hf_hub_download

from commentator.llama_lock import LLAMA_CPP_LOCK

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)

__all__ = ["generate_commentary", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = """\
You are an over-the-top sports commentator calling LIVE stock market action — John Madden meets WWE.

Rules:
- One sentence, 8-16 words max. No quotes, hashtags, or emojis.
- Use sports metaphors, puns, and dramatic reactions mixed with real trading lingo — support, resistance, breakout, pullback, consolidation, squeeze, rally, selloff.
- Frame buyers and sellers as rival teams battling it out — "bulls smashing through resistance", "bears defending support", etc.
- Natural spoken style — contractions, exclamations, ellipses.
- Never write words in ALL CAPS. Use exclamation marks and word choice instead.
- Weave in the actual numbers (price, percentage) naturally.
- ONLY talk about the stock you are given. Never mention other companies or stocks.
- Prefer the company name over the ticker symbol.
- If prior commentary is given, don't reuse its phrases.
- Never say "folks" or "ladies and gentlemen"."""

_COMMENTARY_GGUF_REPO = os.getenv(
    "COMMENTARY_GGUF_REPO", "bartowski/google_gemma-3-4b-it-GGUF"
)
_COMMENTARY_GGUF_FILE = os.getenv(
    "COMMENTARY_GGUF_FILE", "google_gemma-3-4b-it-Q4_K_M.gguf"
)
_COMMENTARY_CTX = int(os.getenv("COMMENTARY_CTX", "4096"))
_COMMENTARY_GPU_LAYERS = int(os.getenv("COMMENTARY_GPU_LAYERS", "-1"))
_COMMENTARY_MAIN_GPU = int(os.getenv("COMMENTARY_MAIN_GPU", "0"))
_COMMENTARY_TENSOR_SPLIT = [
    float(x)
    for x in os.getenv("COMMENTARY_TENSOR_SPLIT", "1.0").split(",")
    if x.strip()
]
_COMMENTARY_DEBUG = os.getenv("COMMENTARY_DEBUG", "0") == "1"
_EMOTE_CHANCE_1 = float(os.getenv("COMMENTARY_EMOTE_CHANCE_1", "0.75"))
_EMOTE_CHANCE_2 = float(os.getenv("COMMENTARY_EMOTE_CHANCE_2", "0.50"))
_COMMENTARY_LLM = None


def _get_commentary_llm() -> "Llama":
    """Lazy-init the commentary LLM singleton (thread-safe double-checked lock)."""
    global _COMMENTARY_LLM
    if _COMMENTARY_LLM is not None:
        return _COMMENTARY_LLM
    if Llama is None:
        raise RuntimeError("llama_cpp is not installed")
    with LLAMA_CPP_LOCK:
        if _COMMENTARY_LLM is not None:
            return _COMMENTARY_LLM
        gguf_path = hf_hub_download(
            repo_id=_COMMENTARY_GGUF_REPO,
            filename=_COMMENTARY_GGUF_FILE,
        )
        logger.debug(
            "Loading commentary GGUF: %s (ctx=%d, gpu_layers=%d, main_gpu=%d)",
            gguf_path,
            _COMMENTARY_CTX,
            _COMMENTARY_GPU_LAYERS,
            _COMMENTARY_MAIN_GPU,
        )
        _COMMENTARY_LLM = Llama(
            model_path=gguf_path,
            n_ctx=_COMMENTARY_CTX,
            n_gpu_layers=_COMMENTARY_GPU_LAYERS,
            n_batch=64,
            main_gpu=_COMMENTARY_MAIN_GPU,
            tensor_split=_COMMENTARY_TENSOR_SPLIT,
            verbose=_COMMENTARY_DEBUG,
        )
    return _COMMENTARY_LLM


def _generate_with_llama_cpp(user_prompt: str) -> str:
    """Run a single chat completion and return the cleaned text."""
    llm = _get_commentary_llm()
    logger.debug("Running create_chat_completion")
    with LLAMA_CPP_LOCK:
        response: Any = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}\n/no_think"},
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=96,
            stream=False,
        )
    text = str(response["choices"][0]["message"]["content"])
    # Strip Qwen3 thinking blocks if present.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip banned phrases the LLM tends to overuse.
    text = re.sub(r"\b[Ff]olks\b[,!]?\s*", "", text)
    text = re.sub(r"\b[Ll]adies and [Gg]entlemen[,!]?\s*", "", text)
    return text.strip()


_POSITIVE_TAGS = ["<laugh>", "<chuckle>"]
_NEGATIVE_TAGS = ["<sigh>", "<groan>"]
_SURPRISE_TAGS = ["<gasp>"]
_NEUTRAL_TAGS = ["<chuckle>", "<sniffle>", "<yawn>"]


def _inject_emotion_tags(text: str, analysis: dict[str, Any]) -> str:
    """Insert 1-2 Orpheus emotion tags based on market sentiment.

    Tags are chosen deterministically by sentiment category, then placed
    probabilistically — first after the earliest punctuation pause, second
    appended to the end.  The randomness here is cosmetic, not security-
    relevant: it controls voice inflection variety.
    """
    # Strip any tags the LLM may have hallucinated.
    text = re.sub(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>", "", text)
    text = text.strip()
    if not text:
        return text

    trend = analysis.get("trend", "sideways")
    change = abs(analysis.get("price_change_pct", 0))
    volatility = analysis.get("volatility", "unknown")

    # Pick tag pool based on sentiment.
    if trend == "bullish":
        pool = _POSITIVE_TAGS
    elif trend == "bearish":
        pool = _NEGATIVE_TAGS
    else:
        pool = _NEUTRAL_TAGS

    # Big moves or high volatility get a surprise tag mixed in.
    if change > 3 or volatility == "high":
        pool = pool + _SURPRISE_TAGS

    tag1 = random.choice(pool)
    tag2 = random.choice([t for t in pool if t != tag1] or pool)

    # Insert first tag after the first punctuation/pause.
    if random.random() < _EMOTE_CHANCE_1:
        pause = re.search(r"(?<!\d)[,;!?…—]|\.(?!\d)", text)
        if pause:
            pos = pause.end()
            text = text[:pos] + f" {tag1}" + text[pos:]
        else:
            text = f"{tag1} {text}"

    # Only add second tag some of the time to feel natural.
    if random.random() < _EMOTE_CHANCE_2:
        text = text.rstrip(".!") + f" {tag2}"

    return text


def generate_commentary(
    analysis: dict[str, Any],
    ticker: str,
    company_name: str,
    previous_commentary: list[str] | None = None,
) -> str:
    """Generate sports-style comedy commentary from stock analysis.

    Never raises — returns a safe fallback string on any LLM failure.
    Internal errors are logged but never exposed to the caller.
    """
    trend = analysis.get("trend", "sideways")
    change = analysis.get("price_change_pct", 0)
    price = analysis.get("current_price", 0)
    high = analysis.get("high", 0)
    low = analysis.get("low", 0)
    vol = analysis.get("volume_trend", "normal")
    volatility = analysis.get("volatility", "unknown")
    rsi = analysis.get("rsi")
    sma_cross = analysis.get("sma_cross")

    live_move = analysis.get("live_move")
    live_move_pct = analysis.get("live_move_pct")
    live_dir = analysis.get("live_direction")

    stats = f"""- Current price: ${price:.2f}
- Today's change: {change:+.2f}%
- Trend: {trend}
- Day range: ${low:.2f} - ${high:.2f}
- Volume: {vol}
- Volatility: {volatility}"""

    if live_move is not None:
        user_prompt = f"""{company_name} just moved ${live_move:+.2f} ({live_move_pct:+.3f}%) {live_dir}!
{stats}"""
    else:
        user_prompt = f"""Opening commentary for {company_name}:
{stats}"""

    if rsi is not None:
        user_prompt += f"\n- RSI: {rsi}"
    if sma_cross:
        label = (
            "Golden Cross — bullish signal!"
            if sma_cross == "golden_cross"
            else "Death Cross — bearish signal!"
        )
        user_prompt += f"\n- Moving average signal: {label}"

    if previous_commentary:
        recent = previous_commentary[-5:]
        numbered = "\n".join(f"  {i+1}. {line}" for i, line in enumerate(recent))
        user_prompt += f"\n- Prior commentary (don't repeat these phrases, ignore any stock names in them):\n{numbered}"

    try:
        text = _generate_with_llama_cpp(user_prompt)
        return _inject_emotion_tags(text, analysis)
    except Exception:
        logger.exception("Commentary generation failed")
        return "The commentator is having technical difficulties!"
