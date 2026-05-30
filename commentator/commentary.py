"""Generate sports-style stock commentary via llama.cpp."""

import logging
import os
import random
import re
import time
from typing import Any

from huggingface_hub import hf_hub_download

from commentator._config import env_float, env_int, parse_tensor_split
from commentator.analysis import AnalysisResult
from commentator.llama_lock import LLAMA_CPP_LOCK

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)

__all__ = ["generate_commentary"]


_SYSTEM_PROMPT = (
    "You are an over-the-top sports commentator calling LIVE stock market action"
    " — John Madden meets WWE.\n"
    "\n"
    "Rules:\n"
    "- One sentence, 8-16 words max. No quotes, hashtags, or emojis.\n"
    "- Use sports metaphors, puns, and dramatic reactions mixed with real trading"
    " lingo — support, resistance, breakout, pullback, consolidation, squeeze,"
    " rally, selloff.\n"
    '- Frame buyers and sellers as rival teams battling it out — "bulls smashing'
    ' through resistance", "bears defending support", etc.\n'
    "- Natural spoken style — contractions, exclamations, ellipses.\n"
    "- Never write words in ALL CAPS. Use exclamation marks and word choice instead.\n"
    "- Weave in the actual numbers (price, percentage) naturally.\n"
    "- ONLY talk about the stock you are given. Never mention other companies or stocks.\n"
    "- Prefer the company name over the ticker symbol.\n"
    "- If prior commentary is given, don't reuse its phrases.\n"
    '- Never say "folks" or "ladies and gentlemen".'
)

_COMMENTARY_GGUF_REPO = os.getenv(
    "COMMENTARY_GGUF_REPO", "unsloth/Qwen3.5-4B-GGUF"
)
_COMMENTARY_GGUF_FILE = os.getenv(
    "COMMENTARY_GGUF_FILE", "Qwen3.5-4B-Q8_0.gguf"
)
_COMMENTARY_CTX = env_int("COMMENTARY_CTX", 4096)
# llama.cpp's default n_batch is 512; the previous 64 throttled prompt prefill
# (system prompt + stats + up to 5 history lines) for no benefit on GPU.
_COMMENTARY_BATCH = env_int("COMMENTARY_BATCH", 512)
_COMMENTARY_UBATCH = env_int("COMMENTARY_UBATCH", 512)
_COMMENTARY_GPU_LAYERS = env_int("COMMENTARY_GPU_LAYERS", -1)
_COMMENTARY_MAIN_GPU = env_int("COMMENTARY_MAIN_GPU", 0)
_COMMENTARY_TENSOR_SPLIT = parse_tensor_split("COMMENTARY_TENSOR_SPLIT")
# Flash attention speeds attention and halves KV-cache memory. Supported on the
# target gfx1100 GPU; disable (COMMENTARY_FLASH_ATTN=0) if the build lacks FA.
_COMMENTARY_FLASH_ATTN = os.getenv("COMMENTARY_FLASH_ATTN", "1") == "1"
_COMMENTARY_DEBUG = os.getenv("COMMENTARY_DEBUG", "0") == "1"
# Warm the LLM kernels at load so the first commentary doesn't pay HIP kernel
# JIT cost during generation. Set COMMENTARY_WARMUP=0 to skip.
_COMMENTARY_WARMUP = os.getenv("COMMENTARY_WARMUP", "1") == "1"
# Probability values are clamped to [0.0, 1.0] so out-of-range config never
# causes always-on or always-off tag injection.
# Defaults lowered (was 0.75/0.50) after measuring that emotion tags degrade
# clarity (no-tag WER 14% vs ~30-60% with tags) — most lines now stay clean,
# a minority get a single tag for character. See tts_eval/.
_EMOTE_CHANCE_1 = max(0.0, min(1.0, env_float("COMMENTARY_EMOTE_CHANCE_1", 0.4)))
_EMOTE_CHANCE_2 = max(0.0, min(1.0, env_float("COMMENTARY_EMOTE_CHANCE_2", 0.15)))
_COMMENTARY_TEMPERATURE = env_float("COMMENTARY_TEMPERATURE", 0.7)
_COMMENTARY_TOP_P = env_float("COMMENTARY_TOP_P", 0.9)
# Small models (e.g. Qwen3.5-0.8B) loop without a repeat penalty; >1.0 keeps the
# one-line commentary from degenerating into repeated phrases.
_COMMENTARY_REPEAT_PENALTY = env_float("COMMENTARY_REPEAT_PENALTY", 1.2)
_COMMENTARY_MAX_TOKENS = env_int("COMMENTARY_MAX_TOKENS", 96)
_COMMENTARY_LLM: "Llama | None" = None

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FOLKS_RE = re.compile(r"\b[Ff]olks\b[,!]?\s*")
_LADIES_RE = re.compile(r"\b[Ll]adies and [Gg]entlemen[,!]?\s*")
_EMOTION_TAG_RE = re.compile(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>")
# Matches punctuation pauses but not decimal points or thousands separators
# (e.g. "105.0" or "1,000").
_PAUSE_RE = re.compile(r"(?<!\d)[,;!?…—]|\.(?!\d)")


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
        logger.info(
            "Loading commentary GGUF: %s (ctx=%d, gpu_layers=%d, main_gpu=%d)",
            gguf_path,
            _COMMENTARY_CTX,
            _COMMENTARY_GPU_LAYERS,
            _COMMENTARY_MAIN_GPU,
        )
        t0 = time.time()

        def _load(flash_attn: bool) -> "Llama":
            return Llama(
                model_path=gguf_path,
                n_ctx=_COMMENTARY_CTX,
                n_gpu_layers=_COMMENTARY_GPU_LAYERS,
                n_batch=_COMMENTARY_BATCH,
                n_ubatch=_COMMENTARY_UBATCH,
                flash_attn=flash_attn,
                main_gpu=_COMMENTARY_MAIN_GPU,
                tensor_split=_COMMENTARY_TENSOR_SPLIT,
                verbose=_COMMENTARY_DEBUG,
            )

        try:
            _COMMENTARY_LLM = _load(_COMMENTARY_FLASH_ATTN)
        except Exception:
            if not _COMMENTARY_FLASH_ATTN:
                raise
            # Some llama.cpp builds lack flash-attention kernels and fail to load
            # rather than silently falling back; retry once without it.
            logger.warning(
                "Commentary model load with flash_attn failed; retrying without it",
                exc_info=True,
            )
            _COMMENTARY_LLM = _load(False)
        logger.info("Commentary GGUF loaded in %.1fs", time.time() - t0)
        if _COMMENTARY_WARMUP:
            try:
                tw = time.time()
                _COMMENTARY_LLM.create_chat_completion(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=1
                )
                logger.info("Commentary warmup done in %.2fs", time.time() - tw)
            except Exception:
                logger.warning("Commentary warmup failed (non-fatal)", exc_info=True)
    return _COMMENTARY_LLM


def _generate_with_llama_cpp(user_prompt: str) -> str:
    """Run a single chat completion and return the stripped text.

    Appends /no_think to the user message to suppress chain-of-thought in
    models that support it (e.g. Gemma 3). Any <think>…</think> blocks that
    still appear in the output are stripped by _THINK_RE as a fallback.
    Also removes banned phrases the model tends to overuse ("folks", "ladies
    and gentlemen").
    """
    llm = _get_commentary_llm()
    logger.info("commentary_llm_start prompt_len=%d", len(user_prompt))
    t0 = time.time()
    try:
        with LLAMA_CPP_LOCK:
            response: Any = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"{user_prompt}\n/no_think"},
                ],
                temperature=_COMMENTARY_TEMPERATURE,
                top_p=_COMMENTARY_TOP_P,
                repeat_penalty=_COMMENTARY_REPEAT_PENALTY,
                max_tokens=_COMMENTARY_MAX_TOKENS,
                stream=False,
            )
    finally:
        logger.info("commentary_inference elapsed=%.2fs", time.time() - t0)
    text = str(response["choices"][0]["message"]["content"])
    text = _THINK_RE.sub("", text).strip()
    text = _FOLKS_RE.sub("", text)
    text = _LADIES_RE.sub("", text)
    return text.strip()


# Tag pools pruned to the three Orpheus speaks cleanly in a commentator voice:
# <laugh>, <chuckle>, <sigh>. Dropped <gasp>/<groan> (triggered runaway/looping
# generation — 70-80s decode timeouts in tts_eval) and <cough>/<sniffle>/<yawn>
# (low naturalness and off-character for a hyped announcer).
_POSITIVE_TAGS = ["<laugh>", "<chuckle>"]
_NEGATIVE_TAGS = ["<sigh>"]
# High-drama "surprise" reuses <laugh> instead of the runaway-prone <gasp>.
_SURPRISE_TAGS = ["<laugh>"]
# <chuckle> is lighter than <laugh> and suits sideways-market commentary.
_NEUTRAL_TAGS = ["<chuckle>"]
_POSITIVE_TAGS_SURPRISED = _POSITIVE_TAGS + _SURPRISE_TAGS
_NEGATIVE_TAGS_SURPRISED = _NEGATIVE_TAGS + _SURPRISE_TAGS
_NEUTRAL_TAGS_SURPRISED = _NEUTRAL_TAGS + _SURPRISE_TAGS


def _inject_emotion_tags(text: str, analysis: dict[str, Any]) -> str:
    """Insert 0-2 Orpheus emotion tags based on market sentiment.

    The tag pool is determined by sentiment category (bullish → positive,
    bearish → negative, sideways → neutral). If |price_change_pct| > 3% or
    volatility is "high", a high-drama pool is used. Individual tags are drawn
    randomly from that pool, then placed probabilistically — first after the
    earliest punctuation pause (or prepended if none exists), second appended to
    the end. Tags are kept infrequent (see _EMOTE_CHANCE_*) because they reduce
    speech clarity.
    """
    # Strip any tags the LLM may have hallucinated; fast-path avoids regex
    # overhead (which includes scanning the full string) when no tags present.
    if "<" in text:
        text = _EMOTION_TAG_RE.sub("", text)
    text = text.strip()
    if not text:
        return text

    trend = analysis.get("trend", "sideways")
    change = abs(analysis.get("price_change_pct", 0))
    volatility = analysis.get("volatility", "unknown")

    high_drama = change > 3 or volatility == "high"
    if trend == "bullish":
        pool = _POSITIVE_TAGS_SURPRISED if high_drama else _POSITIVE_TAGS
    elif trend == "bearish":
        pool = _NEGATIVE_TAGS_SURPRISED if high_drama else _NEGATIVE_TAGS
    else:
        pool = _NEUTRAL_TAGS_SURPRISED if high_drama else _NEUTRAL_TAGS

    # Cosmetic randomness — controls voice inflection variety, not security-sensitive.
    tag1 = random.choice(pool)
    tag2 = random.choice([t for t in pool if t != tag1] or pool)

    if random.random() < _EMOTE_CHANCE_1:
        pause = _PAUSE_RE.search(text)
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
    analysis: AnalysisResult,
    ticker: str,
    company_name: str,
    previous_commentary: list[str] | None = None,
    *,
    live_move: float | None = None,
    live_move_pct: float | None = None,
    live_direction: str | None = None,
) -> str:
    """Generate sports-style stock commentary from analysis data.

    Never raises — returns a safe fallback string on any LLM failure.
    Internal errors are logged but never exposed to the caller.

    All three live_move* keyword arguments must be provided together to
    activate the live-move prompt branch.  Supplying only some of them logs
    a warning and falls back to the opening-commentary prompt.

    Only the last 5 entries of previous_commentary are included in the prompt
    to limit context size; callers may pass a longer history list.
    """
    # Warn when only some live_move args are provided — the condition at the
    # prompt-building step requires all three, so a partial set silently falls
    # back to the opening-commentary branch, which is almost certainly not what
    # the caller intended.
    _live_args = (live_move, live_move_pct, live_direction)
    if any(a is not None for a in _live_args) and not all(a is not None for a in _live_args):
        logger.warning(
            "generate_commentary: partial live_move args provided "
            "(live_move=%r, live_move_pct=%r, live_direction=%r); "
            "falling back to opening-commentary prompt",
            live_move, live_move_pct, live_direction,
        )

    trend = analysis.get("trend", "sideways")
    change = analysis.get("price_change_pct", 0)
    price = analysis.get("current_price", 0)
    high = analysis.get("high", 0)
    low = analysis.get("low", 0)
    vol = analysis.get("volume_trend", "normal")
    volatility = analysis.get("volatility", "unknown")
    rsi = analysis.get("rsi")
    sma_cross = analysis.get("sma_cross")

    stats = f"""- Current price: ${price:.2f}
- Today's change: {change:+.2f}%
- Trend: {trend}
- Day range: ${low:.2f} - ${high:.2f}
- Volume: {vol}
- Volatility: {volatility}"""

    if live_move is not None and live_move_pct is not None and live_direction is not None:
        user_prompt = (
            f"{company_name} just moved"
            f" ${live_move:+.2f} ({live_move_pct:+.3f}%) {live_direction}!\n"
            f"{stats}"
        )
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
        user_prompt += (
            "\n- Prior commentary (don't repeat these phrases,"
            f" ignore any stock names in them):\n{numbered}"
        )

    try:
        text = _generate_with_llama_cpp(user_prompt)
        result = _inject_emotion_tags(text, analysis)
        logger.debug("commentary_result text=%r", result)
        return result
    except Exception:
        logger.exception("Commentary generation failed")
        return "The commentator is having technical difficulties!"
