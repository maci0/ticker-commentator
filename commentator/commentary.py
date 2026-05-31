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

__all__ = [
    "generate_commentary",
    "available_personalities",
    "default_voice_for",
    "default_speed_for",
]


# Shared, persona-agnostic rules. "No ALL CAPS" is a TTS constraint (Orpheus
# mis-pronounces capitalized words), not a style choice — keep it for every voice.
_COMMON_RULES = (
    "Rules:\n"
    "- One sentence, 8-16 words max. No quotes, hashtags, or emojis.\n"
    "- Never write words in ALL CAPS. Use exclamation marks and word choice instead.\n"
    "- Weave in the actual numbers (price, percentage) naturally.\n"
    "- Write prices and percentages as digits ($312.07, +1.2%); never spell"
    " numbers out in words.\n"
    "- ONLY talk about the stock you are given. Never mention other companies or stocks.\n"
    "- Prefer the company name over the ticker symbol.\n"
    "- Anchor on the specific move, signal, or level you're given — be concrete,"
    " not generic.\n"
    "- If prior commentary is given, open with a fresh angle and don't reuse its"
    " phrasing."
)

# Per-personality persona blocks, prepended to _COMMON_RULES.
_PERSONAS: dict[str, str] = {
    "sports": (
        "You are an over-the-top sports commentator calling LIVE stock market"
        " action — John Madden meets WWE. Use sports metaphors, puns, and dramatic"
        " reactions mixed with real trading lingo (support, resistance, breakout,"
        " pullback, squeeze, rally, selloff). Frame buyers and sellers as rival"
        " teams battling it out — bulls smashing resistance, bears defending"
        ' support. Natural spoken style with contractions and exclamations. Never'
        ' say "folks" or "ladies and gentlemen".'
    ),
    "neutral": (
        "You are a calm, professional market analyst. Report the move factually"
        " and concisely in plain spoken English — no hype, no metaphors, no jokes,"
        " no sound effects. Just the clear takeaway a serious trader would want."
    ),
    "kramer": (
        "You are a hyper-energetic financial-TV showman in the style of a Mad Money"
        " host. Rapid-fire, punchy, full of conviction — playful buy/sell energy,"
        " catchphrase enthusiasm, and theatrical excitement, in a spoken style with"
        " exclamations."
    ),
    "seinfeld": (
        "You are an observational stand-up comedian riffing on the stock like Jerry"
        ' Seinfeld. Dry, witty, "what is the deal with..." everyday-analogy humor'
        " and comedic timing, in a natural spoken style."
    ),
    "attenborough": (
        "You are a hushed nature-documentary narrator in the style of David"
        " Attenborough, observing the market as wildlife — bulls and bears as"
        " creatures in their habitat. Calm wonder, vivid imagery, gentle awe, in a"
        " spoken style."
    ),
    "wsb": (
        "You are a self-aware r/wallstreetbets retail trader. Reckless hype,"
        " diamond-hands conviction, 'to the moon', YOLO bravado and gambling humor,"
        " spoken with manic excitement. Keep it PG — no profanity."
    ),
    "noir": (
        "You are a hardboiled 1940s film-noir detective narrating the ticker like a"
        " crime scene. World-weary, clipped, cynical, metaphor-heavy, in a smoky"
        " spoken monologue."
    ),
    "educator": (
        "You are a patient finance teacher. In one plain sentence, explain what the"
        " move and one indicator (RSI, moving-average cross, or volatility) actually"
        " mean for the stock. Clear, calm, jargon-light — no hype, no jokes."
    ),
    "gordon_ramsay": (
        "You are a furious celebrity chef in the style of Gordon Ramsay, berating"
        " the stock like a botched dish — savage, exasperated, hot-tempered kitchen"
        " insults aimed at the chart. Spoken and PG — no profanity."
    ),
    "pirate": (
        "You are a swashbuckling pirate captain calling the market like plunder on"
        " the high seas — arr, booty, treasure, storms and mutiny. Gruff, boisterous"
        " spoken style."
    ),
    "shakespeare": (
        "You are a Shakespearean bard proclaiming the stock in dramatic Early Modern"
        " English — thee, thou, doth, a soliloquy flourish. Theatrical and grand."
    ),
    "surfer": (
        "You are a laid-back surfer dude narrating the stock totally chill — whoa,"
        " gnarly, stoked, riding the wave of the trend. Relaxed spoken slang."
    ),
    "doomer": (
        "You are a gloomy permabear sure every move is the beginning of the end —"
        " fatalistic, weary, ominous. Spoken with grim resignation."
    ),
    "bob_ross": (
        "You are a serene painting instructor in the style of Bob Ross, treating the"
        " chart like a peaceful landscape — happy little gains, gentle reassurance,"
        " soft warmth. Calm spoken style."
    ),
    "zen": (
        "You are a tranquil zen master watching the market with detached equanimity"
        " — calm, minimal, mindful of impermanence. Quiet spoken style."
    ),
}
_DEFAULT_PERSONALITY = os.getenv("COMMENTARY_PERSONALITY", "sports").strip().lower()

# Per-personality delivery profile: default Orpheus voice, speech speed (clamped
# to the TTS [0.8, 1.4] range), and emote scale (multiplier on the base emotion-
# tag probabilities; 0.0 = no tags). Voices must be in commentator.tts.VALID_VOICES.
# Tuned per persona: energetic/comedic voices are faster with more emotes; calm,
# factual, or brooding voices are slower with fewer or no tags.
_FALLBACK_PROFILE = {"voice": "leo", "speed": 1.2, "emote": 1.0}
_PERSONA_PROFILE: dict[str, dict] = {
    "sports": {"voice": "leo", "speed": 1.3, "emote": 1.0},
    "neutral": {"voice": "dan", "speed": 1.0, "emote": 0.0},
    "kramer": {"voice": "zac", "speed": 1.4, "emote": 1.5},
    "seinfeld": {"voice": "leo", "speed": 1.1, "emote": 1.0},
    "attenborough": {"voice": "dan", "speed": 0.9, "emote": 0.5},
    "wsb": {"voice": "zac", "speed": 1.35, "emote": 1.5},
    "noir": {"voice": "dan", "speed": 0.95, "emote": 0.5},
    "educator": {"voice": "tara", "speed": 1.0, "emote": 0.0},
    "gordon_ramsay": {"voice": "zac", "speed": 1.3, "emote": 1.5},
    "pirate": {"voice": "zac", "speed": 1.1, "emote": 1.0},
    "shakespeare": {"voice": "leo", "speed": 1.0, "emote": 0.5},
    "surfer": {"voice": "leo", "speed": 1.0, "emote": 1.0},
    "doomer": {"voice": "dan", "speed": 0.9, "emote": 0.5},
    "bob_ross": {"voice": "dan", "speed": 0.85, "emote": 0.5},
    "zen": {"voice": "leah", "speed": 0.8, "emote": 0.0},
}


def _profile(personality: str) -> dict:
    return _PERSONA_PROFILE.get((personality or "").strip().lower(), _FALLBACK_PROFILE)


def default_voice_for(personality: str) -> str:
    """The default Orpheus voice for a personality."""
    return _profile(personality)["voice"]


def default_speed_for(personality: str) -> float:
    """The default speech speed for a personality (clamped to [0.8, 1.4])."""
    return max(0.8, min(1.4, float(_profile(personality)["speed"])))


# One exemplar line per persona — few-shot anchoring lifts small-model output
# quality far more than rules alone: it pins the voice, the one-sentence length,
# and the digit-number style. Each uses $312.07 / 1.2% to model the number form.
_PERSONA_EXAMPLES: dict[str, str] = {
    "sports":
        "Bulls storm the gates as Apple rips through resistance to $312.07 on heavy volume!",
    "neutral":
        "Apple is up 1.2% to $312.07 on above-average volume, holding above its averages.",
    "kramer":
        "Apple's on fire at $312.07 — that's a buy buy buy as the bulls take the floor!",
    "seinfeld":
        "What's the deal with Apple at $312.07? Up 1.2% like it's doing us a favor.",
    "attenborough":
        "Here we observe Apple gliding to $312.07 as the herd of bulls grazes on volume.",
    "wsb":
        "Apple ripping to $312.07, diamond hands only, this rocket isn't stopping!",
    "noir":
        "Apple slunk in at $312.07, down 1.2% — the kind of number that means trouble.",
    "educator":
        "Apple's RSI near 68 means it's nearing overbought while price holds $312.07.",
    "gordon_ramsay":
        "This Apple chart at $312.07 is a disaster — the bulls have overcooked it!",
    "pirate":
        "Arr, Apple be sailin' to $312.07 with the wind at her back and a full hold!",
    "shakespeare":
        "Lo, Apple doth ascend to $312.07, as bulls and bears wage eternal war below.",
    "surfer":
        "Whoa, Apple's totally pumping to $312.07, riding a gnarly wave of volume, dude.",
    "doomer":
        "Apple's fragile climb to $312.07 is just the calm before the inevitable collapse.",
    "bob_ross":
        "Apple drifts up to a happy little $312.07 — no mistakes here, just gentle gains.",
    "zen":
        "Apple rests at $312.07; the market rises and falls, and we simply observe.",
}


def _system_prompt(personality: str) -> str:
    """Compose the system prompt for a personality (falls back to 'sports')."""
    key = personality if personality in _PERSONAS else "sports"
    prompt = f"{_PERSONAS[key]}\n\n{_COMMON_RULES}"
    example = _PERSONA_EXAMPLES.get(key)
    if example:
        prompt += f'\n\nExample (match this voice, length, and number style): "{example}"'
    return prompt


def available_personalities() -> list[str]:
    """Sorted list of selectable commentator personalities."""
    return sorted(_PERSONAS)

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
# Splits at sentence-ending punctuation followed by whitespace; the trailing \s
# requirement avoids splitting decimals like "312.07".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str) -> str:
    """Keep only the first sentence — personas tend to run on past the one-line
    limit, so this enforces brevity deterministically regardless of the model."""
    text = text.strip()
    parts = _SENTENCE_SPLIT_RE.split(text, maxsplit=1)
    return parts[0].strip() if parts else text


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


def _generate_with_llama_cpp(user_prompt: str, system_prompt: str) -> str:
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
                    {"role": "system", "content": system_prompt},
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
    # Enforce the one-sentence limit even when the model runs on.
    return _first_sentence(text.strip())


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


def _inject_emotion_tags(text: str, analysis: dict[str, Any], scale: float = 1.0) -> str:
    """Insert 0-2 Orpheus emotion tags based on market sentiment.

    The tag pool is determined by sentiment category (bullish → positive,
    bearish → negative, sideways → neutral). If |price_change_pct| > 3% or
    volatility is "high", a high-drama pool is used. Individual tags are drawn
    randomly from that pool, then placed probabilistically — first after the
    earliest punctuation pause (or prepended if none exists), second appended to
    the end. Tags are kept infrequent (see _EMOTE_CHANCE_*) because they reduce
    speech clarity; `scale` multiplies the per-personality probability (0 = off).
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
    # Distinct second tag; None when the pool has no alternative (avoids the
    # "<chuckle> <chuckle>" double-same-tag artifact).
    alternatives = [t for t in pool if t != tag1]
    tag2 = random.choice(alternatives) if alternatives else None

    chance1 = min(1.0, _EMOTE_CHANCE_1 * scale)
    chance2 = min(1.0, _EMOTE_CHANCE_2 * scale)

    if random.random() < chance1:
        pause = _PAUSE_RE.search(text)
        if pause:
            pos = pause.end()
            text = text[:pos] + f" {tag1}" + text[pos:]
        else:
            text = f"{tag1} {text}"

    # Only add a distinct second tag some of the time to feel natural.
    if tag2 is not None and random.random() < chance2:
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
    personality: str | None = None,
) -> str:
    """Generate stock commentary from analysis data in the chosen personality.

    personality selects the commentator voice (see available_personalities());
    unknown/None falls back to COMMENTARY_PERSONALITY (default 'sports'). The
    'neutral' analyst voice gets no emotion tags.

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

    persona = (personality or _DEFAULT_PERSONALITY).strip().lower()
    if persona not in _PERSONAS:
        logger.warning("generate_commentary: unknown personality %r; using 'sports'", persona)
        persona = "sports"

    emote_scale = float(_profile(persona)["emote"])
    try:
        text = _generate_with_llama_cpp(user_prompt, _system_prompt(persona))
        # Numbers stay as digits here (clean for display); the TTS layer converts
        # them to spoken words at synthesis time (commentator.tts.numbers_to_speech).
        if emote_scale <= 0:
            # No-emote personas (neutral/educator/zen): strip any stray tags.
            result = _EMOTION_TAG_RE.sub("", text).strip() if "<" in text else text
        else:
            result = _inject_emotion_tags(text, analysis, scale=emote_scale)
        logger.debug("commentary_result persona=%s text=%r", persona, result)
        return result
    except Exception:
        logger.exception("Commentary generation failed")
        return "The commentator is having technical difficulties!"
