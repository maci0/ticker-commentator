"""Pure HTML/label helpers for the Streamlit commentary UI."""

from __future__ import annotations

import html

from commentator.live import clean_commentary

# UI flavor for each commentator personality: (emoji, one-line blurb). Keys must
# match commentator.commentary personalities; unknown keys fall back gracefully.
PERSONA_UI: dict[str, tuple[str, str]] = {
    "sports": ("🏟️", "Hyped play-by-play"),
    "neutral": ("📊", "Calm market analyst"),
    "kramer": ("📣", "Mad Money showman"),
    "seinfeld": ("🎤", "Observational comedy"),
    "attenborough": ("🦁", "Nature-doc narrator"),
    "wsb": ("🚀", "Diamond-hands degenerate"),
    "noir": ("🕵️", "Hardboiled detective"),
    "educator": ("🎓", "Explains the indicators"),
    "gordon_ramsay": ("🔥", "Furious chef"),
    "pirate": ("🏴‍☠️", "Swashbuckling captain"),
    "shakespeare": ("🎭", "Dramatic bard"),
    "surfer": ("🏄", "Chill surfer dude"),
    "doomer": ("💀", "Permabear doom"),
    "bob_ross": ("🎨", "Serene painter"),
    "zen": ("🧘", "Tranquil zen master"),
}

_TREND_COLOR = {"bullish": "#26a69a", "bearish": "#ef5350"}


def persona_emoji(name: str) -> str:
    return PERSONA_UI.get(name, ("🎙️", ""))[0]


def persona_label(name: str) -> str:
    emoji, _ = PERSONA_UI.get(name, ("🎙️", ""))
    return f"{emoji} {name.replace('_', ' ').title()}"


def persona_blurb(name: str) -> str:
    return PERSONA_UI.get(name, ("", "Custom style"))[1]


def commentary_card_html(
    text: str,
    personality: str,
    trend: str,
    *,
    live: bool = False,
) -> str:
    """Styled broadcast card HTML (clean text, persona label, trend accent)."""
    color = _TREND_COLOR.get(trend, "#8899aa")
    badge = "🔴 ON AIR" if live else "🎙️"
    safe = html.escape(clean_commentary(text)) or "…"
    return (
        f'<div style="border-left:5px solid {color};'
        f" background:rgba(127,127,127,0.08); padding:14px 18px;"
        f' border-radius:8px; margin:4px 0 12px 0;">'
        f'<div style="font-size:0.78rem; letter-spacing:.04em; opacity:.6;'
        f' text-transform:uppercase; margin-bottom:6px;">'
        f"{html.escape(persona_label(personality))} &nbsp;·&nbsp; {badge}</div>"
        f'<div style="font-size:1.35rem; line-height:1.55; font-weight:500;">'
        f"{safe}</div></div>"
    )
