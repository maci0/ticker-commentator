"""Audio playback helpers shared by the Streamlit app and live prefetch.

Pure functions only — no Streamlit imports — so the live-prefetch worker can
call them off the main thread.
"""

from __future__ import annotations

import base64
import html
import time

from commentator.tts import SAMPLE_RATE, iter_audio_chunks, pcm_chunks_to_wav

# Extra seconds added after measured PCM duration so the next live refresh does
# not cut off the final words of the clip.
_PLAYBACK_TAIL_SECONDS = 0.75


def synthesize_wav(text: str, voice: str, speed: float) -> tuple[bytes | None, float | None]:
    """Synthesize a commentary line to WAV bytes + playback duration.

    Returns ``(None, None)`` when no audio is produced.
    """
    chunks = list(iter_audio_chunks(text, voice=voice, speed=speed))
    if not chunks:
        return None, None
    seconds = sum(len(c) for c in chunks) / (SAMPLE_RATE * 2)
    return pcm_chunks_to_wav(chunks), max(seconds, 1.0) + _PLAYBACK_TAIL_SECONDS


def data_uri_audio_html(
    audio: bytes,
    safe_ticker: str,
    *,
    autoplay: bool = False,
    uid: str | None = None,
) -> str:
    """Build an ``<audio>`` element with a base64 data URI source.

    When ``autoplay`` is True a unique DOM id is required so Streamlit creates a
    new element on each render (without it, autoplay will not retrigger).
    """
    b64 = base64.b64encode(audio).decode()
    if autoplay:
        element_id = uid if uid is not None else f"a{int(time.time() * 1000)}"
        uid_attr = f'id="{html.escape(element_id, quote=True)}" '
        autoplay_attr = "autoplay "
    else:
        uid_attr = ""
        autoplay_attr = ""
    title = html.escape(f"Stock commentary for {safe_ticker}", quote=True)
    return (
        f'<audio {uid_attr}{autoplay_attr}controls src="data:audio/wav;base64,{b64}"'
        f' style="width:100%;display:block"'
        f' title="{title}"'
        f' aria-label="Stock commentary audio"></audio>'
    )


def streaming_audio_html(
    stream_id: str,
    port: int,
    safe_ticker: str,
    *,
    uid: str | None = None,
) -> str:
    """Build an autoplaying ``<audio>`` element pointing at the local stream server."""
    element_id = uid if uid is not None else f"a{int(time.time() * 1000)}"
    src = f"http://127.0.0.1:{int(port)}/audio/{html.escape(stream_id, quote=True)}.wav"
    title = html.escape(f"Stock commentary for {safe_ticker}", quote=True)
    return (
        f'<audio id="{html.escape(element_id, quote=True)}" autoplay controls src="{src}"'
        f' style="width:100%;display:block"'
        f' title="{title}"'
        f' aria-label="Stock commentary audio (streaming)"></audio>'
    )
