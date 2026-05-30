"""Tests for the streaming-WAV HTTP server (headless; no browser needed)."""

import http.client
import struct
import threading
import time

from commentator import audio_server


def _get(port: int, path: str, headers: "dict[str, str] | None" = None) -> "tuple":
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path, headers=headers or {})
    r = conn.getresponse()
    body = r.read()
    status = r.status
    content_range = r.getheader("Content-Range")
    accept_ranges = r.getheader("Accept-Ranges")
    conn.close()
    return status, body, content_range, accept_ranges


def _make_completed_stream(payload: list[bytes]) -> "tuple[int, str, bytes]":
    """Open a stream, feed payload, finish it, and return (port, id, full_wav)."""
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    for c in payload:
        audio_server.feed(sid, c)
    audio_server.finish(sid)
    # the complete WAV = 44-byte header + concatenated payload
    full = audio_server._complete_wav(audio_server._DEFAULT_SAMPLE_RATE, b"".join(payload))
    return port, sid, full


def test_streaming_wav_header_is_valid() -> None:
    hdr = audio_server._streaming_wav_header(24000)
    assert len(hdr) == 44
    assert hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE"
    assert hdr[12:16] == b"fmt " and hdr[36:40] == b"data"
    channels, = struct.unpack("<H", hdr[22:24])
    rate, = struct.unpack("<I", hdr[24:28])
    bits, = struct.unpack("<H", hdr[34:36])
    assert (channels, rate, bits) == (1, 24000, 16)


def test_open_stream_returns_id_and_port() -> None:
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    assert sid and isinstance(port, int) and port > 0
    audio_server.finish(sid)  # cleanup


def test_stream_serves_header_plus_fed_pcm() -> None:
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    payload = [b"\x01\x02" * 80, b"\x03\x04" * 80]

    def producer() -> None:
        for c in payload:
            time.sleep(0.05)
            audio_server.feed(sid, c)
        audio_server.finish(sid)

    threading.Thread(target=producer, daemon=True).start()
    status, body, _cr, accept = _get(port, f"/audio/{sid}.wav")
    assert status == 200
    assert body[:4] == b"RIFF"
    assert body[44:] == b"".join(payload)  # exact PCM after the 44-byte header
    assert accept == "bytes"  # advertises range support


def test_unknown_stream_id_returns_404() -> None:
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    audio_server.finish(sid)
    status, *_ = _get(port, "/audio/does_not_exist.wav")
    assert status == 404


def test_bad_path_returns_404() -> None:
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    audio_server.finish(sid)
    status, *_ = _get(port, "/not-audio")
    assert status == 404


# ── Range / seek (206) on a finished clip ────────────────────────────


def test_finished_clip_full_get_has_correct_length() -> None:
    port, sid, full = _make_completed_stream([b"\xaa\xbb" * 100])
    status, body, _cr, accept = _get(port, f"/audio/{sid}.wav")
    assert status == 200
    assert body == full
    assert accept == "bytes"


def test_range_request_returns_206_slice() -> None:
    port, sid, full = _make_completed_stream([b"\x10\x20" * 200])
    total = len(full)
    status, body, cr, accept = _get(
        port, f"/audio/{sid}.wav", {"Range": "bytes=100-199"}
    )
    assert status == 206
    assert body == full[100:200]
    assert cr == f"bytes 100-199/{total}"
    assert accept == "bytes"


def test_open_ended_range_returns_to_end() -> None:
    port, sid, full = _make_completed_stream([b"\x01\x02" * 150])
    total = len(full)
    status, body, cr, _a = _get(port, f"/audio/{sid}.wav", {"Range": "bytes=50-"})
    assert status == 206
    assert body == full[50:]
    assert cr == f"bytes 50-{total - 1}/{total}"


def test_unsatisfiable_range_returns_416() -> None:
    port, sid, full = _make_completed_stream([b"\x00\x00" * 10])
    total = len(full)
    status, _body, cr, _a = _get(
        port, f"/audio/{sid}.wav", {"Range": f"bytes={total + 5}-{total + 10}"}
    )
    assert status == 416
    assert cr == f"bytes */{total}"
