"""Tests for the streaming-WAV HTTP server (headless; no browser needed)."""

import http.client
import struct
import threading
import time

from commentator import audio_server


def _get(port: int, path: str) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, body


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
    status, body = _get(port, f"/audio/{sid}.wav")
    assert status == 200
    assert body[:4] == b"RIFF"
    assert body[44:] == b"".join(payload)  # exact PCM after the 44-byte header


def test_unknown_stream_id_returns_404() -> None:
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    audio_server.finish(sid)
    status, _ = _get(port, "/audio/does_not_exist.wav")
    assert status == 404


def test_bad_path_returns_404() -> None:
    res = audio_server.open_stream()
    assert res is not None
    sid, port = res
    audio_server.finish(sid)
    status, _ = _get(port, "/not-audio")
    assert status == 404
