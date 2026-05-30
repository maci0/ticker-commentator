"""Local streaming-WAV HTTP server for progressive audio playback.

The default playback path waits for the whole clip to synthesize (~4 s) before
the browser gets a complete WAV. With streaming enabled, an `<audio>` element
points at this server's `/audio/<id>.wav` endpoint; PCM chunks are fed in as
they decode and streamed to the browser, so playback starts at the first chunk
(~0.5 s) instead of the full duration.

Seeking: a live stream has unknown length and is consumed once, so the first
request is streamed (HTTP 200). Chunks are also buffered, and once the clip is
finished the complete WAV is retained so later Range requests (the browser
seeking) are answered with 206 Partial Content from that buffer.

Single background ThreadingHTTPServer bound to 127.0.0.1 (local playback only).
Opt-in via STREAM_AUDIO=1.
"""

import collections
import logging
import os
import queue
import re
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

_SENTINEL = object()
_DEFAULT_SAMPLE_RATE = 24000
# How many finished clips to retain for seek/Range requests (~300 KB each).
_MAX_COMPLETE = 8
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

_LOCK = threading.Lock()
_STREAMS: "dict[str, _Stream]" = {}
_COMPLETE: "collections.OrderedDict[str, bytes]" = collections.OrderedDict()
_SERVER: "ThreadingHTTPServer | None" = None
_PORT: int = 0
_COUNTER = 0


class _Stream:
    """An in-progress audio stream: a live queue plus a buffer accumulated for
    the seekable complete WAV."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.queue: "queue.Queue" = queue.Queue(maxsize=256)
        self.buffer: list[bytes] = []


def _streaming_wav_header(sample_rate: int) -> bytes:
    """44-byte WAV header for a stream of unknown length: RIFF/data sizes are set
    to a large sentinel so browsers keep reading until the connection closes."""
    big = 0x7FFFFFFF
    return _wav_header(sample_rate, big, big)


def _complete_wav(sample_rate: int, pcm: bytes) -> bytes:
    """A finished, seekable 16-bit mono WAV with correct length fields."""
    return _wav_header(sample_rate, 36 + len(pcm), len(pcm)) + pcm


def _wav_header(sample_rate: int, riff_size: int, data_size: int) -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
    )


def _parse_range(header: str | None, total: int) -> "tuple[int, int] | None":
    """Parse a `Range: bytes=start-end` header into inclusive (start, end), or
    None if absent/unsatisfiable. Open-ended forms (bytes=N-, bytes=-N) handled."""
    if not header:
        return None
    m = _RANGE_RE.search(header)
    if not m:
        return None
    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        return None
    if start_s == "":  # suffix range: last N bytes
        length = min(int(end_s), total)
        return (total - length, total - 1)
    start = int(start_s)
    end = int(end_s) if end_s else total - 1
    # Validation (out-of-range start, start>end) is left to the caller so it can
    # respond 416 rather than silently serving the whole body.
    return (start, end)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:  # silence default stderr logging
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if not path.startswith("/audio/") or not path.endswith(".wav"):
            self.send_error(404)
            return
        stream_id = path[len("/audio/") : -len(".wav")]
        with _LOCK:
            complete = _COMPLETE.get(stream_id)
            stream = _STREAMS.get(stream_id)

        # Finished clip: support range/seek from the complete buffer.
        if complete is not None:
            self._serve_complete(complete)
            return
        if stream is None:
            self.send_error(404)
            return
        self._serve_stream(stream, stream_id)

    def _serve_complete(self, wav: bytes) -> None:
        total = len(wav)
        rng = _parse_range(self.headers.get("Range"), total)
        if rng is None:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(total))
            self.end_headers()
            self._write(wav)
            return
        start, end = rng
        end = min(end, total - 1)
        if start < 0 or start >= total or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = wav[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write(body)

    def _serve_stream(self, stream: "_Stream", stream_id: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")  # ranges available once finished
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(_streaming_wav_header(stream.sample_rate))
            self.wfile.flush()
            while True:
                chunk = stream.queue.get()
                if chunk is _SENTINEL:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed/seeked — normal
        finally:
            with _LOCK:
                _STREAMS.pop(stream_id, None)

    def _write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _ensure_server() -> int:
    """Start the singleton server if needed; return its port (0 if unavailable)."""
    global _SERVER, _PORT
    with _LOCK:
        if _SERVER is not None:
            return _PORT
        port = int(os.getenv("STREAM_AUDIO_PORT", "0"))
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        except OSError:
            logger.warning("audio_server: could not bind port %d; streaming disabled", port)
            return 0
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, name="audio-server", daemon=True).start()
        _SERVER = srv
        _PORT = srv.server_address[1]
        logger.info("audio_server listening on 127.0.0.1:%d", _PORT)
        return _PORT


def open_stream(sample_rate: int = _DEFAULT_SAMPLE_RATE) -> "tuple[str, int] | None":
    """Register a new audio stream. Returns (stream_id, port) or None if the
    server is unavailable."""
    global _COUNTER
    port = _ensure_server()
    if not port:
        return None
    with _LOCK:
        _COUNTER += 1
        stream_id = f"s{_COUNTER}"
        _STREAMS[stream_id] = _Stream(sample_rate)
    return stream_id, port


def feed(stream_id: str, pcm: bytes) -> None:
    with _LOCK:
        s = _STREAMS.get(stream_id)
    if s is not None:
        s.buffer.append(pcm)
        s.queue.put(pcm)


def finish(stream_id: str) -> None:
    """Mark the stream complete: unblock the live handler and retain the full
    WAV so subsequent Range/seek requests can be served."""
    with _LOCK:
        s = _STREAMS.get(stream_id)
    if s is None:
        return
    s.queue.put(_SENTINEL)
    wav = _complete_wav(s.sample_rate, b"".join(s.buffer))
    with _LOCK:
        _COMPLETE[stream_id] = wav
        while len(_COMPLETE) > _MAX_COMPLETE:
            _COMPLETE.popitem(last=False)
