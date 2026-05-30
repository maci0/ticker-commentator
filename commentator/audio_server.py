"""Local streaming-WAV HTTP server for progressive audio playback.

The default playback path waits for the whole clip to synthesize (~4 s) before
the browser gets a complete WAV. With streaming enabled, an `<audio>` element
points at this server's `/audio/<id>.wav` endpoint; PCM chunks are fed in as
they decode and streamed to the browser, so playback starts at the first chunk
(~1 s) instead of the full duration.

Single background ThreadingHTTPServer bound to 127.0.0.1 (local playback only).
Each clip gets an id + a bounded queue; the handler writes a streaming WAV header
then drains the queue until a sentinel. Opt-in via STREAM_AUDIO=1.
"""

import logging
import os
import queue
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

_SENTINEL = object()
_DEFAULT_SAMPLE_RATE = 24000

_LOCK = threading.Lock()
_STREAMS: "dict[str, queue.Queue]" = {}
_SERVER: "ThreadingHTTPServer | None" = None
_PORT: int = 0
_COUNTER = 0


def _streaming_wav_header(sample_rate: int) -> bytes:
    """44-byte WAV header for a stream of unknown length: the RIFF/data sizes are
    set to a large sentinel so browsers keep reading until the connection closes
    (16-bit mono PCM)."""
    big = 0x7FFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", big)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", big)
    )


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
            q = _STREAMS.get(stream_id)
        if q is None:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        sr = int(self.headers.get("X-Sample-Rate", _DEFAULT_SAMPLE_RATE) or _DEFAULT_SAMPLE_RATE)
        try:
            self.wfile.write(_streaming_wav_header(sr))
            self.wfile.flush()
            while True:
                chunk = q.get()
                if chunk is _SENTINEL:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed/seeked — normal
        finally:
            with _LOCK:
                _STREAMS.pop(stream_id, None)


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


def open_stream() -> "tuple[str, int] | None":
    """Register a new audio stream. Returns (stream_id, port) or None if the
    server is unavailable."""
    global _COUNTER
    port = _ensure_server()
    if not port:
        return None
    with _LOCK:
        _COUNTER += 1
        stream_id = f"s{_COUNTER}"
        _STREAMS[stream_id] = queue.Queue(maxsize=256)
    return stream_id, port


def feed(stream_id: str, pcm: bytes) -> None:
    with _LOCK:
        q = _STREAMS.get(stream_id)
    if q is not None:
        q.put(pcm)


def finish(stream_id: str) -> None:
    with _LOCK:
        q = _STREAMS.get(stream_id)
    if q is not None:
        q.put(_SENTINEL)
