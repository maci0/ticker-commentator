"""Shared mutex for llama.cpp inference.

Concurrent llama.cpp inference in the same process causes GPU/BLAS contention.
Commentary and TTS each hold a separate Llama instance but share this lock so
only one runs at a time.
"""

import threading

LLAMA_CPP_LOCK = threading.Lock()
