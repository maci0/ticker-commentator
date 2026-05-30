"""ticker-commentator: real-time stock chart commentary with TTS.

Importing any submodule triggers load_dotenv() here so that module-level
constants in commentary.py and tts.py (which call os.getenv at import time)
always see the values from .env, regardless of what imported the package first.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Restrict ROCm/CUDA to a single GPU before torch or llama.cpp initialize their
# backends. Without this, llama.cpp splits layers across every visible device —
# including a weak integrated GPU (e.g. the Ryzen gfx1036 iGPU) whose kernels are
# absent from a discrete-GPU-only build, which crashes inference. Default to
# device 0 (the discrete GPU); override GPU_DEVICE in .env for multi-GPU setups,
# or set it to an empty string to leave device selection untouched.
_gpu_device = os.environ.get("GPU_DEVICE", "0")
if _gpu_device != "" and "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = _gpu_device
if _gpu_device != "" and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_device

from commentator.analysis import AnalysisError, AnalysisResult, analyze_stock  # noqa: E402
from commentator.commentary import generate_commentary  # noqa: E402
from commentator.data import StockInfo, fetch_stock_data, fetch_stock_info  # noqa: E402
from commentator.tts import (  # noqa: E402
    SAMPLE_RATE,
    VALID_VOICES,
    iter_audio_chunks,
    pcm_chunks_to_wav,
    text_to_speech,
)

__all__ = [
    # analysis
    "analyze_stock",
    "AnalysisResult",
    "AnalysisError",
    # commentary
    "generate_commentary",
    # data
    "fetch_stock_data",
    "fetch_stock_info",
    "StockInfo",
    # tts
    "iter_audio_chunks",
    "pcm_chunks_to_wav",
    "text_to_speech",
    "SAMPLE_RATE",
    "VALID_VOICES",
]
