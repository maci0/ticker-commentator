"""Pytest session setup.

Disable GPU visibility before torch or llama.cpp can initialize their backends.
The unit tests exercise pure functions only and never need a GPU; importing
torch with the ROCm runtime active can crash the test runner during HSA init.
Forcing an empty device list keeps torch on CPU and makes test runs
deterministic across machines.

These assignments run at import time, before any test module (and therefore
before `commentator/__init__.py`, torch, or llama_cpp) is imported.
"""

import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
