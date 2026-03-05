uv sync
uv pip uninstall llama-cpp-python
CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100" FORCE_CMAKE=1 uv pip install --no-cache-dir llama-cpp-python
