.PHONY: help run test lint fmt fmt-check check audit setup install clean

help:
	@echo "Available targets:"
	@echo "  run         Start the Streamlit app"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Lint with ruff"
	@echo "  fmt         Format with ruff"
	@echo "  fmt-check   Check formatting without modifying files"
	@echo "  check       Lint, format check, and test"
	@echo "  audit       Scan dependencies for known CVEs"
	@echo "  install     Install dependencies"
	@echo "  setup       AMD ROCm GPU build"
	@echo "  clean       Remove build and cache artifacts"

run:
	uv run streamlit run app.py

test:
	uv run pytest --tb=short --cov=commentator --cov-report=term-missing

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

check: lint fmt-check test

# pip-audit cannot resolve ROCm-local pins (torch==…+rocm7.1, triton-rocm, …)
# and the editable local package. Audit third-party deps only on Python 3.12.
# diskcache (via llama-cpp-python) has PYSEC-2026-2447 with no fixed release yet
# (latest on PyPI is still 5.6.3); re-check when a patch ships.
audit:
	@uv export --no-hashes --format requirements-txt --no-emit-project \
		| grep -Eiv '^(torch|torchaudio|torchvision|triton)([-=<>! ]|$$)' \
		| grep -Ev '^-e |^file:' \
		> /tmp/ticker-commentator-audit-req.txt
	uv tool run --python 3.12 pip-audit \
		--ignore-vuln PYSEC-2026-2447 \
		-r /tmp/ticker-commentator-audit-req.txt

# AMD ROCm build — see setup.sh for details
setup:
	./setup.sh

install:
	uv sync

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
