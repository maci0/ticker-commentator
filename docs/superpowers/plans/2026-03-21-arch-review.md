# Architecture Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all architectural issues identified by the arch-review audit.

**Architecture:** Small personal project with a clean layered pipeline (data → analysis → commentary → TTS → UI). Previous audits addressed CI/CD and performance. Remaining issues are confined to `experiments/` naming.

**Tech Stack:** Python 3.12, uv, Streamlit, llama-cpp-python, SNAC

---

## Executive Summary

**Overall architectural health: Good.** The previous infra-hardening and perf-fixes sessions resolved the CI/CD and performance issues. The module structure is clean with correct separation of concerns, no circular dependencies, well-defined public API via `__init__.py`, and strong test coverage (63 tests).

**Top structural themes:**
1. `experiments/` naming is inconsistent with the rest of the project and creates confusion
2. All other modules are correctly structured — no changes needed

**Top 3 highest ROI structural improvements:**
1. Rename `experiments/test_tts.py` → avoids naming conflict with `tests/test_tts.py`
2. Rename `experiments/orpheus-cpp.py` → aligns with PEP 8 and other experiment scripts
3. Rename `experiments/orpheus-podman-test.py` → same reason

---

## Structural Defects

None.

---

## Boundary Violations

None.

---

## Inconsistencies

### F-1: `experiments/test_tts.py` name conflicts with `tests/test_tts.py`
- **Severity:** Medium
- **Category:** Naming and Discoverability
- **Location:** `experiments/test_tts.py`
- **Confidence:** Confirmed
- **Why it matters:** `grep`, IDE file pickers, and `git log --all -- '*test_tts.py'` all conflate the two files. A new contributor reading "found test_tts.py" won't know which one is the real test suite entry.
- **Evidence:** Both files exist side-by-side; `experiments/` is excluded from pytest testpaths but the name still causes ambiguity.
- **Recommendation:** Rename to `experiments/integration_tts.py` — accurately describes purpose (integration/smoke test, not unit test).
- **Expected benefit:** Navigability, onboarding
- **Estimated effort:** Minutes
- **Blast radius:** 1 file rename, no imports affected (scripts only)

### F-2: Hyphenated filenames in `experiments/` — inconsistent with Python convention
- **Severity:** Low
- **Category:** Naming and Conventions
- **Location:** `experiments/orpheus-cpp.py`, `experiments/orpheus-podman-test.py`
- **Confidence:** Confirmed
- **Why it matters:** `bench_snac.py` and `check_snac_wav.py` use underscores. PEP 8 recommends underscores for Python filenames. Hyphens in filenames are non-standard (can't be imported as modules; shell completion requires quoting).
- **Evidence:** `orpheus-cpp.py` and `orpheus-podman-test.py` use hyphens; `bench_snac.py`, `check_snac_wav.py` use underscores.
- **Recommendation:** Rename to `experiments/orpheus_cpp.py` and `experiments/orpheus_podman_test.py`.
- **Expected benefit:** Navigability, consistency
- **Estimated effort:** Minutes
- **Blast radius:** 2 file renames, no imports affected (scripts only)

---

## Naming and Discoverability Issues

See F-1 and F-2 above.

---

## Colocation Problems

None.

---

## Scaling Risks

None significant at this project scale.

---

## Quick Wins

- [x] Rename `experiments/test_tts.py` → `experiments/integration_tts.py` (F-1)
- [x] Rename `experiments/orpheus-cpp.py` → `experiments/orpheus_cpp.py` (F-2)
- [x] Rename `experiments/orpheus-podman-test.py` → `experiments/orpheus_podman_test.py` (F-2)

---

## What is Already Well-Configured (do not change)

- `load_dotenv()` in `__init__.py`: intentional design; documented in CLAUDE.md; necessary for module-level `os.getenv()` calls in `commentary.py` and `tts.py`
- `_inject_emotion_tags(dict[str, Any])`: intentional robustness; uses `.get()` with defaults to tolerate partial dicts from tests
- `app.py` at 509 lines: appropriate for Streamlit's programming model; splitting would require complex callback/state passing that would be worse
- `_config.py` underscore prefix: correct package-private convention; not re-exported from `__init__.py`
- `llama_lock.py` as a single-variable module: prevents circular imports between `commentary.py` and `tts.py`; correct design
- Module-level `os.getenv()` in `commentary.py` and `tts.py`: each module owns its own config; clean encapsulation
- CI hardening (`ubuntu-24.04`, `uv`, `ruff format --check`): already done
- Performance fixes (`np.where`, `del pending[:]`, pre-computed emotion pools): already done
- TypedDicts, `__all__`, `__init__.py` re-exports: clean API boundaries throughout
- Test coverage (63 tests across 5 modules): strong, no gaps

---

## Dependency Map

```
app.py → commentator/__init__.py
           ├── analysis.py (AnalysisResult, AnalysisError, analyze_stock)
           ├── commentary.py (generate_commentary)
           │     ├── analysis.py (AnalysisResult type)
           │     ├── _config.py (env_int, env_float, parse_tensor_split)
           │     └── llama_lock.py (LLAMA_CPP_LOCK)
           ├── data.py (fetch_stock_data, fetch_stock_info, StockInfo)
           └── tts.py (iter_audio_chunks, pcm_chunks_to_wav, text_to_speech, ...)
                 ├── _config.py (env_int, env_float, parse_tensor_split)
                 └── llama_lock.py (LLAMA_CPP_LOCK)
```

No cycles. Dependency direction is correct: UI → Application → Domain/Infrastructure → Utilities.

---

## Restructuring Plan

1. **Rename experiment files** — low risk, high navigability gain (see Tasks 1-3 below)
2. No further restructuring needed at this scale.

---

## Open Questions

None — all structural choices are either clearly intentional or clearly fixable.

---

## Tasks

### Task 1: Rename `experiments/test_tts.py` → `experiments/integration_tts.py`

**Files:**
- Rename: `experiments/test_tts.py` → `experiments/integration_tts.py`

- [ ] **Step 1:** `git mv experiments/test_tts.py experiments/integration_tts.py`
- [ ] **Step 2:** Verify rename

---

### Task 2: Rename `experiments/orpheus-cpp.py` → `experiments/orpheus_cpp.py`

**Files:**
- Rename: `experiments/orpheus-cpp.py` → `experiments/orpheus_cpp.py`

- [ ] **Step 1:** `git mv "experiments/orpheus-cpp.py" experiments/orpheus_cpp.py`
- [ ] **Step 2:** Verify rename

---

### Task 3: Rename `experiments/orpheus-podman-test.py` → `experiments/orpheus_podman_test.py`

**Files:**
- Rename: `experiments/orpheus-podman-test.py` → `experiments/orpheus_podman_test.py`

- [ ] **Step 1:** `git mv "experiments/orpheus-podman-test.py" experiments/orpheus_podman_test.py`
- [ ] **Step 2:** Verify rename

---

### Task 4: Run tests

- [ ] **Step 1:** `uv run pytest tests/ -q`
- [ ] **Step 2:** Confirm all pass
