# Infrastructure Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all infrastructure issues identified by the infra-review audit.

**Architecture:** Small personal project — Streamlit app with local LLM inference. No containers, no IaC. CI is GitHub Actions on ubuntu. Changes are targeted and minimal.

**Tech Stack:** Python 3.12, uv, GitHub Actions, ruff, pytest

---

## Findings Summary

| # | Severity | Category | Finding |
|---|----------|----------|---------|
| 1 | medium | CI/CD | `ubuntu-latest` is a floating tag — breaks reproducibility |
| 2 | medium | CI/CD | `ruff format --check` missing — formatting not enforced in CI |
| 3 | medium | CI/CD | CI uses raw `pip` instead of `uv`, bypassing `uv.lock` for most deps |
| 4 | low | CI/CD | `cache: pip` configured but uv provides better caching via `astral-sh/setup-uv` |
| 5 | low | Dev UX | No `fmt-check` Makefile target — no local equivalent of CI format gate |

**What is already well-configured (do not change):**
- `.env`/`.gitignore` secret handling is correct
- `_TICKER_SAFE_RE` XSS sanitization in app.py is correct
- `LLAMA_CPP_LOCK` concurrency design is correct
- `setup.sh` comment clearly warns AMD-only — no code change needed
- Test coverage is strong (63 tests)
- `set -euo pipefail` in setup.sh is correct

---

### Task 1: Pin ubuntu-latest to ubuntu-24.04

**Files:**
- Modify: `.github/workflows/ci.yml:11`

- [ ] **Step 1:** Edit `runs-on: ubuntu-latest` → `runs-on: ubuntu-24.04`
- [ ] **Step 2:** Verify file looks correct

---

### Task 2: Switch CI from pip to uv

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1:** Replace `actions/setup-python` + manual pip installs with `astral-sh/setup-uv` + `uv python install` + `uv pip install --system`
- [ ] **Step 2:** Keep CPU torch pre-install workaround (required: pyproject.toml sources torch from ROCm index which isn't available on standard runners)
- [ ] **Step 3:** Verify CI yaml is syntactically valid

---

### Task 3: Add ruff format check to CI

**Files:**
- Modify: `.github/workflows/ci.yml:48-49`

- [ ] **Step 1:** Add `ruff format --check .` after existing `ruff check .`
- [ ] **Step 2:** Verify lint step covers both check and format

---

### Task 4: Add fmt-check Makefile target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1:** Add `fmt-check` target that runs `ruff format --check .`
- [ ] **Step 2:** Add to `help` text
- [ ] **Step 3:** Add to `.PHONY`
