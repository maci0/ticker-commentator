# Performance Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all genuine performance issues found during the perf-review audit.

**Architecture:** Targeted edits to three files; no API or behavioral changes. All fixes are purely internal optimizations.

**Tech Stack:** Python 3.11+, pandas, numpy, PyTorch, llama-cpp-python, Streamlit

---

## Performance Audit Summary

The codebase is already well-optimized (lazy LLM init, tail-sliced rolling computations, yfinance caching in session state, module-level regex compilation). The bottlenecks below are the remaining actionable items.

### Confirmed Issues

| # | Severity | File | Description |
|---|----------|------|-------------|
| 1 | Low | `app.py:316` | `pd.Series.map({True, False})` → `np.where` — avoids intermediate Series and dict-lookup overhead |
| 2 | Low | `tts.py:316,326` | `pending_tokens = pending_tokens[n:]` creates a new list on every batch decode → `del pending_tokens[:n]` in-place |
| 3 | Low | `commentary.py:171` | `pool = pool + _SURPRISE_TAGS` allocates a new list on every call → pre-compute combined pools at module level |

---

### Task 1: `app.py` — replace `pd.map` with `np.where` for chart colors

**Files:**
- Modify: `app.py:7-12` (imports), `app.py:316`

- [ ] **Step 1: Add `import numpy as np` to imports**

Add after `import re` in the import block.

- [ ] **Step 2: Replace the colors line**

```python
# Old
colors = (df["Close"] >= df["Open"]).map({True: "#26a69a", False: "#ef5350"}).tolist()

# New
colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350").tolist()
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "perf: use np.where for chart colors instead of pd.Series.map"
```

---

### Task 2: `tts.py` — in-place list mutation to avoid copies

**Files:**
- Modify: `commentator/tts.py:316`, `commentator/tts.py:326`

- [ ] **Step 1: Replace slice-reassign with in-place delete in the main decode loop**

```python
# Old (line 316)
pending_tokens = pending_tokens[_CHUNK_FRAMES * 7 :]

# New
del pending_tokens[: _CHUNK_FRAMES * 7]
```

- [ ] **Step 2: Replace slice-reassign with in-place delete in the drain loop**

```python
# Old (line 326)
pending_tokens = pending_tokens[take:]

# New
del pending_tokens[:take]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add commentator/tts.py
git commit -m "perf: use in-place del instead of slice-reassign for pending_tokens"
```

---

### Task 3: `commentary.py` — pre-compute combined emotion tag pools

**Files:**
- Modify: `commentator/commentary.py` (add 3 module-level constants, simplify `_inject_emotion_tags`)

- [ ] **Step 1: Add pre-computed pool constants after the existing pool lists**

```python
_POSITIVE_TAGS_SURPRISED = _POSITIVE_TAGS + _SURPRISE_TAGS
_NEGATIVE_TAGS_SURPRISED = _NEGATIVE_TAGS + _SURPRISE_TAGS
_NEUTRAL_TAGS_SURPRISED = _NEUTRAL_TAGS + _SURPRISE_TAGS
```

- [ ] **Step 2: Replace the pool-selection logic in `_inject_emotion_tags`**

```python
# Old
if trend == "bullish":
    pool = _POSITIVE_TAGS
elif trend == "bearish":
    pool = _NEGATIVE_TAGS
else:
    pool = _NEUTRAL_TAGS

if change > 3 or volatility == "high":
    pool = pool + _SURPRISE_TAGS

# New
high_drama = change > 3 or volatility == "high"
if trend == "bullish":
    pool = _POSITIVE_TAGS_SURPRISED if high_drama else _POSITIVE_TAGS
elif trend == "bearish":
    pool = _NEGATIVE_TAGS_SURPRISED if high_drama else _NEGATIVE_TAGS
else:
    pool = _NEUTRAL_TAGS_SURPRISED if high_drama else _NEUTRAL_TAGS
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add commentator/commentary.py
git commit -m "perf: pre-compute combined emotion tag pools to avoid per-call list alloc"
```
