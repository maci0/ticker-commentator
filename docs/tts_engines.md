# TTS engines

The app supports three text-to-speech engines, selected with the `TTS_ENGINE`
env var. **Orpheus** is the default and the only one installed by the project's
`uv sync` — it is the fastest and keeps the emotion-tag gimmick.

`chatterbox` and `kokoro` are **opt-in alternatives**. They are *not* in
`pyproject.toml` on purpose: each pins torch/transformers versions that conflict
with the project (and with each other), so adding them as extras would break
`uv sync`/`uv lock`. Install the one you want into its **own venv** and run the
app from there.

## Benchmark (RX 7900 XTX, gfx1100; one sports-commentary line)

Objective metrics from `tts_eval/` — UTMOS = neural naturalness MOS (1–5, higher
better), WER = Whisper word-error vs the script (lower better), pitchSD =
semitone std of voiced F0 (higher = more expressive intonation), RTF = seconds
of compute per second of audio (lower = faster).

| engine | UTMOS | WER | pitchSD | warm RTF | notes |
|--------|------:|----:|--------:|---------:|-------|
| orpheus-3B (clean) | 4.26 | 7% | 5.03 | **0.62** (GPU) | default; expressive + faster than real time |
| orpheus-3B (+emotion tags) | 3.30 | 27% | 4.24 | 0.62 | tag injection hurts clarity — see note below |
| **chatterbox** (exa 0.8) | **4.37** | 7% | **5.36** | 1.51 (GPU) | best quality + most expressive; slower than real time |
| **kokoro-82M** | 4.40 | 7% | 3.85 | **0.20** (CPU) | fastest, very clean, but emotionally flat |

Note: the Orpheus emotion-tag injection (`<laugh>`, `<gasp>`, …) measurably
lowers naturalness and intelligibility. Lower `COMMENTARY_EMOTE_CHANCE_1/2` if
you want cleaner Orpheus output.

## Chatterbox (most expressive)

Runs on ROCm. torch 2.6 pin is overridden to the project's rocm build.

```bash
uv venv /opt/tts-chatterbox --python 3.12
printf 'torch==2.11.0+rocm7.1\ntorchaudio==2.11.0+rocm7.1\ntorchvision==0.26.0+rocm7.1\n' > /tmp/cbx_override.txt
uv pip install --python /opt/tts-chatterbox \
  --extra-index-url https://download.pytorch.org/whl/rocm7.1 \
  --override /tmp/cbx_override.txt \
  chatterbox-tts soundfile
# run the app from this env:
TTS_ENGINE=chatterbox HIP_VISIBLE_DEVICES=0 /opt/tts-chatterbox/bin/python -m streamlit run app.py
```

Tunables: `CHATTERBOX_EXAGGERATION` (default 0.8; >1.0 hurt naturalness in the
eval), `CHATTERBOX_CFG` (default 0.5).

## Kokoro (fastest)

```bash
uv venv /opt/tts-kokoro --python 3.12
uv pip install --python /opt/tts-kokoro \
  "kokoro==0.9.4" "torch==2.4.1" "torchvision==0.19.1" "transformers==4.47.1" "numpy<2" soundfile
TTS_ENGINE=kokoro /opt/tts-kokoro/bin/python -m streamlit run app.py
```

Tunables: `KOKORO_VOICE` (default `am_michael`), `KOKORO_LANG` (default `a`).

## Qwen3-TTS (multilingual, named speakers)

Open weights from Alibaba (https://github.com/QwenLM/Qwen3-TTS). Runs on ROCm;
24 kHz output, 9 named speakers, 8 languages, plus a natural-language `instruct`
style control. Heavy (1.7B, slow: cold RTF ~5 on the 7900 XTX).

Unlike chatterbox/kokoro, qwen-tts **coexists in the main env** — it only adds
transformers/accelerate, which the project core doesn't use, so torch and the
HIP llama.cpp build are untouched. Just enable the extra:

```bash
uv sync --extra qwen
TTS_ENGINE=qwen HIP_VISIBLE_DEVICES=0 uv run streamlit run app.py
```

Tunables: `QWEN_TTS_REPO` (default `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`),
`QWEN_TTS_SPEAKER` (aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu,
vivian), `QWEN_TTS_LANGUAGE` (auto/english/chinese/…), `QWEN_TTS_INSTRUCT`
(optional, e.g. "speak excitedly").

## How it works

`commentator/tts.py` reads `TTS_ENGINE`. For `orpheus` it uses the built-in
llama.cpp + SNAC path. Otherwise it routes to `commentator/tts_engines.py`,
which lazily imports the chosen library (raising a clear install hint if it is
missing) and yields 16-bit mono PCM at 24 kHz — the same format the Orpheus path
produces, so the rest of the app is unchanged.
