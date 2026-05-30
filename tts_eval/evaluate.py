"""Objective TTS quality eval: UTMOS (naturalness MOS 1-5) + Whisper WER.

Reads every WAV in tts_eval/samples/ and prints naturalness + intelligibility
so models can be ranked without a human listening.
"""
import glob
import re

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

import jiwer

TARGET = (
    "Bulls are absolutely demolishing resistance as Apple rockets past "
    "the golden cross with heavy volume!"
)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


print("loading UTMOS + Whisper...")
utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
asr = WhisperModel("base.en", device="cpu", compute_type="int8")

def expressiveness(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Rough prosody proxies: pitch range (semitone std of voiced F0) and energy
    dynamics (coeff. of variation of RMS). Higher = more dynamic/expressive."""
    f0, voiced, _ = librosa.pyin(
        y, fmin=70, fmax=400, sr=sr, frame_length=2048
    )
    f0v = f0[~np.isnan(f0)]
    pitch_semitone_std = float(np.std(12 * np.log2(f0v / np.median(f0v)))) if f0v.size > 5 else 0.0
    rms = librosa.feature.rms(y=y)[0]
    energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-9))
    return pitch_semitone_std, energy_cv


rows = []
for wav in sorted(glob.glob("tts_eval/samples/*.wav")):
    y, sr = sf.read(wav)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype("float32")
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    with torch.no_grad():
        mos = utmos(torch.from_numpy(y16).unsqueeze(0), 16000).item()
    segs, _ = asr.transcribe(wav, language="en")
    hyp = " ".join(s.text for s in segs).strip()
    wer = jiwer.wer(norm(TARGET), norm(hyp)) * 100
    dur = len(y) / sr
    pitch_std, energy_cv = expressiveness(y, sr)
    name = wav.split("/")[-1]
    rows.append((name, mos, wer, dur, pitch_std, energy_cv, hyp))

print("\n=== RESULTS ===")
print(f"{'sample':34s} {'UTMOS':>6s} {'WER%':>6s} {'dur':>5s} {'pitchSD':>7s} {'engCV':>6s}")
for name, mos, wer, dur, pstd, ecv, hyp in rows:
    print(f"{name:34s} {mos:6.2f} {wer:6.0f} {dur:5.1f} {pstd:7.2f} {ecv:6.2f}")
print("\n(pitchSD = semitone std of voiced F0; engCV = RMS energy coeff-of-variation; higher = more dynamic/expressive)")
for name, *_rest, hyp in rows:
    print(f"  {name}: {hyp!r}")
