import time, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
text="Bulls are absolutely demolishing resistance as Apple rockets past the golden cross with heavy volume!"
m=ChatterboxTTS.from_pretrained(device="cpu")
t0=time.time(); wav=m.generate(text, exaggeration=0.8, cfg_weight=0.5); gen=time.time()-t0
ta.save("tts_eval/samples/chatterbox.wav", wav, m.sr)
secs=wav.shape[-1]/m.sr
print(f"CHATTERBOX: gen={gen:.2f}s audio={secs:.2f}s RTF={gen/max(secs,.01):.2f} sr={m.sr}")
