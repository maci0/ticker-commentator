import time, soundfile as sf
from kokoro import KPipeline
text="Bulls are absolutely demolishing resistance as Apple rockets past the golden cross with heavy volume!"
pipe=KPipeline(lang_code='a')
t0=time.time(); audio_all=[]
for gs,ps,audio in pipe(text, voice='am_michael'):
    audio_all.append(audio)
import numpy as np
audio=np.concatenate(audio_all)
gen=time.time()-t0
sf.write("tts_eval/samples/kokoro_82m_ammichael.wav", audio, 24000)
secs=len(audio)/24000
print(f"KOKORO-82M: gen={gen:.2f}s audio={secs:.2f}s RTF={gen/max(secs,0.01):.2f}")
