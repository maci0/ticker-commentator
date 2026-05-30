import time, torch, soundfile as sf
from chatterbox.tts import ChatterboxTTS
text="Bulls are absolutely demolishing resistance as Apple rockets past the golden cross with heavy volume!"
t0=time.time(); m=ChatterboxTTS.from_pretrained(device="cuda"); print(f"LOAD {time.time()-t0:.1f}s")
def gen(tag, exa, cfg, out):
    t0=time.time(); wav=m.generate(text, exaggeration=exa, cfg_weight=cfg); torch.cuda.synchronize(); dt=time.time()-t0
    secs=wav.shape[-1]/m.sr
    print(f"CHATTERBOX-GPU {tag}: gen={dt:.2f}s audio={secs:.2f}s RTF={dt/max(secs,.01):.2f}")
    sf.write(out, wav.squeeze(0).cpu().numpy(), m.sr)
gen("cold",0.8,0.5,"tts_eval/samples/_cbx_cold.wav")
gen("warm",0.8,0.5,"tts_eval/samples/chatterbox_gpu.wav")
gen("dramatic",1.3,0.4,"tts_eval/samples/chatterbox_gpu_dramatic.wav")
