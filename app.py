from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import io
import numpy as np
from scipy.io import wavfile

app = FastAPI(title="Vocal Accompaniment Generator", version="0.1")

# Serve UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/ui', StaticFiles(directory=STATIC_DIR, html=True), name='ui')

@app.get('/')
async def root_ui():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

# --- DSP helpers ---

def detect_voice(audio: np.ndarray, sr: int) -> bool:
    if audio.size == 0:
        return False
    energy = np.sqrt(np.mean(audio**2))
    return energy > 0.01

A4 = 440.0

def estimate_key_freq(audio: np.ndarray, sr: int) -> float:
    # naive pitch tracker: autocorrelation peak
    if audio.size < sr//2:
        return A4
    segment = audio[:sr] if audio.size >= sr else audio
    segment = segment - np.mean(segment)
    corr = np.correlate(segment, segment, mode='full')[len(segment)-1:]
    corr[0:int(sr/1000)] = 0
    peak = np.argmax(corr)
    if peak <= 0:
        return A4
    f0 = sr / peak
    return float(np.clip(f0, 80.0, 800.0))

def estimate_tempo(audio: np.ndarray, sr: int) -> float:
    # crude onset energy tempo estimate
    hop = 1024
    frames = max(1, (len(audio)-hop)//hop)
    if frames < 4:
        return 100.0
    env = [np.sum(audio[i*hop:(i+1)*hop]**2) for i in range(frames)]
    env = np.array(env)
    env = (env - env.mean())/(env.std()+1e-8)
    corr = np.correlate(env, env, mode='full')[len(env)-1:]
    corr[:2] = 0
    lag = np.argmax(corr[2:400])+2
    bpm = 60.0 * sr / (lag*hop)
    return float(np.clip(bpm, 60.0, 180.0))

STYLE_SETTINGS = {
    'afrobeat': {'tempo_mult': 1.05, 'chords': 'minor'},
    'hi_life': {'tempo_mult': 1.10, 'chords': 'major'},
    'hip_hop': {'tempo_mult': 0.95, 'chords': 'minor'},
    'calypso': {'tempo_mult': 1.08, 'chords': 'major'},
    'reggae': {'tempo_mult': 0.90, 'chords': 'major'},
    'rnb': {'tempo_mult': 0.98, 'chords': 'minor'},
    'classical': {'tempo_mult': 1.00, 'chords': 'major'},
    'davido_mix': {'tempo_mult': 1.07, 'chords': 'minor'},
    'nigerian_gospel': {'tempo_mult': 1.05, 'chords': 'major'},
}

def synthesize_accompaniment(duration: float, sr: int, key_freq: float, tempo: float, style: str | None = None) -> np.ndarray:
    n = int(sr*duration)
    t = np.linspace(0, duration, n, endpoint=False)
    major = np.array([1.0, 1.25, 1.5])
    minor = np.array([1.0, 1.2, 1.5])
    chords = minor if (style and STYLE_SETTINGS.get(style, {}).get('chords')=='minor') else major
    base = np.sin(2*np.pi*key_freq*t)*0.15
    pad = sum(np.sin(2*np.pi*(key_freq*c)*t) for c in chords)*0.07
    beat = np.zeros_like(t)
    beats_per_sec = tempo/60.0
    for k in range(int(duration*beats_per_sec)):
        idx = int(k/beats_per_sec*sr)
        beat[idx:idx+int(0.02*sr)] += 0.4*np.hanning(int(0.02*sr))
    drum = beat
    mix = base + pad + drum
    return np.clip(mix, -1.0, 1.0).astype(np.float32)

@app.post('/generate')
async def generate(request: Request, style: str = Query('afrobeat')):
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail='No audio data')
    try:
        sr, audio = wavfile.read(io.BytesIO(raw))
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype.kind in 'iu':
            maxv = np.iinfo(audio.dtype).max
            audio = audio / maxv
        else:
            audio = audio / (np.max(np.abs(audio))+1e-9)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid WAV: {e}')

    if not detect_voice(audio, sr):
        raise HTTPException(status_code=422, detail='No clear voice detected')

    f0 = estimate_key_freq(audio, sr)
    bpm = estimate_tempo(audio, sr)
    s = STYLE_SETTINGS.get(style, {'tempo_mult':1.0})
    bpm *= s.get('tempo_mult', 1.0)

    dur = max(3.0, min(30.0, len(audio)/sr))
    accomp = synthesize_accompaniment(dur, sr, f0, bpm, style)

    # simple mix under vocal
    mix = 0.75*audio[:len(accomp)] + 0.5*accomp[:len(audio)]
    mix = np.clip(mix, -1.0, 1.0)

    out = io.BytesIO()
    wavfile.write(out, sr, (mix*32767).astype(np.int16))
    out.seek(0)
    return FileResponse(out, media_type='audio/wav', filename='accompaniment.wav')
