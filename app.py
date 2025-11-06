import os
import io
import tempfile
import numpy as np
import requests
import install_ffmpeg  # noqa

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from scipy.io import wavfile
import soundfile as sf
import pretty_midi

from pydub import AudioSegment
from pydub.utils import which

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
app = FastAPI(title="Vocal Accompaniment Generator", version="0.2")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")
SOUNDFONT_URL = "https://archive.org/download/fluidr3gm/FluidR3_GM.sf2"

# make pydub find ffmpeg (we installed it in Dockerfile)
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"

# ---------------------------------------------------------
# UTIL: ensure soundfont exists
# ---------------------------------------------------------
def ensure_soundfont():
    os.makedirs(SOUNDFONT_DIR, exist_ok=True)
    if not os.path.exists(SOUNDFONT_PATH):
        print("SoundFont not found. Downloading FluidR3_GM.sf2 ...")
        with requests.get(SOUNDFONT_URL, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(SOUNDFONT_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("SoundFont downloaded to", SOUNDFONT_PATH)

ensure_soundfont()


# ---------------------------------------------------------
# DSP helpers (from your earlier version, slightly cleaned)
# ---------------------------------------------------------
def detect_voice(audio: np.ndarray, sr: int) -> bool:
    if audio.size == 0:
        return False
    energy = np.sqrt(np.mean(audio**2))
    return energy > 0.01

A4 = 440.0

def estimate_key_freq(audio: np.ndarray, sr: int) -> float:
    if audio.size < sr // 2:
        return A4
    segment = audio[:sr] if audio.size >= sr else audio
    segment = segment - np.mean(segment)
    corr = np.correlate(segment, segment, mode="full")[len(segment) - 1 :]
    corr[0 : int(sr / 1000)] = 0
    peak = np.argmax(corr)
    if peak <= 0:
        return A4
    f0 = sr / peak
    return float(np.clip(f0, 80.0, 800.0))

def estimate_tempo(audio: np.ndarray, sr: int) -> float:
    hop = 1024
    frames = max(1, (len(audio) - hop) // hop)
    if frames < 4:
        return 100.0
    env = [np.sum(audio[i * hop : (i + 1) * hop] ** 2) for i in range(frames)]
    env = np.array(env)
    env = (env - env.mean()) / (env.std() + 1e-8)
    corr = np.correlate(env, env, mode="full")[len(env) - 1 :]
    corr[:2] = 0
    lag = np.argmax(corr[2:400]) + 2
    bpm = 60.0 * sr / (lag * hop)
    return float(np.clip(bpm, 60.0, 180.0))

# ---------------------------------------------------------
# Style definitions (you can tune these later)
# ---------------------------------------------------------
STYLE_SETTINGS = {
    "afrobeat": {
        "tempo_mult": 1.05,
        "chords": ["i", "iv", "v", "iv"],
        "drum_pattern": "kick-snare",
    },
    "nigerian_gospel": {
        "tempo_mult": 1.03,
        "chords": ["i", "vi", "iv", "v"],
        "drum_pattern": "four-on-floor",
    },
    "reggae": {
        "tempo_mult": 0.90,
        "chords": ["i", "v", "iv", "v"],
        "drum_pattern": "one-drop",
    },
    "rnb": {
        "tempo_mult": 0.95,
        "chords": ["i", "vi", "iv", "v"],
        "drum_pattern": "slow-groove",
    },
    "hip_hop": {
        "tempo_mult": 0.90,
        "chords": ["i", "iv", "v", "iv"],
        "drum_pattern": "boom-bap",
    },
}

def midi_note_from_freq(freq: float) -> int:
    return int(69 + 12 * np.log2(freq / 440.0))

def build_chord_progression(root_midi: int, style: str, bars: int) -> list[int]:
    style_def = STYLE_SETTINGS.get(style, STYLE_SETTINGS["afrobeat"])
    degrees = style_def["chords"]
    prog = []
    for i in range(bars):
        deg = degrees[i % len(degrees)]
        if deg == "i":
            prog.append(root_midi)
        elif deg == "iv":
            prog.append(root_midi + 5)
        elif deg == "v":
            prog.append(root_midi + 7)
        elif deg == "vi":
            prog.append(root_midi + 9)
        else:
            prog.append(root_midi)
    return prog

# ---------------------------------------------------------
# Instrument render
# ---------------------------------------------------------
def render_midi_band(sr: int, duration: float, bpm: float, root_freq: float, style: str,
                     use_piano=True, use_bass=True, use_drums=True) -> np.ndarray:
    root_midi = midi_note_from_freq(root_freq)
    bars = int(np.ceil(duration / (60.0 / bpm * 4))) + 1
    progression = build_chord_progression(root_midi, style, bars=bars)

    pm = pretty_midi.PrettyMIDI()
    seconds_per_bar = 60.0 / bpm * 4

    # Piano
    if use_piano:
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand
        t = 0.0
        for chord_root in progression:
        
    # triad
        for n in [chord_root, chord_root + 4, chord_root + 7]:
                note = pretty_midi.Note(velocity=85, pitch=n, start=t, end=t + seconds_per_bar * 0.95)
                piano.notes.append(note)
            t += seconds_per_bar
            if t > duration:
                break
        pm.instruments.append(piano)

    # Bass
    if use_bass:
        bass = pretty_midi.Instrument(program=32)  # Acoustic Bass
        t = 0.0
        for chord_root in progression:
            note = pretty_midi.Note(velocity=100, pitch=chord_root - 12,
                                    start=t, end=t + seconds_per_bar * 0.9)
            bass.notes.append(note)
            t += seconds_per_bar
            if t > duration:
                break
        # Drums
    if use_drums:
        drum = pretty_midi.Instrument(is_drum=True)
        beat = 60.0 / bpm
        t = 0.0
        pattern = STYLE_SETTINGS.get(style, {}).get("drum_pattern", "kick-snare")
        while t < duration:
            # kick
            kick = pretty_midi.Note(velocity=105, pitch=36, start=t, end=t + 0.1)
            drum.notes.append(kick)
            if pattern in ("kick-snare", "boom-bap", "slow-groove"):
                sn = pretty_midi.Note(velocity=110, pitch=38, start=t + beat, end=t + beat + 0.1)
                drum.notes.append(sn)
            if pattern == "four-on-floor":
                # extra kicks
                for off in [beat, beat * 2, beat * 3]:
                    k2 = pretty_midi.Note(velocity=90, pitch=36, start=t + off, end=t + off + 0.08)
                    drum.notes.append(k2)
            t += beat * 2
        pm.instruments.append(drum)
        # fluidsynth render to wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        pm.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH, filename=tmpwav.name)
        accomp, _ = sf.read(tmpwav.name)
    return accomp

# ---------------------------------------------------------
# Helper: load any audio → float32 mono, sr
# ---------------------------------------------------------
def load_audio_from_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    # try WAV fast path
    try:
        sr, audio = wavfile.read(io.BytesIO(raw))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype.kind in "iu":
            maxv = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / maxv
        else:
            audio = audio.astype(np.float32)
        return audio, sr
    except Exception:
        pass

    # fallback to pydub (mp3/m4a)
    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    audio = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return audio, sr

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
async def root_ui():
    if os.path.exists(os.path.join(STATIC_DIR, "index.html")):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    return {"detail": "UI not found, but API is running."}

@app.post("/generate")
async def generate(request: Request,
                   style: str = Query("afrobeat"),
                   piano: bool = Query(True),
                   bass: bool = Query(True),
                   drums: bool = Query(True)):
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio data received")
      audio, sr = load_audio_from_bytes(raw)

    if not detect_voice(audio, sr):
        # we can still proceed, but tell the user
        print("Warning: low vocal energy detected, generating anyway.")

    f0 = estimate_key_freq(audio, sr)
    bpm = estimate_tempo(audio, sr)
    style_def = STYLE_SETTINGS.get(style, {})
    bpm *= style_def.get("tempo_mult", 1.0)

    duration = len(audio) / sr
    accomp = render_midi_band(sr, duration, bpm, f0, style,
                              use_piano=piano, use_bass=bass, use_drums=drums)

# mix
    min_len = min(len(audio), len(accomp))
    mix = 0.9 * audio[:min_len] + 0.9 * accomp[:min_len]
    mix = np.clip(mix, -1.0, 1.0)

    out_buf = io.BytesIO()
    wavfile.write(out_buf, sr, (mix * 32767).astype(np.int16))
    out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
    )


STYLE_SETTINGS = {
    "afrobeat": {"tempo_mult": 1.05, "chords": "minor"},
    "hi_life": {"tempo_mult": 1.10, "chords": "major"},
    "hip_hop": {"tempo_mult": 0.95, "chords": "minor"},
    "calypso": {"tempo_mult": 1.08, "chords": "major"},
    "reggae": {"tempo_mult": 0.90, "chords": "major"},
    "rnb": {"tempo_mult": 0.98, "chords": "minor"},
    "classical": {"tempo_mult": 1.00, "chords": "major"},
    "davido_mix": {"tempo_mult": 1.07, "chords": "minor"},
    "nigerian_gospel": {"tempo_mult": 1.05, "chords": "major"},
}


def synthesize_accompaniment(
    duration: float, sr: int, key_freq: float, tempo: float, style: str | None = None
) -> np.ndarray:
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    major = np.array([1.0, 1.25, 1.5])
    minor = np.array([1.0, 1.2, 1.5])
    chords = (
        minor
        if (style and STYLE_SETTINGS.get(style, {}).get("chords") == "minor")
        else major
    )
    base = np.sin(2 * np.pi * key_freq * t) * 0.15
    pad = sum(np.sin(2 * np.pi * (key_freq * c) * t) for c in chords) * 0.07
    beat = np.zeros_like(t)
    beats_per_sec = tempo / 60.0
    for k in range(int(duration * beats_per_sec)):
        idx = int(k / beats_per_sec * sr)
        beat[idx : idx + int(0.02 * sr)] += 0.4 * np.hanning(int(0.02 * sr))
    drum = beat
    mix = base + pad + drum
    return np.clip(mix, -1.0, 1.0).astype(np.float32)


def _load_audio_from_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """
    Try to load audio bytes as WAV first.
    If that fails, try to decode with pydub (mp3/m4a/aac) and return as PCM.
    Returns (audio_float, sample_rate)
    """
    # 1) Try straight WAV first (fast path, what you had before)
    try:
        sr, audio = wavfile.read(io.BytesIO(raw))
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # normalize if integer
        if audio.dtype.kind in "iu":
            maxv = np.iinfo(audio.dtype).max
            audio = audio / maxv
        else:
            audio = audio / (np.max(np.abs(audio)) + 1e-9)
        return audio, sr
    except Exception:
        pass

    # 2) Try with soundfile (handles wav/flac/ogg/aiff)
    try:
        data, sr = sf.read(io.BytesIO(raw))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        return data, sr
    except Exception:
        pass

    # 3) Fallback: pydub (mp3, m4a, aac...) → WAV PCM
    try:
        seg = AudioSegment.from_file(io.BytesIO(raw))  # format is auto-detected
        seg = seg.set_channels(1)
        seg = seg.set_frame_rate(44100)
        pcm = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
        return pcm, seg.frame_rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {e}")


@app.post("/generate")
async def generate(request: Request, style: str = Query("afrobeat")):
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="No audio data")

    # --- read/convert to mono float32 + sr ---
    audio, sr = _load_audio_from_bytes(raw)

    # voice detection
    if not detect_voice(audio, sr):
        raise HTTPException(status_code=422, detail="No clear voice detected")

    # feature estimation
    f0 = estimate_key_freq(audio, sr)
    bpm = estimate_tempo(audio, sr)
    s = STYLE_SETTINGS.get(style, {"tempo_mult": 1.0})
    bpm *= s.get("tempo_mult", 1.0)

    dur = max(3.0, min(30.0, len(audio) / sr))
    accomp = synthesize_accompaniment(dur, sr, f0, bpm, style)

    # simple mix under vocal
    min_len = min(len(audio), len(accomp))
    mix = 0.75 * audio[:min_len] + 0.5 * accomp[:min_len]
    mix = np.clip(mix, -1.0, 1.0)

    # write to in-memory buffer
    out_buf = io.BytesIO()
    wavfile.write(out_buf, sr, (mix * 32767).astype(np.int16))
    out_buf.seek(0)

    # IMPORTANT: use StreamingResponse for BytesIO
    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
    )
