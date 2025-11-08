import os
import io
import tempfile
import random
import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from scipy.io import wavfile
import soundfile as sf
import pretty_midi

from pydub import AudioSegment
from pydub.utils import which


# =========================================================
# PATHS & GLOBALS
# =========================================================
BASE_DIR = os.path.dirname(__file__)

STATIC_DIR = os.path.join(BASE_DIR, "static")

SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
os.makedirs(SOUNDFONT_DIR, exist_ok=True)

SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")

# your GitHub release URL (can be overridden in Railway)
SOUNDFONT_URL = os.environ.get(
    "SOUNDFONT_URL",
    "https://github.com/jossy450/vocal-accompaniment-ai/releases/download/soundfont-v1/FluidR3_GM.sf2",
)

# let pydub find ffmpeg
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"

app = FastAPI(title="Vocal Accompaniment Generator", version="0.4")

if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


# =========================================================
# SOUNDFONT HANDLING
# =========================================================
def ensure_soundfont_safe() -> None:
    """Download SF2 if missing, but don't crash if download fails."""
    if os.path.exists(SOUNDFONT_PATH):
        return
    print("[soundfont] Not found, downloading from:", SOUNDFONT_URL)
    try:
        with requests.get(SOUNDFONT_URL, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print("[soundfont] download failed:", r.status_code)
                return
            with open(SOUNDFONT_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("[soundfont] download complete →", SOUNDFONT_PATH)
    except Exception as e:
        print("[soundfont] ERROR:", e)


# call once at startup
ensure_soundfont_safe()


# =========================================================
# STYLE DEFINITIONS
# =========================================================

AFRO_PROGRESSIONS = [
    ["i", "v", "vi", "iv"],   # 1–5–6–4
    ["i", "iv", "v", "iv"],   # 1–4–5–4
    ["ii", "v", "i", "i"],    # 2–5–1
    ["i", "vi", "ii", "v"],   # 1–6–2–5
    ["i", "v", "iv", "v"],    # 1–5–4–5
    ["i", "iv", "vi", "v"],   # 1–4–6–5
]

STYLE_SETTINGS = {
    "afrobeat": {
        "tempo_mult": 1.05,
        "progressions": AFRO_PROGRESSIONS,
        "drum_pattern": "afro-groove",
        "guitar_skank": False,
    },
    "hi_life": {
        "tempo_mult": 1.10,
        "progressions": [["i", "iv", "v", "iv"], ["i", "v", "vi", "iv"]],
        "drum_pattern": "afro-groove",
        "guitar_skank": False,
    },
    "nigerian_gospel": {
        "tempo_mult": 1.03,
        "progressions": [["i", "vi", "iv", "v"], ["i", "vi", "ii", "v"]],
        "drum_pattern": "four-on-floor",
        "guitar_skank": False,
    },
    "reggae": {
        "tempo_mult": 0.90,
        "progressions": [["i", "v", "iv", "v"], ["ii", "v", "i", "i"]],
        "drum_pattern": "one-drop",
        "guitar_skank": True,
    },
    "rnb": {
        "tempo_mult": 0.95,
        "progressions": [["i", "vi", "iv", "v"], ["i", "vi", "ii", "v"]],
        "drum_pattern": "slow-groove",
        "guitar_skank": False,
    },
    "hip_hop": {
        "tempo_mult": 0.90,
        "progressions": [["i", "iv", "v", "iv"]],
        "drum_pattern": "boom-bap",
        "guitar_skank": False,
    },
}


# =========================================================
# DSP / ANALYSIS HELPERS
# =========================================================
def detect_voice(audio: np.ndarray, sr: int) -> bool:
    if audio.size == 0:
        return False
    energy = np.sqrt(np.mean(audio ** 2))
    return energy > 0.01


A4 = 440.0


def estimate_key_freq(audio: np.ndarray, sr: int) -> float:
    if audio.size < sr // 2:
        return A4
    segment = audio[:sr] if audio.size >= sr else audio
    segment = segment - np.mean(segment)
    corr = np.correlate(segment, segment, mode="full")[len(segment) - 1:]
    corr[0:int(sr / 1000)] = 0
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
    env = [np.sum(audio[i * hop:(i + 1) * hop] ** 2) for i in range(frames)]
    env = np.array(env)
    env = (env - env.mean()) / (env.std() + 1e-8)
    corr = np.correlate(env, env, mode="full")[len(env) - 1:]
    corr[:2] = 0
    lag = np.argmax(corr[2:400]) + 2
    bpm = 60.0 * sr / (lag * hop)
    return float(np.clip(bpm, 60.0, 180.0))


# =========================================================
# AUDIO LOADER
# =========================================================
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

    # fallback
    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    audio = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return audio, sr


# =========================================================
# MIDI / HARMONY HELPERS
# =========================================================
def midi_note_from_freq(freq: float) -> int:
    return int(69 + 12 * np.log2(freq / 440.0))


def degree_to_midi(root_midi: int, deg: str) -> int:
    if deg == "i":
        return root_midi
    if deg == "ii":
        return root_midi + 2
    if deg == "iii":
        return root_midi + 4
    if deg == "iv":
        return root_midi + 5
    if deg == "v":
        return root_midi + 7
    if deg == "vi":
        return root_midi + 9
    if deg == "vii":
        return root_midi + 11
    return root_midi


def pick_progression(style: str) -> list[str]:
    style_def = STYLE_SETTINGS.get(style)
    if not style_def:
        return ["i", "v", "vi", "iv"]
    progs = style_def.get("progressions")
    if not progs:
        return ["i", "v", "vi", "iv"]
    return random.choice(progs)


def build_chord_progression(root_midi: int, style: str, bars: int) -> list[int]:
    degrees = pick_progression(style)
    prog = []
    for i in range(bars):
        deg = degrees[i % len(degrees)]
        prog.append(degree_to_midi(root_midi, deg))
    return prog


# =========================================================
# QUANTIZATION
# =========================================================
def quantize_time(t: float, grid: float) -> float:
    """Snap a time value to the nearest grid (in seconds)."""
    return round(t / grid) * grid


# =========================================================
# INSTRUMENT RENDERING
# =========================================================
def render_midi_band(
    sr: int,
    duration: float,
    bpm: float,
    root_freq: float,
    style: str,
    use_piano: bool = True,
    use_bass: bool = True,
    use_drums: bool = True,
    use_guitar: bool = False,
) -> np.ndarray:
    # make sure SF2 exists
    if not os.path.exists(SOUNDFONT_PATH):
        ensure_soundfont_safe()
    if not os.path.exists(SOUNDFONT_PATH):
        raise ValueError(
            f"SoundFont not found at {SOUNDFONT_PATH}. "
            "Upload it or set SOUNDFONT_URL to a reachable location."
        )

    root_midi = midi_note_from_freq(root_freq)
    bar_seconds = 60.0 / bpm * 4
    bars = int(np.ceil(duration / bar_seconds)) + 1
    progression = build_chord_progression(root_midi, style, bars)

    # timing grid
    sixteenth = bar_seconds / 16.0

    pm = pretty_midi.PrettyMIDI()

    # --- Piano ---
    if use_piano:
        piano = pretty_midi.Instrument(program=0)
        t = 0.0
        for chord_root in progression:
            start = quantize_time(t, sixteenth)
            end = quantize_time(t + bar_seconds * 0.95, sixteenth)
            for n in [chord_root, chord_root + 4, chord_root + 7]:
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=85,
                        pitch=n,
                        start=start,
                        end=end,
                    )
                )
            t += bar_seconds
            if t > duration:
                break
        pm.instruments.append(piano)

    # --- Bass ---
    if use_bass:
        bass = pretty_midi.Instrument(program=32)
        t = 0.0
        for chord_root in progression:
            start = quantize_time(t, sixteenth)
            end = quantize_time(t + bar_seconds * 0.9, sixteenth)
            bass.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=chord_root - 12,
                    start=start,
                    end=end,
                )
            )
            t += bar_seconds
            if t > duration:
                break
        pm.instruments.append(bass)

    # --- Drums ---
    if use_drums:
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        beat = 60.0 / bpm
        half_beat = beat / 2.0
        t = 0.0
        pattern = STYLE_SETTINGS.get(style, {}).get("drum_pattern", "kick-snare")

        while t < duration:
            if pattern == "afro-groove":
                # kick on 1
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=110,
                        pitch=36,
                        start=quantize_time(t, sixteenth),
                        end=quantize_time(t + 0.1, sixteenth),
                    )
                )
                # perc on & of 1
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=70,
                        pitch=60,
                        start=quantize_time(t + half_beat, sixteenth),
                        end=quantize_time(t + half_beat + 0.08, sixteenth),
                    )
                )
                # snare on 2
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=105,
                        pitch=38,
                        start=quantize_time(t + beat, sixteenth),
                        end=quantize_time(t + beat + 0.1, sixteenth),
                    )
                )
                # perc on & of 2
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=65,
                        pitch=60,
                        start=quantize_time(t + beat + half_beat, sixteenth),
                        end=quantize_time(t + beat + half_beat + 0.08, sixteenth),
                    )
                )
                # light kick on 3
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=90,
                        pitch=36,
                        start=quantize_time(t + 2 * beat, sixteenth),
                        end=quantize_time(t + 2 * beat + 0.08, sixteenth),
                    )
                )
                # clap on 4
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=110,
                        pitch=39,
                        start=quantize_time(t + 3 * beat, sixteenth),
                        end=quantize_time(t + 3 * beat + 0.1, sixteenth),
                    )
                )
                t += 4 * beat

            elif pattern == "one-drop":
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=95,
                        pitch=36,
                        start=quantize_time(t + beat, sixteenth),
                        end=quantize_time(t + beat + 0.1, sixteenth),
                    )
                )
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=105,
                        pitch=38,
                        start=quantize_time(t + beat, sixteenth),
                        end=quantize_time(t + beat + 0.1, sixteenth),
                    )
                )
                t += 2 * beat

            elif pattern == "four-on-floor":
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=36,
                        start=quantize_time(t, sixteenth),
                        end=quantize_time(t + 0.1, sixteenth),
                    )
                )
                t += beat

            else:
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=110,
                        pitch=36,
                        start=quantize_time(t, sixteenth),
                        end=quantize_time(t + 0.1, sixteenth),
                    )
                )
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=110,
                        pitch=38,
                        start=quantize_time(t + beat, sixteenth),
                        end=quantize_time(t + beat + 0.1, sixteenth),
                    )
                )
                t += 2 * beat

        pm.instruments.append(drum)

    # --- Guitar skank ---
    wants_skank = STYLE_SETTINGS.get(style, {}).get("guitar_skank", False)
    if use_guitar or wants_skank:
        guitar = pretty_midi.Instrument(program=25)
        tbar = 0.0
        while tbar < duration:
            for beat_i in range(4):
                beat_start = tbar + (beat_i * (bar_seconds / 4.0))
                offbeat = beat_start + (bar_seconds / 8.0)
                q_offbeat = quantize_time(offbeat, sixteenth)
                if q_offbeat < duration:
                    guitar.notes.append(
                        pretty_midi.Note(
                            velocity=80,
                            pitch=root_midi + 7,
                            start=q_offbeat,
                            end=q_offbeat + 0.12,
                        )
                    )
            tbar += bar_seconds
        pm.instruments.append(guitar)

    # --- Render to audio via FluidSynth (version-safe) ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        tmp_path = tmpwav.name

    try:
        # newer pretty_midi → returns numpy array
        audio_data = pm.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH)
        sf.write(tmp_path, audio_data, sr)
    except TypeError:
        # older pretty_midi → accepts filename=
        pm.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH, filename=tmp_path)

    accomp, _ = sf.read(tmp_path)
    return accomp

    def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)) + 1e-9)

def simple_highpass(accomp: np.ndarray, sr: int, cutoff: float = 150.0) -> np.ndarray:
    rc = np.exp(-2 * np.pi * cutoff / sr)
    y = np.zeros_like(accomp)
    prev_x = 0.0
    prev_y = 0.0
    for i, x in enumerate(accomp):
        y[i] = x - prev_x + rc * prev_y
        prev_x = x
        prev_y = y[i]
    return y


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
async def root_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "UI not found, but API is running."}


@app.post("/generate")
async def generate(
    request: Request,
    style: str = Query("afrobeat"),
    piano: bool = Query(True),
    bass: bool = Query(True),
    drums: bool = Query(True),
    guitar: bool = Query(False),
):
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio data received")

    # 1) load user audio (vocal)
    vocal, sr = load_audio_from_bytes(raw)

    # 2) analyse vocal
    f0 = estimate_key_freq(vocal, sr)
    bpm = estimate_tempo(vocal, sr)
    style_def = STYLE_SETTINGS.get(style, {})
    bpm *= style_def.get("tempo_mult", 1.0)

    duration = len(vocal) / sr

    # 3) render band (MIDI → audio)
    accomp = render_midi_band(
        sr,
        duration,
        bpm,
        f0,
        style,
        use_piano=piano,
        use_bass=bass,
        use_drums=drums,
        use_guitar=guitar,
    )

    # 4) make sure same length
    min_len = min(len(vocal), len(accomp))
    vocal = vocal[:min_len]
    band = accomp[:min_len]

    # ====== MIX IMPROVEMENT ======
    # helpers must exist above: rms(), simple_highpass()
    # high-pass band to avoid mud
    band = simple_highpass(band, sr, cutoff=140.0)

    # loudness match
    v_r = rms(vocal)
    b_r = rms(band)
    if b_r > 0:
        band = band * (v_r / (b_r * 1.5))

    # simple sidechain ducking
    env = np.abs(vocal)
    win = 256
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")
    duck_amount = 0.45
    band_gain = 1.0 - duck_amount * np.clip(env_smooth * 4.0, 0.0, 1.0)
    band = band * band_gain

    # combine with headroom
    mix = vocal * 0.98 + band * 0.9

    # final safety limiter
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix = mix / peak

    # 5) return WAV
    out_buf = io.BytesIO()
    wavfile.write(out_buf, sr, (mix * 32767).astype(np.int16))
    out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
    )
