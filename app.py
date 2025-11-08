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
# APP + PATHS
# =========================================================
app = FastAPI(title="Vocal Accompaniment Generator", version="0.3")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")
SOUNDFONT_URL = "https://archive.org/download/fluidr3gm/FluidR3_GM.sf2"

# Make pydub find ffmpeg (installed in Dockerfile)
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"


# =========================================================
# STYLE DEFINITIONS
# =========================================================

# Expanded West African progressions
AFRO_PROGRESSIONS = [
    ["i", "v", "vi", "iv"],   # 1–5–6–4  (common afrobeats / gospel)
    ["i", "iv", "v", "iv"],   # 1–4–5–4  (classic highlife)
    ["ii", "v", "i", "i"],    # 2–5–1    (jazzy gospel)
    ["i", "vi", "ii", "v"],   # 1–6–2–5  (gospel / soulful)
    ["i", "v", "iv", "v"],    # 1–5–4–5  (afro-fusion)
    ["i", "iv", "vi", "v"],   # 1–4–6–5  (pop feel)
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
        "guitar_skank": True,  # off-beat guitar default
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
# INIT: ensure soundfont once
# =========================================================
# ---------- SOUNDFONT AUTO / SAFE ----------
SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
os.makedirs(SOUNDFONT_DIR, exist_ok=True)

# allow override from env if you host it yourself
SOUNDFONT_URL = os.environ.get(
    "SOUNDFONT_URL",
    "https://archive.org/download/fluidr3gm/FluidR3_GM.sf2",
)
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")

def ensure_soundfont_safe() -> None:
    """Try to download the SoundFont, but never crash the app if it fails."""
    if os.path.exists(SOUNDFONT_PATH):
        return
    print("[soundfont] Not found locally, attempting download...")
    try:
        with requests.get(SOUNDFONT_URL, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print(f"[soundfont] Download failed, status={r.status_code}")
                return
            with open(SOUNDFONT_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("[soundfont] Downloaded to", SOUNDFONT_PATH)
    except Exception as e:
        # don't kill the app on startup
        print(f"[soundfont] ERROR downloading soundfont: {e}")
        print("[soundfont] You can set SOUNDFONT_URL to your own hosted file.")
        # just return — we'll check again at request time


# call once at startup, but non-fatal
ensure_soundfont_safe()

# =========================================================
# DSP HELPERS
# =========================================================
def detect_voice(audio: np.ndarray, sr: int) -> bool:
    if audio.size == 0:
        return False
    energy = np.sqrt(np.mean(audio ** 2))
    return energy > 0.01

A4 = 440.0


def estimate_key_freq(audio: np.ndarray, sr: int) -> float:
    """Very simple autocorrelation-based f0 estimate."""
    if audio.size < sr // 2:
        return A4
    segment = audio[:sr] if audio.size >= sr else audio
    segment = segment - np.mean(segment)
    corr = np.correlate(segment, segment, mode="full")[len(segment) - 1 :]
    # ignore very small lags
    corr[0 : int(sr / 1000)] = 0
    peak = np.argmax(corr)
    if peak <= 0:
        return A4
    f0 = sr / peak
    return float(np.clip(f0, 80.0, 800.0))


def estimate_tempo(audio: np.ndarray, sr: int) -> float:
    """Crude energy-based tempo estimate."""
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

# =========================================================
# AUDIO LOADER (wav → float OR fallback to pydub)
# =========================================================
def load_audio_from_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    # WAV fast path
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

    # fallback: mp3/m4a…
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
    """Map roman-ish degree to MIDI pitch in a simple major-ish scale."""
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
    progressions = style_def.get("progressions")
    if not progressions:
        return ["i", "v", "vi", "iv"]
    return random.choice(progressions)

def build_chord_progression(root_midi: int, style: str, bars: int) -> list[int]:
    degrees = pick_progression(style)
    prog = []
    for i in range(bars):
        deg = degrees[i % len(degrees)]
        prog.append(degree_to_midi(root_midi, deg))
    return prog

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
    root_midi = midi_note_from_freq(root_freq)
    bar_seconds = 60.0 / bpm * 4
    bars = int(np.ceil(duration / bar_seconds)) + 1
    progression = build_chord_progression(root_midi, style, bars)

    pm = pretty_midi.PrettyMIDI()

    # --- Piano block chords ---
    if use_piano:
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand
        t = 0.0
        for chord_root in progression:
            # simple triad
            for n in [chord_root, chord_root + 4, chord_root + 7]:
                note = pretty_midi.Note(
                    velocity=85,
                    pitch=n,
                    start=t,
                    end=t + bar_seconds * 0.95,
                )
                piano.notes.append(note)
            t += bar_seconds
            if t > duration:
                break
        pm.instruments.append(piano)

    # --- Bass on roots ---
    if use_bass:
        bass = pretty_midi.Instrument(program=32)  # Acoustic Bass
        t = 0.0
        for chord_root in progression:
            note = pretty_midi.Note(
                velocity=100,
                pitch=chord_root - 12,
                start=t,
                end=t + bar_seconds * 0.9,
            )
            bass.notes.append(note)
            t += bar_seconds
            if t > duration:
                break
        pm.instruments.append(bass)

    # --- Drums ---
    if use_drums:
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        beat = 60.0 / bpm           # 1 beat (quarter note)
        half_beat = beat / 2.0      # 8th
        t = 0.0
        pattern = STYLE_SETTINGS.get(style, {}).get("drum_pattern", "kick-snare")

        while t < duration:
            if pattern == "afro-groove":
                # Kick on 1
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=36, start=t, end=t + 0.1)
                )
                # light conga / percussion on the & of 1
                drum.notes.append(
                    pretty_midi.Note(velocity=70, pitch=60, start=t + half_beat, end=t + half_beat + 0.08)
                )
                # Snare on 2
                drum.notes.append(
                    pretty_midi.Note(velocity=105, pitch=38, start=t + beat, end=t + beat + 0.1)
                )
                # Percussion on the & of 2
                drum.notes.append(
                    pretty_midi.Note(velocity=65, pitch=60, start=t + beat + half_beat, end=t + beat + half_beat + 0.08)
                )
                # Kick again on 3 (lighter)
                drum.notes.append(
                    pretty_midi.Note(velocity=90, pitch=36, start=t + 2*beat, end=t + 2*beat + 0.08)
                )
                # Snare / clap on 4
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=39, start=t + 3*beat, end=t + 3*beat + 0.1)
                )

                # advance by a bar (4 beats)
                t += 4 * beat

            elif pattern == "one-drop":
                # reggae-ish
                drum.notes.append(
                    pretty_midi.Note(velocity=95, pitch=36, start=t + beat, end=t + beat + 0.1)
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=105, pitch=38, start=t + beat, end=t + beat + 0.1)
                )
                t += 2 * beat

            elif pattern == "four-on-floor":
                # kick every beat
                drum.notes.append(
                    pretty_midi.Note(velocity=100, pitch=36, start=t, end=t + 0.1)
                )
                t += beat

            else:
                # fallback: simple kick-snare
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=36, start=t, end=t + 0.1)
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=38, start=t + beat, end=t + beat + 0.1)
                )
                t += 2 * beat

        pm.instruments.append(drum)

    # --- Reggae guitar skank (offbeat stabs) ---
    style_wants_skank = STYLE_SETTINGS.get(style, {}).get("guitar_skank", False)
    if use_guitar or style_wants_skank:
        guitar = pretty_midi.Instrument(program=25)  # Acoustic Guitar (nylon-ish)
        tbar = 0.0
        while tbar < duration:
            for beat_i in range(4):  # 4 beats per bar
                beat_start = tbar + (beat_i * (bar_seconds / 4.0))
                offbeat = beat_start + (bar_seconds / 8.0)
                if offbeat < duration:
                    guitar.notes.append(
                        pretty_midi.Note(
                            velocity=80,
                            pitch=root_midi + 7,  # 5th above root
                            start=offbeat,
                            end=offbeat + 0.12,
                        )
                    )
            tbar += bar_seconds
        pm.instruments.append(guitar)

       # --- Render to WAV via fluidsynth (version-safe) ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        tmp_path = tmpwav.name

    try:
        # Newer pretty_midi: returns np.array
        audio_data = pm.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH)
        # write it ourselves
        sf.write(tmp_path, audio_data, sr)
    except TypeError:
        # Older pretty_midi that accepted filename=
        pm.fluidsynth(
            fs=sr,
            sf2_path=SOUNDFONT_PATH,
            filename=tmp_path,
        )

    # now load the rendered accompaniment
    accomp, _ = sf.read(tmp_path)
    return accomp


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

    # load & normalise
    audio, sr = load_audio_from_bytes(raw)

    # detect vocal presence (warn but don't fail)
    if not detect_voice(audio, sr):
        print("Warning: low vocal energy detected. Proceeding anyway.")

    # analyse
    f0 = estimate_key_freq(audio, sr)
    bpm = estimate_tempo(audio, sr)
    style_def = STYLE_SETTINGS.get(style, {})
    bpm *= style_def.get("tempo_mult", 1.0)

    duration = len(audio) / sr

    # render full band
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

    # mix under vocal
    min_len = min(len(audio), len(accomp))
    mix = 0.9 * audio[:min_len] + 0.9 * accomp[:min_len]
    mix = np.clip(mix, -1.0, 1.0)

    # return as WAV
    out_buf = io.BytesIO()
    wavfile.write(out_buf, sr, (mix * 32767).astype(np.int16))
    out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
    )
