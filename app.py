import os
import io
import tempfile
import random
import numpy as np
import requests
import json
import base64

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from scipy.io import wavfile
import soundfile as sf
import pretty_midi

from pydub import AudioSegment
from pydub.utils import which

# =========================================================
# REMOTE MUSIC MODEL
# =========================================================
def call_remote_music_model(vocal_bytes: bytes, style: str) -> bytes | None:
    """
    Calls a hosted AI music model (e.g. Replicate) and returns audio bytes.
    Adjust payload/keys to match the model you pick.
    """
    remote_url = os.environ.get("REMOTE_MUSIC_URL")
    remote_token = os.environ.get("REMOTE_MUSIC_TOKEN")
    if not remote_url or not remote_token:
        return None  # not configured

    # many hosted models want base64 audio
    vocal_b64 = base64.b64encode(vocal_bytes).decode("utf-8")

    # this is an EXAMPLE payload for a musicgen-like model
    payload = {
        "version": "musicgen-or-your-model-version",
        "input": {
            "audio": vocal_b64,
            "prompt": f"backing track, {style}, no lead vocals, high quality, mixed",
            "duration": 30,  # seconds, or omit if model infers
        },
    }

    try:
        resp = requests.post(
            remote_url,
            headers={
                "Authorization": f"Token {remote_token}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=180,
        )
        if resp.status_code not in (200, 201):
            print("[remote-music] bad status:", resp.status_code, resp.text)
            return None

        data = resp.json()

        # some providers return an audio URL
        audio_url = (
            data.get("output")
            if isinstance(data.get("output"), str)
            else None
        )
        if audio_url:
            audio_resp = requests.get(audio_url, timeout=180)
            if audio_resp.status_code == 200:
                return audio_resp.content

        # or base64 right away
        if "audio_b64" in data:
            return base64.b64decode(data["audio_b64"])

    except Exception as e:
        print("[remote-music] ERROR:", e)

    return None



# =========================================================
# PATHS & GLOBALS
# =========================================================
BASE_DIR = os.path.dirname(__file__)

STATIC_DIR = os.path.join(BASE_DIR, "static")

SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
os.makedirs(SOUNDFONT_DIR, exist_ok=True)
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")

# your GitHub release URL (overridable in Railway)
SOUNDFONT_URL = os.environ.get(
    "SOUNDFONT_URL",
    "https://github.com/jossy450/vocal-accompaniment-ai/releases/download/soundfont-v1/FluidR3_GM.sf2",
)

# let pydub find ffmpeg
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"

app = FastAPI(title="Vocal Accompaniment Generator", version="0.5")

if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


# =========================================================
# SOUNDFONT HANDLING
# =========================================================
def ensure_soundfont_safe() -> None:
    """Download SF2 if missing, but don't crash if it fails."""
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


# do it once on startup
ensure_soundfont_safe()


# =========================================================
# STYLE DEFINITIONS
# =========================================================
AFRO_PROGRESSIONS = [
    ["i", "v", "vi", "iv"],
    ["i", "iv", "v", "iv"],
    ["ii", "v", "i", "i"],
    ["i", "vi", "ii", "v"],
    ["i", "v", "iv", "v"],
    ["i", "iv", "vi", "v"],
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
# ANALYSIS / DSP HELPERS
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

    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    audio = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return audio, sr


# =========================================================
# MIDI / HARMONY
# =========================================================
def midi_note_from_freq(freq: float) -> int:
    return int(69 + 12 * np.log2(freq / 440.0))


def degree_to_midi(root_midi: int, deg: str) -> int:
    mapping = {
        "i": 0,
        "ii": 2,
        "iii": 4,
        "iv": 5,
        "v": 7,
        "vi": 9,
        "vii": 11,
    }
    return root_midi + mapping.get(deg, 0)


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
    out = []
    for i in range(bars):
        deg = degrees[i % len(degrees)]
        out.append(degree_to_midi(root_midi, deg))
    return out


# =========================================================
# QUANTIZATION + SIMPLE MIX HELPERS
# =========================================================
def quantize_time(t: float, grid: float) -> float:
    return round(t / grid) * grid


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def simple_highpass(accomp: np.ndarray, sr: int, cutoff: float = 150.0) -> np.ndarray:
    """very cheap HPF to reduce mud from band."""
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
# REMOTE GENERATOR CLIENT
# =========================================================

def call_remote_music_model(vocal_bytes: bytes, style: str) -> bytes | None:
    """
    Call an external AI music/accompaniment model.
    Returns raw audio bytes (wav/mp3) or None on failure.

    This version is written like a Replicate / generic HTTP model.
    Adjust the URL/payload to match the provider you choose.
    """
    remote_url = os.environ.get("REMOTE_MUSIC_URL")
    remote_token = os.environ.get("REMOTE_MUSIC_TOKEN")

    if not remote_url or not remote_token:
        return None  # not configured

    # some APIs like base64 input
    vocal_b64 = base64.b64encode(vocal_bytes).decode("utf-8")

    payload = {
        "input_vocal": vocal_b64,
        "style": style,
        "tempo": None,
    }

    try:
        resp = requests.post(
            remote_url,
            headers={"Authorization": f"Bearer {remote_token}"},
            json=payload,
            timeout=180,
        )
        if resp.status_code != 200:
            print("[remote-model] bad status:", resp.status_code, resp.text)
            return None

        data = resp.json()
        # assume API returns base64 audio
        if "audio_b64" in data:
            return base64.b64decode(data["audio_b64"])

        # or a direct URL
        if "audio_url" in data:
            audio_resp = requests.get(data["audio_url"], timeout=180)
            if audio_resp.status_code == 200:
                return audio_resp.content

    except Exception as e:
        print("[remote-model] ERROR:", e)

    return None


# ==========================================================
# MASTERING API
# ==========================================================
def call_mastering_api(audio_bytes: bytes) -> bytes | None:
    """
    Sends the audio to a mastering/processing API (Auphonic / Dolby).
    Returns mastered audio bytes, or None if it fails.
    """
    master_url = os.environ.get("MASTERING_URL")
    master_token = os.environ.get("MASTERING_TOKEN")
    master_enabled = os.environ.get("MASTERING_ENABLED", "false").lower() == "true"

    if not (master_url and master_token and master_enabled):
        return None  # not configured

    try:
        files = {"audio": ("mix.wav", audio_bytes, "audio/wav")}
        headers = {"Authorization": f"Bearer {master_token}"}
        resp = requests.post(master_url, headers=headers, files=files, timeout=180)
        if resp.status_code != 200:
            print("[mastering] bad status:", resp.status_code, resp.text)
            return None
        # assuming API returns raw audio — some return JSON with URL instead
        return resp.content
    except Exception as e:
        print("[mastering] ERROR:", e)
        return None


# =========================================================
# RENDER MIDI BAND
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
    if not os.path.exists(SOUNDFONT_PATH):
        ensure_soundfont_safe()
    if not os.path.exists(SOUNDFONT_PATH):
        raise ValueError(
            f"SoundFont not found at {SOUNDFONT_PATH}. "
            "Upload it or set SOUNDFONT_URL."
        )

    root_midi = midi_note_from_freq(root_freq)
    bar_seconds = 60.0 / bpm * 4
    bars = int(np.ceil(duration / bar_seconds)) + 1
    progression = build_chord_progression(root_midi, style, bars)

    sixteenth = bar_seconds / 16.0

    pm = pretty_midi.PrettyMIDI()

    # piano
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

    # bass
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

    # drums
    if use_drums:
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        beat = 60.0 / bpm
        half = beat / 2.0
        t = 0.0
        pattern = STYLE_SETTINGS.get(style, {}).get("drum_pattern", "kick-snare")
        while t < duration:
            if pattern == "afro-groove":
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
                        velocity=70,
                        pitch=60,
                        start=quantize_time(t + half, sixteenth),
                        end=quantize_time(t + half + 0.08, sixteenth),
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
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=65,
                        pitch=60,
                        start=quantize_time(t + beat + half, sixteenth),
                        end=quantize_time(t + beat + half + 0.08, sixteenth),
                    )
                )
                drum.notes.append(
                    pretty_midi.Note(
                        velocity=90,
                        pitch=36,
                        start=quantize_time(t + 2 * beat, sixteenth),
                        end=quantize_time(t + 2 * beat + 0.08, sixteenth),
                    )
                )
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

            else:  # fallback
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

    # guitar skank
    wants_skank = STYLE_SETTINGS.get(style, {}).get("guitar_skank", False)
    if use_guitar or wants_skank:
        guitar = pretty_midi.Instrument(program=25)
        tbar = 0.0
        while tbar < duration:
            for beat_i in range(4):
                beat_start = tbar + (beat_i * (bar_seconds / 4.0))
                offbeat = beat_start + (bar_seconds / 8.0)
                q_off = quantize_time(offbeat, sixteenth)
                if q_off < duration:
                    guitar.notes.append(
                        pretty_midi.Note(
                            velocity=80,
                            pitch=root_midi + 7,
                            start=q_off,
                            end=q_off + 0.12,
                        )
                    )
            tbar += bar_seconds
        pm.instruments.append(guitar)

    # render via fluidsynth
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        tmp_path = tmpwav.name

    try:
        audio_data = pm.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH)
        sf.write(tmp_path, audio_data, sr)
    except TypeError:
        pm.fluidsynth(fs=sr, sf2_path=SOUNDFONT_PATH, filename=tmp_path)

    accomp, _ = sf.read(tmp_path)
    return accomp


# =========================================================
# ROUTES
# =========================================================
@app.post("/generate")
async def generate(
    request: Request,
    style: str = Query("afrobeat"),
    piano: bool = Query(True),
    bass: bool = Query(True),
    drums: bool = Query(True),
    guitar: bool = Query(False),
):
    # raw upload (keep this for remote call)
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio data received")

    # =====================================================
    # 1) TRY SOPHISTICATED REMOTE MODEL FIRST
    # =====================================================
    remote_audio = call_remote_music_model(raw, style)
    if remote_audio is not None:
        # we have a band from the model — now mix it with the original vocal
        # load vocal
        vocal, sr = load_audio_from_bytes(raw)
        # load remote band
        band, sr_band = load_audio_from_bytes(remote_audio)
        # resample / align if SR differs
        if sr_band != sr:
            # quick-and-dirty: trim to min len
            min_len = min(len(vocal), len(band))
            vocal = vocal[:min_len]
            band = band[:min_len]
        else:
            min_len = min(len(vocal), len(band))
            vocal = vocal[:min_len]
            band = band[:min_len]

        # light ducking so vocal sits on top
        band = simple_highpass(band, sr, cutoff=140.0)
        v_r = rms(vocal)
        b_r = rms(band)
        if b_r > 0:
            band = band * (v_r / (b_r * 1.3))
        # no complicated sidechain — remote audio already mixed
        mix = vocal * 0.98 + band * 0.7
        peak = np.max(np.abs(mix)) + 1e-9
        if peak > 1.0:
            mix = mix / peak

        # to bytes
        buf = io.BytesIO()
        wavfile.write(buf, sr, (mix * 32767).astype(np.int16))
        buf.seek(0)
        mastered = call_mastering_api(buf.getvalue())
        final_bytes = mastered if mastered is not None else buf.getvalue()

        return StreamingResponse(
            io.BytesIO(final_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
        )

    # =====================================================
    # 2) FALLBACK: YOUR CURRENT LOCAL MIDI ENGINE
    # =====================================================
    vocal, sr = load_audio_from_bytes(raw)

    f0 = estimate_key_freq(vocal, sr)
    bpm = estimate_tempo(vocal, sr)
    style_def = STYLE_SETTINGS.get(style, {})
    bpm *= style_def.get("tempo_mult", 1.0)
    duration = len(vocal) / sr

    band = render_midi_band(
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

    min_len = min(len(vocal), len(band))
    vocal = vocal[:min_len]
    band = band[:min_len]

    band = simple_highpass(band, sr, cutoff=140.0)
    v_r = rms(vocal)
    b_r = rms(band)
    if b_r > 0:
        band = band * (v_r / (b_r * 1.5))

    env = np.abs(vocal)
    win = 256
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")
    duck_amount = 0.45
    band_gain = 1.0 - duck_amount * np.clip(env_smooth * 4.0, 0.0, 1.0)
    band = band * band_gain

    mix = vocal * 0.98 + band * 0.9
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix = mix / peak

    out_buf = io.BytesIO()
    wavfile.write(out_buf, sr, (mix * 32767).astype(np.int16))
    out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
    )
