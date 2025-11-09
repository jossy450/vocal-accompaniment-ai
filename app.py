import os
import io
import tempfile
import random
import json
import base64

import numpy as np
import requests
import librosa

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from scipy.io import wavfile
import soundfile as sf
import pretty_midi

from pydub import AudioSegment
from pydub.utils import which


# =========================================================
# PATHS & APP
# =========================================================
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")

SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
os.makedirs(SOUNDFONT_DIR, exist_ok=True)
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")

# GitHub-hosted soundfont (can be overridden in Railway)
SOUNDFONT_URL = os.environ.get(
    "SOUNDFONT_URL",
    "https://github.com/jossy450/vocal-accompaniment-ai/releases/download/soundfont-v1/FluidR3_GM.sf2",
)

# Make pydub see ffmpeg (Docker installs it)
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"

app = FastAPI(title="Vocal Accompaniment Generator", version="0.6")

if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


# =========================================================
# SOUND FONT DOWNLOAD (non-fatal)
# =========================================================
def ensure_soundfont_safe() -> None:
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
# DSP / ANALYSIS
# =========================================================
A4 = 440.0


def detect_voice(audio: np.ndarray, sr: int) -> bool:
    if audio.size == 0:
        return False
    energy = np.sqrt(np.mean(audio ** 2))
    return energy > 0.01


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


def rough_key_from_freq(freq: float) -> str | None:
    if not freq:
        return None
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return scale[midi % 12]


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


# =========================================================
# LOAD AUDIO (wav/m4a/mp3)
# =========================================================
def load_audio_from_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    # fast wav path
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

    # fallback: pydub
    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    audio = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return audio, sr


# =========================================================
# SMALL UTILITIES (mix, quantize)
# =========================================================
def quantize_time(t: float, grid: float) -> float:
    return round(t / grid) * grid


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


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


def align_to_vocal(
    vocal: np.ndarray, band: np.ndarray, sr: int, target_bpm: float, band_bpm: float | None = None
) -> np.ndarray:
    """time-stretch band to match vocal tempo"""
    if not band_bpm or band_bpm <= 0:
        return band
    rate = target_bpm / band_bpm
    band_float = band.astype(np.float32)
    stretched = librosa.effects.time_stretch(band_float, rate)
    min_len = min(len(vocal), len(stretched))
    return stretched[:min_len]


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
# MASTERING (optional)
# =========================================================
def call_mastering_api(audio_bytes: bytes) -> bytes | None:
    """
    Master the audio using Auphonic.

    Expects these env vars:
      MASTERING_ENABLED=true
      MASTERING_URL=https://auphonic.com/api/simple/productions.json
      MASTERING_TOKEN= sOmuFXlhpryFuxzh7AQsWRN4c3JKtbMP

    This uses Auphonic's "simple" endpoint, which returns the processed file
    directly when 'output_files' is set to 'wav'.
    """
    master_enabled = os.environ.get("MASTERING_ENABLED", "false").lower() == "true"
    master_url = os.environ.get("MASTERING_URL")
    master_token = os.environ.get("MASTERING_TOKEN")

    if not master_enabled or not master_url or not master_token:
        return None  # mastering not configured

    try:
        # Auphonic simple API accepts multipart:
        #   audio_file     -> the file to process
        #   output_files[] -> format (e.g. wav, mp3)
        #   token          -> your personal API token
        files = {
            "audio_file": ("mix.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "token": master_token,
            # request WAV back
            "output_files[]": "wav",
        }

        resp = requests.post(master_url, data=data, files=files, timeout=180)
        if resp.status_code != 200:
            print("[mastering] Auphonic error:", resp.status_code, resp.text[:300])
            return None

        # Auphonic simple API actually returns the processed file in the body
        # if you asked for a single file. So we just return resp.content
        return resp.content

    except Exception as e:
        print("[mastering] ERROR calling Auphonic:", e)
        return None


# =========================================================
# MIDI RENDERER (fallback)
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
        raise ValueError(f"SoundFont not found at {SOUNDFONT_PATH}")

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

    # render with fluidsynth
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
# REPLICATE: meta/musicgen (melody)
# =========================================================
REPLICATE_MODEL_VERSION = os.environ.get(
    "REPLICATE_MODEL_VERSION",
    "2b5dc5f29cee83fd5cdf8f9c92e555aae7ca2a69b73c5182f3065362b2fa0a45",
)

def build_gospel_prompt(bpm: float, key: str | None) -> str:
    base = (
        "slow Nigerian gospel / worship backing track, warm piano, soft drums, "
        "subtle bass, ambient pads, no lead vocals, mixed and spatial, "
        "tight timing, congregational feel"
    )
    parts = [base]
    if bpm:
        parts.append(f"{int(bpm)} BPM")
    if key:
        parts.append(f"in key of {key}")
    return ", ".join(parts)


def call_replicate_musicgen_gospel(vocal_bytes: bytes, prompt: str, duration: int = 30) -> bytes | None:
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("[replicate] REPLICATE_API_TOKEN not set")
        return None

    pred_url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }

    try:
        # 1) upload the vocal file
        file_resp = requests.post(
            "https://api.replicate.com/v1/files",
            headers={"Authorization": f"Token {api_token}"},
            files={"file": ("vocal.wav", vocal_bytes, "audio/wav")},
            timeout=60,
        )
        if file_resp.status_code != 200:
            print("[replicate] file upload failed:", file_resp.status_code, file_resp.text)
            return None

        file_data = file_resp.json()
        file_url = file_data["urls"]["get"]

        payload = {
            "version": REPLICATE_MODEL_VERSION,
            "input": {
                "model_version": "stereo-melody-large",
                "prompt": prompt,
                "input_audio": file_url,
                "duration": duration,
                "continuation": False,
                "output_format": "wav",
            },
        }

        # 2) create prediction
        pred_resp = requests.post(pred_url, headers=headers, data=json.dumps(payload), timeout=60)
        if pred_resp.status_code not in (200, 201):
            print("[replicate] prediction create failed:", pred_resp.status_code, pred_resp.text)
            return None

        prediction = pred_resp.json()
        pred_id = prediction["id"]

        # 3) poll
        status = prediction["status"]
        get_url = f"{pred_url}/{pred_id}"
        while status not in ("succeeded", "failed", "canceled"):
            poll = requests.get(get_url, headers=headers, timeout=60)
            prediction = poll.json()
            status = prediction["status"]

        if status != "succeeded":
            print("[replicate] prediction failed:", prediction)
            return None

        output_urls = prediction.get("output")
        if not output_urls:
            return None
        if isinstance(output_urls, list):
            audio_url = output_urls[0]
        else:
            audio_url = output_urls

        audio_resp = requests.get(audio_url, timeout=120)
        if audio_resp.status_code == 200:
            return audio_resp.content

    except Exception as e:
        print("[replicate] ERROR:", e)

    return None


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
async def root_ui():
    idx = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"detail": "Vocal Accompaniment API running"}


@app.post("/generate")
async def generate(
    request: Request,
    style: str = Query("nigerian_gospel"),
    piano: bool = Query(True),
    bass: bool = Query(True),
    drums: bool = Query(True),
    guitar: bool = Query(False),
):
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio data received")

    # ---- load vocal and analyze ----
    vocal, sr = load_audio_from_bytes(raw)
    f0 = estimate_key_freq(vocal, sr)
    bpm = estimate_tempo(vocal, sr)
    key_name = rough_key_from_freq(f0)

    # ---- try remote Replicate first ----
    gospel_prompt = build_gospel_prompt(bpm, key_name)
    remote_band_bytes = call_replicate_musicgen_gospel(
        vocal_bytes=raw,
        prompt=gospel_prompt,
        duration=int(len(vocal) / sr) if len(vocal) / sr < 30 else 30,
    )

    if remote_band_bytes is not None:
        band, band_sr = load_audio_from_bytes(remote_band_bytes)

        # align / resample
        if band_sr != sr:
            band = librosa.resample(band, orig_sr=band_sr, target_sr=sr)
            band_sr = sr

        aligned_band = align_to_vocal(vocal, band, sr, target_bpm=bpm, band_bpm=bpm)

        # gospel mix: vocal on top
        aligned_band = simple_highpass(aligned_band, sr, cutoff=130.0)
        v_r = rms(vocal)
        b_r = rms(aligned_band)
        if b_r > 0:
            aligned_band = aligned_band * (v_r / (b_r * 1.4))

        mix = vocal * 1.0 + aligned_band * 0.7
        peak = np.max(np.abs(mix)) + 1e-9
        if peak > 1.0:
            mix = mix / peak

        # to wav
        buf = io.BytesIO()
        wavfile.write(buf, sr, (mix * 32767).astype(np.int16))
        buf.seek(0)

        mastered = call_mastering_api(buf.getvalue())
        final_bytes = mastered if mastered is not None else buf.getvalue()

        return StreamingResponse(
            io.BytesIO(final_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="gospel_accompaniment.wav"'},
        )

    # ---- FALLBACK: local MIDI band ----
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

    band = simple_highpass(band, sr, cutoff=130.0)
    v_r = rms(vocal)
    b_r = rms(band)
    if b_r > 0:
        band = band * (v_r / (b_r * 1.5))

    mix = vocal * 1.0 + band * 0.85
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix = mix / peak

    out_buf = io.BytesIO()
    wavfile.write(out_buf, sr, (mix * 32767).astype(np.int16))
    out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="gospel_accompaniment.wav"'},
    )
