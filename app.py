import os
import io
import tempfile
import random
import base64
import json

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
# BASIC PATHS
# =========================================================
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")

SOUNDFONT_DIR = os.path.join(BASE_DIR, "soundfonts")
os.makedirs(SOUNDFONT_DIR, exist_ok=True)
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2")

# you uploaded this to GitHub releases
DEFAULT_SF_URL = (
    "https://github.com/jossy450/vocal-accompaniment-ai/"
    "releases/download/soundfont-v1/FluidR3_GM.sf2"
)
SOUNDFONT_URL = os.environ.get("SOUNDFONT_URL", DEFAULT_SF_URL)

# let pydub find ffmpeg (Dockerfile installs ffmpeg)
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="Vocal Accompaniment Generator", version="0.6")

if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


# =========================================================
# SOUND FONT DOWNLOAD (SAFE)
# =========================================================
def ensure_soundfont_safe() -> None:
    """Download SF2 if missing, but never crash the app."""
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

    # fallback to pydub for m4a/mp3
    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    audio = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return audio, sr


# =========================================================
# SMALL AND BIG HELPERS
# =========================================================

def build_style_prompt(style: str, bpm: float | None = None, key: str | None = None) -> str:
    """
    Turn a style string like 'nigerian_gospel' or 'afrobeat' into a rich,
    musicgen-friendly text prompt.
    """
    style = (style or "").lower()
    parts: list[str] = []

    if style in ("nigerian_gospel", "gospel", "worship", "praise"):
        parts.append("Nigerian gospel / worship backing track")
        parts.append("warm grand piano, soft live drums, bass guitar, gentle pads")
        parts.append("no lead vocals, for congregational worship, uplifting, spiritual, African feel")
    elif style == "afrobeat":
        parts.append("modern afrobeat instrumental, west african groove")
        parts.append("live drums, conga, bass guitar, electric piano, guitars")
        parts.append("no vocals, energetic but not too busy, radio mix")
    elif style == "hi_life":
        parts.append("Highlife band from West Africa")
        parts.append("rhythm guitar, shakers, congas, bass, light brass")
        parts.append("no vocals, sunny, joyful")
    elif style == "reggae":
        parts.append("smooth reggae gospel riddim")
        parts.append("offbeat guitar skank, deep bass, light drums")
        parts.append("no vocals, worshipful, laid back")
    else:
        # default
        parts.append(f"{style} backing track")
        parts.append("real instruments, mixed and mastered, no vocals")

    # enrich with tempo and key if we have them
    if bpm:
        parts.append(f"{int(bpm)} BPM")
    if key:
        parts.append(f"in the key of {key}")

    # always enforce that we don't want AI vocals
    parts.append("instrumental only, no singing voice")

    return ", ".join(parts)



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


# =========================================================
# MASTERING (remote → local fallback)
# =========================================================
def call_mastering_api(audio_bytes: bytes) -> bytes:
    """
    Try Auphonic (if configured), otherwise do basic in-app mastering.
    """
    master_enabled = os.environ.get("MASTERING_ENABLED", "false").lower() == "true"
    master_url = os.environ.get("MASTERING_URL")
    master_token = os.environ.get("MASTERING_TOKEN")

    # ----- remote first -----
    if master_enabled and master_url and master_token:
        try:
            files = {"audio_file": ("mix.wav", audio_bytes, "audio/wav")}
            data = {"token": master_token, "output_files[]": "wav"}
            r = requests.post(master_url, data=data, files=files, timeout=180)
            if r.status_code == 200 and len(r.content) > 1000:
                print("[mastering] remote OK")
                return r.content
            else:
                print("[mastering] remote failed:", r.status_code, r.text[:200])
        except Exception as e:
            print("[mastering] remote error:", e)

    # ----- local fallback -----
    try:
        mix, sr = sf.read(io.BytesIO(audio_bytes))
        if mix.ndim > 1:
            mix = mix.mean(axis=1)

        # normalize
        peak = np.max(np.abs(mix)) + 1e-9
        mix = mix / peak * 0.98

        # light RMS gain
        cur_rms = np.sqrt(np.mean(mix ** 2)) + 1e-9
        target_rms = 0.08
        gain = min(target_rms / cur_rms, 1.5)
        mix = mix * gain

        # HPF
        mix = simple_highpass(mix, sr, cutoff=130.0)

        mix = np.clip(mix, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, mix, sr, format="WAV")
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("[mastering] local error:", e)
        return audio_bytes


def call_hf_musicgen_fallback(vocal_bytes, prompt, duration=30):
    url = "https://api-inference.huggingface.co/models/facebook/musicgen-melody"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
    payload = {
        "inputs": prompt,
        "parameters": {"duration": duration}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        return io.BytesIO(resp.content)
    else:
        print("HF API Error:", resp.text)
        return None

# =========================================================
# REPLICATE (MusicGen) CALL
# =========================================================
REPLICATE_MODEL_VERSION = os.environ.get(
    "REPLICATE_MODEL_VERSION",
    "671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
)


def build_gospel_prompt(bpm: float, key_name: str | None) -> str:
    base = (
        "slow Nigerian gospel / worship backing track, warm piano, soft drums, "
        "subtle bass, ambient pads, no vocals, tight timing, mixed"
    )
    parts = [base]
    if bpm:
        parts.append(f"{int(bpm)} BPM")
    if key_name:
        parts.append(f"in key of {key_name}")
    return ", ".join(parts)


def call_replicate_musicgen(
    vocal_bytes: bytes,
    style: str,
    bpm: float | None,
    key: str | None,
    duration: int = 30,
) -> bytes | None:
    """
    Call Replicate's meta/musicgen model and ask it to create an
    instrumental backing that matches the style/tempo/key we detected.

    This version uses a data:audio/... base64 input which matches
    the schema shown on https://replicate.com/meta/musicgen
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    model_version = os.environ.get(
        "REPLICATE_MODEL_VERSION",
        # you already put this in Railway – this keeps it in sync
        "2b5dc5f29cee83fd5cdf8f9c92e555aae7ca2a69b73c5182f3065362b2fa0a45",
    )
    if not api_token:
        print("[replicate] REPLICATE_API_TOKEN missing")
        return None

    # build an intelligent musical prompt
    prompt = build_style_prompt(style, bpm=bpm, key=key)

    # turn uploaded vocal into a data URL (what musicgen expects)
    vocal_b64 = base64.b64encode(vocal_bytes).decode("utf-8")
    data_url = f"data:audio/wav;base64,{vocal_b64}"

    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "version": model_version,
        "input": {
            "prompt": prompt,
            "input_audio": data_url,
            "output_format": "wav",
            "model_version": "stereo-large",  # same as UI
            "duration": duration,
            # you can add "continuation": True here if you want it to follow
            # the vocal more strictly, but some versions don't expose it
        },
    }

    try:
        # 1) create prediction
        resp = requests.post(url, headers=headers, json=payload, timeout=40)
        if resp.status_code not in (200, 201):
            print("[replicate] create failed:", resp.text)
            return None

        if resp.status_code == 402:
            print("[replicate] No credits on your Replicate account. Falling back to local MIDI.")
            return None

        prediction = resp.json()
        pred_id = prediction["id"]
        status = prediction["status"]

        # 2) poll until it's done
        poll_url = f"{url}/{pred_id}"
        while status not in ("succeeded", "failed", "canceled"):
            import time
            time.sleep(4)
            poll_resp = requests.get(poll_url, headers=headers, timeout=40)
            prediction = poll_resp.json()
            status = prediction.get("status", "")

        if status != "succeeded":
            print("[replicate] prediction not successful:", prediction)
            return None

        # 3) download audio
        output = prediction.get("output")
        if isinstance(output, list):
            audio_url = output[0]
        else:
            audio_url = output

        audio_resp = requests.get(audio_url, timeout=60)
        if audio_resp.status_code == 200:
            return audio_resp.content

    except Exception as e:
        print("[replicate] ERROR:", e)

    return None



def call_hf_musicgen(vocal_bytes: bytes, prompt: str, duration: int = 30) -> bytes | None:
    """
    Call a Hugging Face Space that exposes MusicGen (or similar).
    You can set HF_MUSICGEN_URL in Railway to point to your Space.

    Expected Space API (Gradio-like):
    POST { "data": [ prompt, duration ] }  →  returns { "data": [ "https://..." ] }
    """
    space_url = os.environ.get("HF_MUSICGEN_URL")
    if not space_url:
        # no space configured
        return None

    try:
        payload = {
            "data": [
                prompt,
                duration,
            ]
        }
        resp = requests.post(space_url, json=payload, timeout=180)
        if resp.status_code != 200:
            print("[hf-musicgen] bad status:", resp.status_code, resp.text)
            return None

        data = resp.json()

        # Many HF Spaces return a URL in data[0]
        # adjust this to match your actual Space schema
        if isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
            audio_out = data["data"][0]
            # if it's a URL, download it
            if isinstance(audio_out, str) and audio_out.startswith("http"):
                audio_resp = requests.get(audio_out, timeout=120)
                if audio_resp.status_code == 200:
                    return audio_resp.content

        print("[hf-musicgen] unexpected response format:", data)
        return None

    except Exception as e:
        print("[hf-musicgen] ERROR:", e)
        return None



# =========================================================
# MIDI FALLBACK (same as before, but tidy)
# =========================================================
def midi_note_from_freq(freq: float) -> int:
    return int(69 + 12 * np.log2(freq / 440.0))


def degree_to_midi(root_midi: int, deg: str) -> int:
    mapping = {"i": 0, "ii": 2, "iii": 4, "iv": 5, "v": 7, "vi": 9, "vii": 11}
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
    return [degree_to_midi(root_midi, degrees[i % len(degrees)]) for i in range(bars)]


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
        raise ValueError("SoundFont still missing after download attempt")

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
                piano.notes.append(pretty_midi.Note(velocity=85, pitch=n, start=start, end=end))
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
                pretty_midi.Note(velocity=100, pitch=chord_root - 12, start=start, end=end)
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
                    pretty_midi.Note(velocity=110, pitch=36, start=quantize_time(t, sixteenth), end=quantize_time(t + 0.1, sixteenth))
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=70, pitch=60, start=quantize_time(t + half, sixteenth), end=quantize_time(t + half + 0.08, sixteenth))
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=105, pitch=38, start=quantize_time(t + beat, sixteenth), end=quantize_time(t + beat + 0.1, sixteenth))
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=65, pitch=60, start=quantize_time(t + beat + half, sixteenth), end=quantize_time(t + beat + half + 0.08, sixteenth))
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=90, pitch=36, start=quantize_time(t + 2 * beat, sixteenth), end=quantize_time(t + 2 * beat + 0.08, sixteenth))
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=39, start=quantize_time(t + 3 * beat, sixteenth), end=quantize_time(t + 3 * beat + 0.1, sixteenth))
                )
                t += 4 * beat
            else:
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=36, start=quantize_time(t, sixteenth), end=quantize_time(t + 0.1, sixteenth))
                )
                drum.notes.append(
                    pretty_midi.Note(velocity=110, pitch=38, start=quantize_time(t + beat, sixteenth), end=quantize_time(t + beat + 0.1, sixteenth))
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
                            velocity=80, pitch=root_midi + 7, start=q_off, end=q_off + 0.12
                        )
                    )
            tbar += bar_seconds
        pm.instruments.append(guitar)

    # render
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
@app.get("/")
async def root_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "UI not found, but API is running."}


# small helper to turn np audio → wav bytes
def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, sr, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return buf.getvalue()


@app.post("/generate")
async def generate(
    request: Request,
    style: str = Query("nigerian_gospel"),
    piano: bool = Query(True),
    bass: bool = Query(True),
    drums: bool = Query(True),
    guitar: bool = Query(False),
):
    # ------------------ 1. get audio ------------------
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio data received")

    vocal, sr = load_audio_from_bytes(raw)

    # ------------------ 2. analyse vocal ------------------
    f0 = estimate_key_freq(vocal, sr)
    bpm = estimate_tempo(vocal, sr)
    key_name = rough_key_from_freq(f0)

    # build a genre-aware prompt (we already wrote this helper earlier)
    prompt = build_gospel_prompt(bpm, key_name) if "gospel" in style else f"{style} afrobeat backing track, no vocals"

    # ------------------ 3. try Replicate first ------------------
    replicate_band = call_replicate_musicgen(
        vocal_bytes=raw,
        style=style,
        bpm=bpm,
        key=key_name,
        duration=int(min(len(vocal) / sr, 30)),
    )

# If Replicate fails or returns None → try Hugging Face
if not replicate_band_bytes:
    print("⚠️ Replicate failed, switching to Hugging Face fallback...")

    try:
        hf_audio = call_hf_musicgen_fallback(
            vocal_bytes=raw,
            prompt=f"{style} backing track with {key_name} key at {bpm:.1f} BPM",
            duration=int(min(len(vocal) / sr, 30))
        )
        if hf_audio is not None:
            print("✅ Hugging Face fallback succeeded.")
            replicate_band_bytes = hf_audio.read()
        else:
            print("❌ Hugging Face fallback returned None.")
    except Exception as e:
        print("❌ Error calling Hugging Face fallback:", str(e))

    
    if replicate_band is not None:
        print("[pipeline] using Replicate result")
        band_audio, band_sr = load_audio_from_bytes(replicate_band)
        if band_sr != sr:
            band_audio = librosa.resample(band_audio, orig_sr=band_sr, target_sr=sr)

        # mix
        band_audio = simple_highpass(band_audio, sr, cutoff=130.0)
        v_r = rms(vocal); b_r = rms(band_audio)
        if b_r > 0:
            band_audio = band_audio * (v_r / (b_r * 1.4))

        mix = vocal + 0.8 * band_audio
        peak = np.max(np.abs(mix)) + 1e-9
        if peak > 1.0:
            mix = mix / peak

        return StreamingResponse(
            io.BytesIO(_to_wav_bytes(mix, sr)),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
        )

    # ------------------ 4. Replicate failed → try Hugging Face Space ------------------
    hf_band = call_hf_musicgen(
        vocal_bytes=raw,
        prompt=prompt,
        duration=int(min(len(vocal) / sr, 30)),
    )
    if hf_band is not None:
        print("[pipeline] using HF Space result")
        band_audio, band_sr = load_audio_from_bytes(hf_band)
        if band_sr != sr:
            band_audio = librosa.resample(band_audio, orig_sr=band_sr, target_sr=sr)

        # mix
        band_audio = simple_highpass(band_audio, sr, cutoff=130.0)
        v_r = rms(vocal); b_r = rms(band_audio)
        if b_r > 0:
            band_audio = band_audio * (v_r / (b_r * 1.4))

        mix = vocal + 0.75 * band_audio
        peak = np.max(np.abs(mix)) + 1e-9
        if peak > 1.0:
            mix = mix / peak

        return StreamingResponse(
            io.BytesIO(_to_wav_bytes(mix, sr)),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
        )

    # ------------------ 5. HF also failed → FALLBACK to local MIDI ------------------
    print("[pipeline] both Replicate and HF failed → using local MIDI")
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
    v_r = rms(vocal); b_r = rms(band)
    if b_r > 0:
        band = band * (v_r / (b_r * 1.5))

    mix = vocal + 0.85 * band
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix = mix / peak

    return StreamingResponse(
        io.BytesIO(_to_wav_bytes(mix, sr)),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="accompaniment.wav"'},
    )
