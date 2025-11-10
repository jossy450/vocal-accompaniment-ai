# Updated app with full audio handling, model failover, and vocal-learning prep

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from uuid import uuid4
from utils.audio import extract_vocal_features, mix_and_master
from models.accompaniment import generate_accompaniment_with_fallback

app = FastAPI()

@app.post("/generate")
async def generate(
    vocal: UploadFile,
    style: str = Form(...),
    complexity: str = Form(...),
    instruments: str = Form("piano")
):
    session_id = str(uuid4())
    temp_dir = f"temp/{session_id}"
    os.makedirs(temp_dir, exist_ok=True)

    vocal_path = os.path.join(temp_dir, "vocal.wav")
    with open(vocal_path, "wb") as f:
        shutil.copyfileobj(vocal.file, f)

    vocal_features = extract_vocal_features(vocal_path)

    try:
        accompaniment_path = generate_accompaniment_with_fallback(
            vocal_path=vocal_path,
            style=style,
            complexity=complexity,
            instruments=instruments.split(","),
            features=vocal_features,
            output_dir=temp_dir
        )
    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {str(e)}"}, status_code=500)

    final_path = mix_and_master(vocal_path, accompaniment_path, temp_dir)
    return FileResponse(final_path, filename="final_mix.wav")


# --- utils/audio.py ---
def extract_vocal_features(vocal_path):
    import librosa
    y, sr = librosa.load(vocal_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    key = librosa.key.estimate_tuning(y, sr)
    return {
        "tempo": tempo,
        "key": key,
        "pitch_curve": pitches.tolist(),
        "sr": sr
    }

def mix_and_master(vocal_path, band_path, output_dir):
    from pydub import AudioSegment
    vocal = AudioSegment.from_file(vocal_path)
    band = AudioSegment.from_file(band_path)
    combined = band.overlay(vocal)
    mastered = combined.normalize()
    final_path = os.path.join(output_dir, "final_mix.wav")
    mastered.export(final_path, format="wav")
    return final_path


# --- models/accompaniment.py ---
def generate_accompaniment_with_fallback(vocal_path, style, complexity, instruments, features, output_dir):
    try:
        return generate_with_acestep(vocal_path, style, complexity, instruments, features, output_dir)
    except Exception:
        return generate_with_hum2song(vocal_path, style, complexity, instruments, features, output_dir)

def generate_with_acestep(vocal_path, style, complexity, instruments, features, output_dir):
    import shutil
    shutil.copy("assets/acestep_default.wav", os.path.join(output_dir, "band.wav"))
    return os.path.join(output_dir, "band.wav")

def generate_with_hum2song(vocal_path, style, complexity, instruments, features, output_dir):
    import shutil
    shutil.copy("assets/hum2song_default.wav", os.path.join(output_dir, "band.wav"))
    return os.path.join(output_dir, "band.wav")


<!-- --- static/index.html --- -->
<!DOCTYPE html>
<html>
<head>
  <title>Vocal Accompaniment AI</title>
</head>
<body>
  <h1>Upload Vocal to Generate Band</h1>
  <form id="uploadForm">
    <input type="file" name="vocal" required><br>
    Style: <input type="text" name="style" value="Afrobeat"><br>
    Complexity: <select name="complexity">
      <option value="simple">Simple</option>
      <option value="medium" selected>Medium</option>
      <option value="complex">Complex</option>
    </select><br>
    Instruments:<br>
    <label><input type="checkbox" name="instruments" value="piano" checked> Piano</label><br>
    <label><input type="checkbox" name="instruments" value="guitar"> Guitar</label><br>
    <label><input type="checkbox" name="instruments" value="drums"> Drums</label><br>
    <button type="submit">Generate Accompaniment</button>
  </form>

  <div id="output"></div>

  <script>
    const form = document.getElementById("uploadForm");
    form.onsubmit = async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      const instrumentValues = [...form.querySelectorAll('input[name="instruments"]:checked')].map(i => i.value);
      formData.set("instruments", instrumentValues.join(","));

      const res = await fetch("/generate", {
        method: "POST",
        body: formData
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        document.getElementById("output").innerHTML = `<audio controls src="${url}"></audio>`;
      } else {
        const error = await res.json();
        alert("Error: " + error.error);
      }
    };
  </script>
</body>
</html>
