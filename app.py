from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os 
import shutil

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
    temp_dir = frf"temp/{session_id}"
    os.makedirs(temp_dir, exist_ok=True)

    vocal_path = os.path.join(temp_dir, "vocal.wav")
    with open(vocal_path, "wb") as f:
        shutil.copyfileobj (vocal.file, f)

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