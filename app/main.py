"""FastAPI backend for the genre-classifier demo.

Run with:
    uvicorn app.main:app --reload

Then open http://localhost:8000 in a browser.
"""

from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.inference import analyze_audio, load_model

app = FastAPI(title="Audio Genre Classifier Demo")

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@app.on_event("startup")
def _startup() -> None:
    load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        try:
            result = analyze_audio(tmp.name)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc

    return {
        "predictions": [
            {"genre": genre, "probability": probability} for genre, probability in result["predictions"]
        ],
        "spectrogram_png": result["spectrogram_png"],
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
