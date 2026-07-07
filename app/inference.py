"""Loads the trained checkpoint once and predicts a genre for an uploaded audio file.

Reuses the exact preprocessing pipeline and architecture from code/common.py so a
prediction here goes through the same steps as evaluation in 4_predict.ipynb: load
audio -> pad/truncate to 30s -> log-mel spectrogram -> AudioCNN -> softmax.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "code").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Could not find the audio-genre-classifier repo root.")


REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(REPO_ROOT / "code"))

import librosa
import torch
import torch.nn.functional as F

from common import AudioCNN, FIXED_NUM_SAMPLES, SAMPLE_RATE, audio_to_mel_spectrogram, pad_or_truncate

CHECKPOINT_PATH = REPO_ROOT / "models" / "best_model.pth"

_model: torch.nn.Module | None = None
_idx_to_label: dict[int, str] | None = None


def load_model() -> None:
    """Loads the checkpoint into module-level globals. Call once at app startup."""
    global _model, _idx_to_label

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} does not exist. Run code/3_model.ipynb first to train "
            "and save the checkpoint."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    idx_to_label = {int(idx): label for idx, label in checkpoint["idx_to_label"].items()}

    model = AudioCNN(num_classes=len(idx_to_label))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _idx_to_label = idx_to_label


def predict_genre(audio_path: str) -> list[tuple[str, float]]:
    """Returns [(genre, probability), ...] sorted by probability, most likely first."""
    if _model is None or _idx_to_label is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = pad_or_truncate(audio, FIXED_NUM_SAMPLES)
    mel = audio_to_mel_spectrogram(audio)

    features = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = _model(features)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    results = [(_idx_to_label[idx], probabilities[idx].item()) for idx in range(len(_idx_to_label))]
    return sorted(results, key=lambda pair: pair[1], reverse=True)
