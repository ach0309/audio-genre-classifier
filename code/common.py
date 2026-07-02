"""Shared helpers/models for the 2_transform / 3_model / 4_predict notebooks.

!! DO NOT MOVE THIS FILE OUT OF code/ !!
2_transform.ipynb, 3_model.ipynb, and 4_predict.ipynb all import this module
as a plain sibling file (`from common import ...`) via a `sys.path` entry for
their own directory. Moving common.py anywhere else will break every one of
those imports on the next kernel restart.

This is a plain module, not a notebook — it has no kernel state of its own, so
each notebook can still be independently "restart kernel, run top to bottom":
they just import a static file from disk, the same as any other dependency
(e.g. librosa). What's lost versus copy-paste is the "hand someone one .ipynb
and nothing else" property; what's gained is that a fix or guard only needs to
be made once instead of hand-synced across three copies.

See the bottom of this file for a teammate setup/run checklist.
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import text
from torch.utils.data import Dataset

RANDOM_SEED = 42
SAMPLE_RATE = 22_050
CLIP_DURATION_SECONDS = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FREQ_MASK_MAX_WIDTH = 16
FIXED_NUM_SAMPLES = SAMPLE_RATE * CLIP_DURATION_SECONDS


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "code").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Could not find the audio-genre-classifier repo root.")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)


def load_split(split_name: str, engine) -> pd.DataFrame:
    """Load usable (non-corrupted, non-duplicate) songs for one split."""
    assert split_name in ("train", "val", "test"), \
        f"split_name must be 'train', 'val', or 'test' — got '{split_name}'"
    query = text("""
        SELECT ac.file_path, l.label_name AS label, ac.split
        FROM   audio_clips ac
        JOIN   labels l ON l.label_id = ac.label_id
        WHERE  ac.split        = :split_name
          AND  ac.is_corrupted = FALSE
          AND  ac.is_duplicate = FALSE
        ORDER  BY ac.file_path
    """)
    df = pd.read_sql(sql=query, con=engine, params={"split_name": split_name})
    df["file_path"] = df["file_path"].apply(lambda p: str(REPO_ROOT / p))
    return df


def pad_or_truncate(audio: np.ndarray, target_num_samples: int) -> np.ndarray:
    """Return audio with exactly target_num_samples samples."""
    if len(audio) > target_num_samples:
        return audio[:target_num_samples]

    if len(audio) < target_num_samples:
        padding = target_num_samples - len(audio)
        return np.pad(audio, (0, padding), mode="constant")

    return audio


def augment_waveform(audio: np.ndarray) -> np.ndarray:
    """Apply light waveform augmentations for training examples only."""
    # Random gain changes volume without changing genre identity.
    gain = np.random.uniform(0.8, 1.2)
    audio = audio * gain

    # Small time shift makes the model less dependent on exact alignment.
    max_shift = int(0.1 * SAMPLE_RATE)
    shift = np.random.randint(-max_shift, max_shift + 1)
    audio = np.roll(audio, shift)

    # Low-amplitude noise can improve robustness.
    noise = np.random.normal(0, 0.005, size=audio.shape)
    audio = audio + noise

    return audio.astype(np.float32)


def audio_to_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Convert waveform audio into a normalized log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize each spectrogram to a stable range for model input.
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def augment_mel_spectrogram(mel: np.ndarray) -> np.ndarray:
    """Apply light frequency masking for training spectrograms only."""
    mel = mel.copy()
    max_width = min(FREQ_MASK_MAX_WIDTH, mel.shape[0])
    mask_width = np.random.randint(0, max_width + 1)

    if mask_width > 0:
        start = np.random.randint(0, mel.shape[0] - mask_width + 1)
        mel[start:start + mask_width, :] = 0.0

    return mel.astype(np.float32)


class AudioDataset(Dataset):
    """PyTorch dataset that loads wav files and returns mel spectrogram tensors."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_idx: dict[str, int],
        sample_rate: int = SAMPLE_RATE,
        fixed_num_samples: int = FIXED_NUM_SAMPLES,
        augment: bool = False,
    ):
        expected_columns = {"file_path", "label", "split"}
        missing_columns = expected_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Dataframe is missing columns: {sorted(missing_columns)}")
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        self.fixed_num_samples = fixed_num_samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        audio, _ = librosa.load(row["file_path"], sr=self.sample_rate, mono=True)
        audio = pad_or_truncate(audio, self.fixed_num_samples)
        if self.augment and row["split"] == "train":
            audio = augment_waveform(audio)
        mel = audio_to_mel_spectrogram(audio)
        if self.augment and row["split"] == "train":
            mel = augment_mel_spectrogram(mel)
        features = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        return features, target


class AudioCNN(nn.Module):
    """3-block CNN for mel spectrogram genre classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# Teammate checklist — what you need to do to make sure everything still runs
# -----------------------------------------------------------------------------
# This file is imported by 2_transform.ipynb, 3_model.ipynb, and 4_predict.ipynb
# (see the "!! DO NOT MOVE !!" note at the top). If you're pulling this change
# for the first time:
#
# 1. Pull this branch so you get common.py *and* the updated notebooks
#    together. Grabbing only the .ipynb files without this file will break
#    their imports.
#
# 2. No new dependencies to install — this module only uses libraries the
#    notebooks already imported (librosa, torch, pandas, numpy, sqlalchemy).
#    Nothing new to add to requirements.
#
# 3. Restart your kernel before running any of the three notebooks if you had
#    one open from before this change — an old kernel won't know about the
#    new common.py import.
#
# 4. Run the notebooks in this order, since 4_predict.ipynb now depends on a
#    checkpoint that only 3_model.ipynb produces:
#      2_transform.ipynb -> 3_model.ipynb (writes models/best_model.pth,
#      takes a few minutes to train) -> 4_predict.ipynb (fails fast with a
#      clear error if step 2 hasn't been run yet, same as before).
#
# 5. Same environment requirements as always — a running local Postgres with
#    the audio_genre_classifier DB populated, and a .env with DB credentials.
#    This refactor doesn't change any of that.
#
# 6. Keep common.py in code/, next to the notebooks — don't move it. The
#    import relies on it being a sibling file in the same folder.
