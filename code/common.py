from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader, Dataset


RANDOM_SEED = 42
SAMPLE_RATE = 22_050
CLIP_DURATION_SECONDS = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 16
FREQ_MASK_MAX_WIDTH = 16

FIXED_NUM_SAMPLES = SAMPLE_RATE * CLIP_DURATION_SECONDS


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / "code").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Could not find the audio-genre-classifier repo root.")


def build_engine(repo_root: Path):
    load_dotenv(repo_root / ".env")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD", "")
    db_host = "localhost"
    db_port = 5432
    db_name = "audio_genre_classifier"
    return create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )


def load_split(split_name: str, engine, repo_root: Path) -> pd.DataFrame:
    """Load usable non-corrupted, non-duplicate songs for one split."""
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"split_name must be 'train', 'val', or 'test'; got {split_name!r}")

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
    df["file_path"] = df["file_path"].apply(lambda path: str(repo_root / path))
    return df


def load_label_mapping(engine) -> tuple[dict[str, int], dict[int, str]]:
    query = text("""
        SELECT DISTINCT l.label_name AS label
        FROM   audio_clips ac
        JOIN   labels l ON l.label_id = ac.label_id
        WHERE  ac.is_corrupted = FALSE
          AND  ac.is_duplicate = FALSE
        ORDER  BY l.label_name
    """)
    labels = pd.read_sql(sql=query, con=engine)["label"].tolist()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def pad_or_truncate(audio: np.ndarray, target_num_samples: int) -> np.ndarray:
    if len(audio) > target_num_samples:
        return audio[:target_num_samples]
    if len(audio) < target_num_samples:
        return np.pad(audio, (0, target_num_samples - len(audio)), mode="constant")
    return audio


def augment_waveform(audio: np.ndarray) -> np.ndarray:
    gain = np.random.uniform(0.8, 1.2)
    audio = audio * gain
    max_shift = int(0.1 * SAMPLE_RATE)
    shift = np.random.randint(-max_shift, max_shift + 1)
    audio = np.roll(audio, shift)
    noise = np.random.normal(0, 0.005, size=audio.shape)
    return (audio + noise).astype(np.float32)


def audio_to_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def augment_mel_spectrogram(mel: np.ndarray) -> np.ndarray:
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


def build_dataloader(
    split_name: str,
    engine,
    repo_root: Path,
    label_to_idx: dict[str, int],
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    split_df = load_split(split_name, engine, repo_root)
    dataset = AudioDataset(split_df, label_to_idx=label_to_idx, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
