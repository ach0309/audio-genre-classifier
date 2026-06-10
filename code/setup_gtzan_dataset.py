"""Download and stage the GTZAN dataset for local project EDA.

The audio data is intentionally kept under data/, which is ignored by git.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KAGGLE_DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"
DEFAULT_DESTINATION = Path("data/genres_original")


def find_genres_original(download_path: Path) -> Path:
    """Find the GTZAN genres_original folder inside a downloaded dataset."""
    matches = [
        path
        for path in download_path.rglob("genres_original")
        if path.is_dir() and any(child.is_dir() for child in path.iterdir())
    ]
    if not matches:
        raise FileNotFoundError(
            "Could not find a genres_original folder in the downloaded dataset."
        )
    return matches[0]


def stage_dataset(source: Path, destination: Path, overwrite: bool = False) -> None:
    """Copy GTZAN genre folders into the repo-local data directory."""
    if destination.exists():
        if not overwrite:
            print(f"Dataset already exists at {destination}")
            print("Use --overwrite to replace it.")
            return
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"Staged GTZAN audio files at {destination}")


def download_dataset() -> Path:
    """Download GTZAN with kagglehub and return the local cache path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download GTZAN. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Existing dataset folder to stage instead of downloading from Kaggle.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Repo-local destination for genres_original.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.source:
        source_path = args.source
        if source_path.name != "genres_original":
            source_path = find_genres_original(source_path)
    else:
        source_path = find_genres_original(download_dataset())

    stage_dataset(source_path, args.destination, overwrite=args.overwrite)
