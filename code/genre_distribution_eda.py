"""Run first-pass EDA checks for the local GTZAN audio dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba-cache")))

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path("data/genres_original")
EXPECTED_GENRES = {
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
}


def discover_audio_files(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Return one row per GTZAN .wav file with its genre label."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}. "
            "Run `python code/setup_gtzan_dataset.py` or place GTZAN at "
            "`data/genres_original`."
        )

    rows = []
    for genre_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        for wav_file in sorted(genre_dir.glob("*.wav")):
            rows.append(
                {
                    "genre": genre_dir.name,
                    "file_name": wav_file.name,
                    "file_path": str(wav_file),
                }
            )

    return pd.DataFrame(rows, columns=["genre", "file_name", "file_path"])


def build_genre_counts(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Count .wav files in each genre folder and return a tidy DataFrame."""
    try:
        audio_files = discover_audio_files(dataset_path)
    except FileNotFoundError as exc:
        print(exc)
        return pd.DataFrame(columns=["genre", "count"])

    return (
        audio_files.groupby("genre", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("genre")
        .reset_index(drop=True)
    )


def validate_gtzan_layout(genre_counts: pd.DataFrame) -> pd.DataFrame:
    """Return expected genre counts with a status column for quick QA."""
    if genre_counts.empty:
        return pd.DataFrame(columns=["genre", "count", "status"])

    counts_by_genre = dict(zip(genre_counts["genre"], genre_counts["count"]))
    rows = []
    for genre in sorted(EXPECTED_GENRES | set(counts_by_genre)):
        count = counts_by_genre.get(genre, 0)
        if genre not in EXPECTED_GENRES:
            status = "unexpected genre"
        elif count == 100:
            status = "ok"
        elif count == 0:
            status = "missing genre"
        else:
            status = "expected 100 files"

        rows.append({"genre": genre, "count": count, "status": status})

    return pd.DataFrame(rows, columns=["genre", "count", "status"])


def build_audio_metadata(
    dataset_path: Path = DATASET_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load GTZAN files with librosa and return basic audio metadata."""
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for audio metadata. Install project "
            "dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    audio_files = discover_audio_files(dataset_path)
    if limit is not None:
        audio_files = audio_files.head(limit)

    rows = []
    for row in audio_files.itertuples(index=False):
        samples, sample_rate = librosa.load(row.file_path, sr=None, mono=True)
        rows.append(
            {
                "genre": row.genre,
                "file_name": row.file_name,
                "sample_rate": sample_rate,
                "sample_count": len(samples),
                "duration_seconds": librosa.get_duration(
                    y=samples,
                    sr=sample_rate,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_genre_distribution(
    genre_counts: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plot a simple genre class distribution bar chart."""
    if genre_counts.empty:
        print("No genre counts available to plot.")
        return

    ax = genre_counts.plot(
        kind="bar",
        x="genre",
        y="count",
        legend=False,
        color="steelblue",
        figsize=(10, 5),
    )
    ax.set_title("GTZAN Genre Class Distribution")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of .wav files")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved genre distribution chart to {output_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for repeatable local EDA runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATASET_PATH,
        help="Path to the GTZAN genres_original directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/genre_distribution.png"),
        help="Where to save the genre distribution chart.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart interactively instead of saving it.",
    )
    parser.add_argument(
        "--audio-summary",
        action="store_true",
        help="Load audio files with librosa and print duration/sample-rate EDA.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional file limit for quicker librosa metadata checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    counts_df = build_genre_counts(args.dataset_path)
    print("\nGenre counts")
    print(counts_df.to_string(index=False))

    layout_df = validate_gtzan_layout(counts_df)
    print("\nLayout validation")
    print(layout_df.to_string(index=False))

    output_path = None if args.show else args.output_path
    plot_genre_distribution(counts_df, output_path=output_path)

    if args.audio_summary:
        metadata_df = build_audio_metadata(args.dataset_path, limit=args.limit)
        print("\nAudio metadata summary")
        print(
            metadata_df.groupby("genre")
            .agg(
                files=("file_name", "count"),
                avg_duration_seconds=("duration_seconds", "mean"),
                min_sample_rate=("sample_rate", "min"),
                max_sample_rate=("sample_rate", "max"),
            )
            .round(2)
            .to_string()
        )
