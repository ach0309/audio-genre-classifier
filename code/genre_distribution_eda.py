"""Scaffold genre class distribution EDA for the GTZAN dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# TODO: Update this path after the team lead's GTZAN loading work is merged.
DATASET_PATH = Path("data/genres_original")


def build_genre_counts(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Count .wav files in each genre folder and return a tidy DataFrame."""
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        print("Update DATASET_PATH after the GTZAN dataset-loading work is merged.")
        return pd.DataFrame(columns=["genre", "count"])

    genre_counts = []

    for genre_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        wav_count = sum(1 for wav_file in genre_dir.glob("*.wav") if wav_file.is_file())
        genre_counts.append({"genre": genre_dir.name, "count": wav_count})

    return pd.DataFrame(genre_counts, columns=["genre", "count"])


def plot_genre_distribution(genre_counts: pd.DataFrame) -> None:
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
    plt.show()


if __name__ == "__main__":
    counts_df = build_genre_counts()
    print(counts_df)
    plot_genre_distribution(counts_df)
