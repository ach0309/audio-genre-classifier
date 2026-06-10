"""Plot example MFCCs for each GTZAN genre."""

from __future__ import annotations

import argparse
from pathlib import Path

from genre_distribution_eda import DATASET_PATH, EXPECTED_GENRES, discover_audio_files

import matplotlib.pyplot as plt


def build_example_file_paths(dataset_path: Path = DATASET_PATH) -> dict[str, Path]:
    """Choose one deterministic example .wav file per genre."""
    audio_files = discover_audio_files(dataset_path)
    examples = {}

    for genre in sorted(EXPECTED_GENRES):
        genre_files = audio_files[audio_files["genre"] == genre].sort_values("file_name")
        if genre_files.empty:
            continue
        examples[genre] = Path(genre_files.iloc[0]["file_path"])

    return examples


def plot_mfcc_examples(
    dataset_path: Path = DATASET_PATH,
    output_path: Path | None = Path("reports/mfcc_examples_by_genre.png"),
    n_mfcc: int = 13,
    duration: float = 30.0,
) -> None:
    """Load one audio example per genre and save/display MFCC plots."""
    try:
        import librosa
        import librosa.display
    except ImportError as exc:
        raise ImportError(
            "librosa is required for MFCC plotting. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    examples = build_example_file_paths(dataset_path)
    if not examples:
        print(f"No GTZAN .wav files found at {dataset_path}")
        return

    fig, axes = plt.subplots(5, 2, figsize=(12, 14), constrained_layout=True)
    axes = axes.flatten()

    for ax, genre in zip(axes, sorted(examples)):
        samples, sample_rate = librosa.load(
            examples[genre],
            sr=None,
            mono=True,
            duration=duration,
        )
        mfccs = librosa.feature.mfcc(
            y=samples,
            sr=sample_rate,
            n_mfcc=n_mfcc,
        )
        image = librosa.display.specshow(
            mfccs,
            x_axis="time",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(f"{genre}: {examples[genre].name}")
        ax.set_ylabel("MFCC")
        ax.set_xlabel("Time")

    for ax in axes[len(examples) :]:
        ax.axis("off")

    fig.suptitle("GTZAN MFCC Examples by Genre", fontsize=16)
    fig.colorbar(image, ax=axes, format="%+2.0f dB", shrink=0.65)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved MFCC examples chart to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
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
        default=Path("reports/mfcc_examples_by_genre.png"),
        help="Where to save the MFCC examples chart.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart interactively instead of saving it.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients to plot.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds to load from each example file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = None if args.show else args.output_path
    plot_mfcc_examples(
        dataset_path=args.dataset_path,
        output_path=output_path,
        n_mfcc=args.n_mfcc,
        duration=args.duration,
    )
