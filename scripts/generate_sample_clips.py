"""One-time dev tool: copies 5 full GTZAN clips into app/static/samples/,
used by the demo app's "try a sample" chips.

Requires data/genres_original/ locally (not committed to git — see README).
Re-run only if you want to swap which clips are used.
"""

from pathlib import Path

import librosa
import soundfile as sf

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "data" / "genres_original"
OUT_DIR = REPO_ROOT / "app" / "static" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050

# Full clips, not trimmed: the model always evaluates a fixed 30s window
# (pad_or_truncate). A short trimmed sample gets zero-padded to fill that
# window, which renders as a mostly-black mel spectrogram in the demo's
# "what the model sees" visualization. A full clip fills the window exactly
# like a real upload would (most real songs get truncated, not padded).

SELECTIONS = [
    ("blues", "00000"),
    ("classical", "00000"),
    ("hiphop", "00000"),
    ("metal", "00000"),
    ("reggae", "00000"),
]

for genre, track_num in SELECTIONS:
    src = SRC_DIR / genre / f"{genre}.{track_num}.wav"
    audio, sr = librosa.load(src, sr=SAMPLE_RATE, mono=True)
    out_path = OUT_DIR / f"{genre}_sample.wav"
    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(out_path, out_path.stat().st_size, "bytes")
