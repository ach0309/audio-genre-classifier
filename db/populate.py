"""
Scans data/genres_original/ for wav files, assigns 70/15/15 train/val/test
splits at the song level, and populates the audio_clips table.

file_path is stored relative to the repo root (e.g. data/genres_original/blues/blues.00000.wav)
so it works on any teammate's machine regardless of where the repo is cloned.

Requires a .env file in the project root (see .env.example).

Usage:
    python db/populate.py
"""

import os
import random
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / '.env')

AUDIO_DIR = ROOT / 'data' / 'genres_original'
CORRUPT_FILE = 'jazz.00054.wav'
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def get_connection():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ.get('POSTGRES_PASSWORD', ''),
    )


def collect_songs() -> list[tuple[str, str]]:
    """Return (relative_path, label) for every non-corrupt wav in genres_original/."""
    songs = []
    for wav_path in sorted(AUDIO_DIR.glob('*/*.wav')):
        if wav_path.name == CORRUPT_FILE:
            continue
        label = wav_path.parent.name
        rel_path = str(wav_path.relative_to(ROOT))
        songs.append((rel_path, label))
    return songs


def assign_splits(songs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Shuffle songs with a fixed seed and assign 70/15/15 train/val/test."""
    shuffled = songs[:]
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    rows = []
    for i, (path, label) in enumerate(shuffled):
        if i < n_train:
            split = 'train'
        elif i < n_train + n_val:
            split = 'val'
        else:
            split = 'test'
        rows.append((path, label, split))

    return rows


def main():
    print(f'Scanning {AUDIO_DIR} ...')
    songs = collect_songs()
    print(f'  {len(songs)} songs  ({CORRUPT_FILE} excluded)')

    rows = assign_splits(songs)

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute('TRUNCATE audio_clips RESTART IDENTITY')
            execute_values(
                cur,
                'INSERT INTO audio_clips (file_path, label, split) VALUES %s',
                rows,
            )
    conn.close()

    from collections import Counter
    counts = Counter(r[2] for r in rows)
    print('\nInserted rows:')
    for split in ['train', 'val', 'test']:
        print(f'  {split}: {counts[split]}')
    print(f'  total: {len(rows)}')


if __name__ == '__main__':
    main()
