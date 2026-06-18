"""
Reads data/features_3_sec.csv, assigns 70/15/15 train/val/test splits at the
parent-song level, and populates the audio_clips table.

Requires:
    DATABASE_URL env var, e.g.:
    export DATABASE_URL="postgresql://localhost/audio_genre_classifier"

Usage:
    python db/populate.py
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_3_sec.csv')
CORRUPT_SONG = 'jazz.00054'
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # e.g. "blues.00000.3.wav" → strip .wav → "blues.00000.3" → "blues.00000"
    df['parent'] = df['filename'].str.replace(r'\.wav$', '', regex=True).str.rsplit('.', n=1).str[0]

    # Drop all clips from the corrupt recording
    df = df[df['parent'] != CORRUPT_SONG].reset_index(drop=True)

    # Shuffle unique parent songs with a fixed seed for reproducibility
    parents = sorted(df['parent'].unique())
    shuffled = pd.Series(parents).sample(frac=1, random_state=RANDOM_SEED).tolist()

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    split_map = {}
    for i, song in enumerate(shuffled):
        if i < n_train:
            split_map[song] = 'train'
        elif i < n_train + n_val:
            split_map[song] = 'val'
        else:
            split_map[song] = 'test'

    df['split'] = df['parent'].map(split_map)
    return df


def main():
    print(f'Loading {CSV_PATH} ...')
    df = pd.read_csv(CSV_PATH, usecols=['filename', 'label'])
    print(f'  {len(df):,} rows loaded')

    df = assign_splits(df)
    print(f'  {len(df):,} rows after excluding {CORRUPT_SONG!r}')

    db_url = os.environ['DATABASE_URL']
    conn = psycopg2.connect(db_url)

    with conn:
        with conn.cursor() as cur:
            cur.execute('TRUNCATE audio_clips RESTART IDENTITY')
            rows = list(zip(df['filename'], df['label'], df['split']))
            execute_values(
                cur,
                'INSERT INTO audio_clips (file_path, label, split) VALUES %s',
                rows,
            )

    conn.close()

    summary = df.groupby('split').size().reindex(['train', 'val', 'test'])
    print('\nInserted rows:')
    for split_name, count in summary.items():
        print(f'  {split_name}: {count:,}')
    print(f'  total: {summary.sum():,}')


if __name__ == '__main__':
    main()
