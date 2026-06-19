"""
Scans data/genres_original/ for wav files, assigns 70/15/15 train/val/test
splits at the song level, and populates the labels and audio_clips tables.

All 1,000 songs are inserted. jazz.00054.wav is flagged is_corrupted=True
and will be filtered out by load_split() — it is never used for training.

file_path is stored relative to the repo root so it works on any machine.

Requires a .env file in the project root (see .env.example).

Usage:
    python db/populate.py
"""

import os
import random
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / '.env')

DB_USER     = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_HOST     = 'localhost'
DB_PORT     = 5432
DB_NAME     = 'audio_genre_classifier'

AUDIO_DIR     = ROOT / 'data' / 'genres_original'
CORRUPT_FILES = {'jazz.00054.wav'}
RANDOM_SEED   = 42
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15


def get_engine():
    return create_engine(
        f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )


def collect_songs() -> list[dict]:
    """Return all wav files. Corrupt files are included and flagged."""
    songs = []
    for wav_path in sorted(AUDIO_DIR.glob('*/*.wav')):
        label = wav_path.parent.name
        rel_path = str(wav_path.relative_to(ROOT))
        songs.append({
            'file_path':    rel_path,
            'label':        label,
            'is_corrupted': wav_path.name in CORRUPT_FILES,
            'is_duplicate': False,
        })
    return songs


def assign_splits(songs: list[dict]) -> list[dict]:
    """Shuffle all songs with a fixed seed and assign 70/15/15 train/val/test."""
    shuffled = songs[:]
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    for i, song in enumerate(shuffled):
        if i < n_train:
            song['split'] = 'train'
        elif i < n_train + n_val:
            song['split'] = 'val'
        else:
            song['split'] = 'test'

    return shuffled


def main():
    print(f'Scanning {AUDIO_DIR} ...')
    songs = collect_songs()
    n_corrupt = sum(1 for s in songs if s['is_corrupted'])
    print(f'  {len(songs)} songs found  ({n_corrupt} corrupted, flagged in DB)')

    songs = assign_splits(songs)
    df = pd.DataFrame(songs)

    engine = get_engine()

    with engine.connect() as con:
        con.execute(text('TRUNCATE audio_clips, labels RESTART IDENTITY CASCADE'))
        con.commit()

    # Insert labels (sample_count counts only usable songs)
    labels_df = (
        df[~df['is_corrupted']]
        .groupby('label', as_index=False)
        .size()
        .rename(columns={'label': 'label_name', 'size': 'sample_count'})
        .sort_values('label_name')
        .reset_index(drop=True)
    )
    labels_df.to_sql('labels', engine, if_exists='append', index=False, method='multi')
    print(f'\nInserted {len(labels_df)} labels.')

    # Fetch label_ids back and join
    label_map = pd.read_sql(sql=text('SELECT label_id, label_name FROM labels'), con=engine)
    df = df.merge(label_map, left_on='label', right_on='label_name').drop(
        columns=['label', 'label_name']
    )

    # Insert audio_clips
    clips_df = df[['file_path', 'label_id', 'split', 'is_corrupted', 'is_duplicate']]
    clips_df.to_sql(
        name      = 'audio_clips',
        con       = engine,
        if_exists = 'append',
        index     = False,
        method    = 'multi',
        chunksize = 500,
    )

    usable = df[~df['is_corrupted']]
    print('\nUsable songs per split:')
    for split in ['train', 'val', 'test']:
        print(f'  {split}: {len(usable[usable["split"] == split])}')
    print(f'  total usable : {len(usable)}')
    print(f'  corrupted    : {df["is_corrupted"].sum()}  (in DB, filtered from training)')


if __name__ == '__main__':
    main()
