"""
Migrates feature data from features_3_sec.csv into the audio_features table.

Steps:
  1. Print CSV column names for confirmation
  2. Drop jazz.00054 rows (corrupt file)
  3. CREATE TABLE audio_features with one column per feature
  4. Bulk-insert all rows
  5. Verify: row count, null check, spot-check JOIN with audio_clips
  6. Add foreign key: audio_features.file_path -> audio_clips.file_path

Usage:
  python db/migrate_features.py

Requires a .env file in the project root (see .env.example).
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / '.env')

CSV_PATH = ROOT / 'data' / 'features_3_sec.csv'
CORRUPT_SONG = 'jazz.00054'

# Columns that should NOT become feature columns in audio_features
EXCLUDE_COLS = {'filename', 'label'}


def get_connection():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ.get('POSTGRES_PASSWORD', ''),
    )


def load_and_clean_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    print('── CSV columns ──────────────────────────────')
    for i, col in enumerate(df.columns, 1):
        print(f'  {i:>2}. {col}')
    print(f'\nTotal: {len(df.columns)} columns, {len(df):,} rows\n')

    before = len(df)
    mask = df['filename'].str.contains(CORRUPT_SONG, regex=False)
    df = df[~mask].reset_index(drop=True)
    print(f'Dropped {before - len(df)} rows ({CORRUPT_SONG})')
    print(f'Remaining: {len(df):,} rows\n')

    return df


def build_create_table(feature_cols: list[str]) -> str:
    col_defs = ['file_path TEXT PRIMARY KEY']
    for col in feature_cols:
        dtype = 'BIGINT' if col == 'length' else 'DOUBLE PRECISION'
        col_defs.append(f'{col} {dtype}')
    return 'CREATE TABLE IF NOT EXISTS audio_features (\n    ' + ',\n    '.join(col_defs) + '\n);'


def migrate(df: pd.DataFrame, conn):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    ddl = build_create_table(feature_cols)
    print('── Creating audio_features table ────────────')
    print(ddl[:300] + ' …')

    with conn:
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS audio_features CASCADE')
            cur.execute(ddl)

            rows = [
                (row['filename'], *[row[c] for c in feature_cols])
                for _, row in df.iterrows()
            ]
            cols = ['file_path'] + feature_cols
            execute_values(
                cur,
                f"INSERT INTO audio_features ({', '.join(cols)}) VALUES %s",
                rows,
                page_size=500,
            )

    print(f'\nInserted {len(rows):,} rows into audio_features\n')


def verify(conn):
    print('── Verification ─────────────────────────────')
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM audio_features')
        count = cur.fetchone()[0]
        print(f'Row count:  {count:,}  (expect 9,980)')

        cur.execute('''
            SELECT COUNT(*) FROM audio_features
            WHERE file_path IS NULL
               OR mfcc1_mean IS NULL
               OR tempo IS NULL
        ''')
        nulls = cur.fetchone()[0]
        print(f'Null check: {nulls} nulls  (expect 0)')

        cur.execute('''
            SELECT ac.file_path, ac.label, ac.split, af.tempo, af.mfcc1_mean
            FROM   audio_clips ac
            JOIN   audio_features af USING (file_path)
            LIMIT  5
        ''')
        rows = cur.fetchall()
        print('\nSpot-check JOIN (audio_clips ⋈ audio_features):')
        print(f'  {"file_path":<30} {"label":<12} {"split":<6} {"tempo":>8} {"mfcc1_mean":>12}')
        for r in rows:
            print(f'  {r[0]:<30} {r[1]:<12} {r[2]:<6} {r[3]:>8.2f} {r[4]:>12.4f}')


def add_foreign_key(conn):
    print('\n── Foreign key ──────────────────────────────')
    with conn:
        with conn.cursor() as cur:
            cur.execute('''
                ALTER TABLE audio_features
                ADD CONSTRAINT fk_audio_features_file_path
                FOREIGN KEY (file_path) REFERENCES audio_clips (file_path)
            ''')
    print('audio_features.file_path → audio_clips.file_path  ✓')


def print_sample_query():
    print('\n── Sample training query ────────────────────')
    print('''
SELECT
    ac.file_path,
    ac.label,
    ac.split,
    af.tempo,
    af.mfcc1_mean,
    af.mfcc1_var,
    -- … all other feature columns …
    af.chroma_stft_mean,
    af.spectral_centroid_mean
FROM  audio_clips  ac
JOIN  audio_features af USING (file_path)
WHERE ac.split = \'train\';   -- swap \'val\' or \'test\' as needed
''')


def main():
    df = load_and_clean_csv()
    conn = get_connection()

    try:
        migrate(df, conn)
        verify(conn)
        add_foreign_key(conn)
        print_sample_query()
    finally:
        conn.close()


if __name__ == '__main__':
    main()
