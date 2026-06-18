# PostgreSQL Setup — How the Database Works in This Project

This document explains why this project uses PostgreSQL, how it fits into the pipeline, and how to reproduce the setup locally.

---

## What the database is for

The model trains on raw `.wav` files from `data/genres_original/`. Before training, those 999 songs need to be divided into three folds: train, val, and test. The `audio_clips` table is the single source of truth for those assignments — every contributor queries the same table instead of computing splits independently and risking inconsistency.

The CSVs (`features_30_sec.csv`, `features_3_sec.csv`) are used only for EDA in `1_explore.ipynb`. They are not part of the training pipeline.

---

## How PostgreSQL works locally

PostgreSQL runs as a background server process on each contributor's machine. Two separate programs talk to it:

```
┌─────────────────────────────────┐
│  Local machine                  │
│                                 │
│  postgres server (background)   │  ← stores and retrieves data
│       ↕  (TCP port 5432)        │
│  psql (terminal client)         │  ← interactive SQL from the terminal
│  Python script (psycopg2)       │  ← notebook and scripts connect here
└─────────────────────────────────┘
```

- **`postgres` server** — the database engine. Runs in the background; owns the data files.
- **`psql`** — a terminal program that connects to the server and lets contributors run SQL interactively. Similar to Python's interactive shell, but for the database.
- **`psycopg2`** — a Python library that lets scripts connect to Postgres and get results back as Python objects:

```
Python script
    ↓  (uses psycopg2)
psycopg2 translates Python into database calls
    ↓
Postgres server
    ↓
results returned as Python objects (lists, dicts, etc.)
```

---

## Schema

One table, three columns:

```sql
CREATE TABLE audio_clips (
    id        SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,  -- relative path to wav file, e.g. data/genres_original/blues/blues.00000.wav
    label     TEXT NOT NULL,         -- genre: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
    split     TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
);
```

One row per 30-second wav file. `jazz.00054.wav` is excluded — it is a corrupt file that fails to load.

---

## Split assignment

`db/populate.py` scans `data/genres_original/`, excludes `jazz.00054.wav`, shuffles the remaining 999 songs with a fixed seed (`random.seed(42)`), and assigns:

- **70%** → train (699 songs)
- **15%** → val  (149 songs)
- **15%** → test (151 songs)

The fixed seed guarantees that every contributor who runs `populate.py` gets identical splits.

---

## Reproducing the setup

**Prerequisites:** PostgreSQL 14+ installed and running locally.

```bash
# 1. Create the database
createdb audio_genre_classifier

# 2. Create the table
psql audio_genre_classifier < db/schema.sql

# 3. Copy the env template and fill in local credentials
cp .env.example .env

# 4. Populate split assignments (requires data/genres_original/)
python db/populate.py

# 5. Verify
psql audio_genre_classifier < db/queries.sql
```

Expected output from step 5:
```
 split | songs
-------+-------
 test  |   151
 train |   699
 val   |   149
```

---

## How to inspect the data

The table lives in Postgres's internal binary storage (typically `/opt/homebrew/var/postgresql@14/` on Mac) — it cannot be opened as a file. Use `psql` to inspect it:

```bash
psql audio_genre_classifier
```

```sql
\d audio_clips                        -- table structure
SELECT * FROM audio_clips LIMIT 10;   -- sample rows
\q                                    -- exit
```

Or as a one-liner:

```bash
psql audio_genre_classifier -c "SELECT * FROM audio_clips LIMIT 10;"
```

---

## Each contributor runs their own local database

The database is not shared — each contributor runs the same scripts locally. Because the seed is fixed, the splits are identical across all machines.

```
Contributor A                 Contributor B
┌──────────────┐              ┌──────────────┐
│  postgres    │              │  postgres    │
│  audio_genre │  identical   │  audio_genre │
│  (local)     │   splits     │  (local)     │
└──────────────┘              └──────────────┘
        ↑ both run the same schema.sql + populate.py from the repo
```
