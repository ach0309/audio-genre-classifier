# PostgreSQL Setup

New to Postgres? Read this before running the setup steps in the README.

---

## What the database is for

The model trains on raw `.wav` files from `data/genres_original/`. Those 999 songs need to be divided into train, val, and test before training. Split assignments are stored in the `audio_clips` table so everyone on the team gets the same splits by running the same script, rather than computing them independently and getting different results.

The CSVs (`features_30_sec.csv`, `features_3_sec.csv`) are used only for EDA in `1_explore.ipynb` and are not part of the training pipeline.

---

## How Postgres works locally

Postgres runs as a background server on your machine. Two separate programs connect to it:

```
┌─────────────────────────────────┐
│  Your machine                   │
│                                 │
│  postgres server (background)   │  ← stores the data
│       ↕  (TCP port 5432)        │
│  psql (terminal client)         │  ← run SQL from the terminal
│  Python script (psycopg2)       │  ← notebooks and scripts connect here
└─────────────────────────────────┘
```

- **`postgres` server**: the database engine running in the background. Start it once and it keeps running.
- **`psql`**: a terminal program that connects to the server so you can run SQL interactively. Think of it like the Python interactive shell, but for your database.
- **`psycopg2`**: a Python library that lets scripts talk to Postgres and get results back as Python objects:

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

## The table

One table, three columns:

```sql
CREATE TABLE audio_clips (
    id        SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,  -- e.g. data/genres_original/blues/blues.00000.wav
    label     TEXT NOT NULL,         -- genre name
    split     TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
);
```

One row per 30-second wav file. `jazz.00054.wav` is left out because it is corrupt and fails to load.

---

## How splits are assigned

`db/populate.py` scans `data/genres_original/`, skips `jazz.00054.wav`, shuffles the remaining 999 songs with `random.seed(42)`, and splits them:

- train: 699 songs (70%)
- val: 149 songs (15%)
- test: 151 songs (15%)

The fixed seed means anyone who runs `populate.py` gets the same splits.

---

## Setup steps

PostgreSQL 14+ needs to be installed and running locally first.

```bash
# create the database
createdb audio_genre_classifier

# create the table
psql audio_genre_classifier < db/schema.sql

# copy the env template and fill in your local credentials
cp .env.example .env

# populate the table (requires data/genres_original/)
python db/populate.py

# verify
psql audio_genre_classifier < db/queries.sql
```

Expected output from the last step:

```
 split | songs
-------+-------
 test  |   151
 train |   699
 val   |   149
```

---

## How to inspect the data

The table lives in Postgres's internal binary storage (around `/opt/homebrew/var/postgresql@14/` on Mac). There is no file you can open directly. Use `psql` to look at it:

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

## Teammates

The database is local to each person's machine. Everyone runs the same `schema.sql` and `populate.py` from the repo and ends up with the same table.

```
Your machine              Teammate's machine
┌──────────────┐          ┌──────────────┐
│  postgres    │          │  postgres    │
│  audio_genre │  same    │  audio_genre │
│  (local)     │  splits  │  (local)     │
└──────────────┘          └──────────────┘
```
