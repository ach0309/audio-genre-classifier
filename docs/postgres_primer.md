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
- **`SQLAlchemy` + `psycopg2`**: the Python stack notebooks use. SQLAlchemy builds the SQL; psycopg2 sends it to the server. Results come back as pandas DataFrames via `pd.read_sql()`.

```
Python script
    ↓  (uses SQLAlchemy + psycopg2)
SQLAlchemy builds the query; psycopg2 sends it to the server
    ↓
Postgres server
    ↓
results returned as pandas DataFrames
```

---

## The tables

Two tables and a view:

```sql
CREATE TABLE labels (
    label_id     SERIAL PRIMARY KEY,
    label_name   TEXT NOT NULL UNIQUE,  -- e.g. 'blues'
    sample_count INT  NOT NULL          -- usable songs for that genre
);

CREATE TABLE audio_clips (
    id           SERIAL  PRIMARY KEY,
    file_path    TEXT    NOT NULL UNIQUE,  -- e.g. data/genres_original/blues/blues.00000.wav
    label_id     INT     NOT NULL REFERENCES labels(label_id),
    split        TEXT    NOT NULL CHECK (split IN ('train', 'val', 'test')),
    is_corrupted BOOLEAN NOT NULL DEFAULT FALSE,
    is_duplicate BOOLEAN NOT NULL DEFAULT FALSE
);
```

All 1,000 songs are in `audio_clips`. `jazz.00054.wav` is included but marked `is_corrupted = TRUE` — every query filters it out with `WHERE is_corrupted = FALSE`. `vw_split_summary` is a view that returns usable song counts per split.

---

## How splits are assigned

`db/populate.py` scans `data/genres_original/`, shuffles all 1,000 songs with `random.seed(42)`, and splits them 70/15/15. `jazz.00054.wav` is included but inserted with `is_corrupted = TRUE` — every query filters it out, so it never reaches the model. The fixed seed means anyone who runs `populate.py` gets the same splits.

Usable songs per split (approximate — exact counts depend on where the corrupt file lands in the shuffle):

- train: ~700 songs
- val:   ~150 songs
- test:  ~150 songs

---

## Password

On Mac with a Homebrew Postgres install, no password is needed. Homebrew sets up **trust authentication** for local connections — it lets you in based on your Mac username, no password required. That is why `.env.example` has `DB_PASSWORD=` left blank.

Your `DB_USER` should be your Mac username (run `whoami` in the terminal if unsure).

If you get an authentication error, run:

```bash
psql audio_genre_classifier -c "SHOW hba_file;"
```

Open the file that command prints and look for a line like:

```
local   all   all   trust
```

`trust` means no password is needed. If it says `md5` or `scram-sha-256`, a password was set during installation — fill in `DB_PASSWORD` in your `.env` with that password.

---

## Setup steps

PostgreSQL 14+ needs to be installed and running locally first.

```bash
# create the database
createdb audio_genre_classifier

# create the tables and view
psql audio_genre_classifier < db/schema.sql

# copy the env template and fill in your username
cp .env.example .env

# populate the tables (requires data/genres_original/)
python db/populate.py

# verify
psql audio_genre_classifier < db/queries.sql
```

Expected output from the last step (usable songs only):

```
 split | total_files
-------+-------------
 test  |         ~150
 train |         ~700
 val   |         ~150
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
