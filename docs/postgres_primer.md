# PostgreSQL Primer — What's Going On Here

This doc explains the database setup for the audio genre classifier, written for someone who hasn't used PostgreSQL locally before.

---

## What PostgreSQL is

PostgreSQL (Postgres) is a **relational database** that runs as a server process on your machine. Instead of storing data in a CSV file you open in pandas, you store it in structured **tables** with typed columns. You talk to it using SQL.

Think of it like this:

```
CSV file          →   a spreadsheet anyone can open
PostgreSQL table  →   same data, but locked behind a server that enforces rules,
                      handles multiple readers, and answers queries fast
```

---

## The moving parts (locally)

```
┌─────────────────────────────────┐
│  Your Mac                       │
│                                 │
│  postgres server (background)   │  ← always running, owns the data files
│       ↕  (TCP port 5432)        │
│  psql (terminal client)         │  ← you type SQL here
│  Python script (psycopg2)       │  ← your code talks to the same server
└─────────────────────────────────┘
```

- **`postgres` server** — a background process that actually stores and retrieves data. You start it once; it keeps running.
- **`psql`** — a terminal program that lets you type SQL and see results. Think of it as the "REPL" for Postgres.
- **`psycopg2`** — a Python library that lets your scripts send SQL to the server the same way `psql` does.

---

## The three SQL verbs you'll use

| Verb | What it does |
|---|---|
| `CREATE TABLE` | defines the shape of a table (column names + types) |
| `INSERT INTO` | adds rows |
| `SELECT` | reads rows back out |

---

## What the setup does, step by step

```
1. createdb audio_genre_classifier
   └─ tells the server to create a new empty database

2. psql ... < db/schema.sql
   └─ runs our CREATE TABLE statement — creates the audio_clips table

3. python db/populate.py
   └─ reads data/features_3_sec.csv
      assigns train/val/test to each song (at the parent-song level)
      INSERTs 9,980 rows into audio_clips

4. psql ... < db/queries.sql
   └─ SELECTs rows back out, filtered by split
      your notebook and model code do the same thing
```

---

## Why bother vs. just using the CSV?

For a class project a CSV would work fine. Using Postgres here gives you:

1. **Enforced splits** — the `split` column can only be `'train'`, `'val'`, or `'test'` (a CHECK constraint rejects anything else). No accidental typos.
2. **Single source of truth** — your notebook, your model script, and your teammates all query the same DB instead of each loading their own copy of the CSV and applying splits differently.
3. **Realistic practice** — most real ML pipelines store metadata in a database and pull features on demand.

---

## Schema at a glance

```sql
CREATE TABLE audio_clips (
    id        SERIAL PRIMARY KEY,   -- auto-incrementing row ID
    file_path TEXT NOT NULL UNIQUE, -- clip filename, e.g. blues.00000.0
    label     TEXT NOT NULL,        -- genre: blues, jazz, rock, ...
    split     TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
);
```

One row per 3-second clip. `jazz.00054` and all its clips are excluded before insertion.

---

## Split assignment logic

Splits are assigned at the **parent-song level**, not the clip level.

Why it matters: `blues.00000.0` through `blues.00000.9` are all 3-second windows cut from the same 30-second recording. If clip `.0` goes to train and clip `.5` goes to test, the model is essentially memorizing the same song — that's **data leakage**. Assigning all clips from a song to the same fold prevents this.

```
blues.00000  →  assigned 'train'
  blues.00000.0  →  train
  blues.00000.1  →  train
  ...
  blues.00000.9  →  train

blues.00001  →  assigned 'val'
  blues.00001.0  →  val
  ...
```

Ratio: **70% train / 15% val / 15% test** across 999 parent songs (1,000 minus `jazz.00054`).

---

## How teammates access the database

The database is **local to each person's machine** — teammates can't connect to yours directly. Instead, everyone runs the same setup:

1. Install Postgres locally
2. Run `db/schema.sql` to create the table
3. Run `db/populate.py` to populate it

Because `populate.py` uses a **fixed random seed** (`random_state=42`), everyone generates the exact same splits from the same CSV. The DB isn't shared state — it's a reproducible artifact that any teammate can recreate in minutes.

```
Your Mac                  Teammate's Mac
┌──────────────┐          ┌──────────────┐
│  postgres    │          │  postgres    │
│  audio_genre │  same    │  audio_genre │
│  (local)     │  splits  │  (local)     │
└──────────────┘          └──────────────┘
     ↑ same schema.sql + populate.py from git
```
