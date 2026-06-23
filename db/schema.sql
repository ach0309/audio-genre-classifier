DROP TABLE IF EXISTS audio_clips CASCADE;
DROP TABLE IF EXISTS labels CASCADE;

CREATE TABLE labels (
    label_id     SERIAL PRIMARY KEY,
    label_name   TEXT   NOT NULL UNIQUE,
    sample_count INT    NOT NULL DEFAULT 0
);

CREATE TABLE audio_clips (
    id           SERIAL  PRIMARY KEY,
    file_path    TEXT    NOT NULL UNIQUE,
    label_id     INT     NOT NULL REFERENCES labels(label_id),
    split        TEXT    NOT NULL CHECK (split IN ('train', 'val', 'test')),
    is_corrupted BOOLEAN NOT NULL DEFAULT FALSE,
    is_duplicate BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_audio_clips_split       ON audio_clips (split);
CREATE INDEX IF NOT EXISTS idx_audio_clips_label_split ON audio_clips (label_id, split);

CREATE OR REPLACE VIEW vw_split_summary AS
SELECT
    split,
    COUNT(*) AS total_files
FROM  audio_clips
WHERE is_corrupted = FALSE
  AND is_duplicate = FALSE
GROUP BY split
ORDER BY split;
