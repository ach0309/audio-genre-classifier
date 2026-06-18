CREATE TABLE IF NOT EXISTS audio_clips (
    id        SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    label     TEXT NOT NULL,
    split     TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
);

CREATE INDEX IF NOT EXISTS idx_audio_clips_split       ON audio_clips (split);
CREATE INDEX IF NOT EXISTS idx_audio_clips_label_split ON audio_clips (label, split);
