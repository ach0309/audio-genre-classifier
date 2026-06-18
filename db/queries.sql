-- Pull each split
SELECT file_path, label FROM audio_clips WHERE split = 'train' ORDER BY file_path;
SELECT file_path, label FROM audio_clips WHERE split = 'val'   ORDER BY file_path;
SELECT file_path, label FROM audio_clips WHERE split = 'test'  ORDER BY file_path;

-- Clip counts per genre × split (verify balance)
SELECT split, label, COUNT(*) AS clips
FROM   audio_clips
GROUP  BY split, label
ORDER  BY split, label;

-- Top-level summary
SELECT split, COUNT(*) AS clips
FROM   audio_clips
GROUP  BY split
ORDER  BY split;
