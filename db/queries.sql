-- Pull each split (usable files only)
SELECT ac.file_path, l.label_name AS label
FROM   audio_clips ac
JOIN   labels l ON l.label_id = ac.label_id
WHERE  ac.split = 'train' AND ac.is_corrupted = FALSE AND ac.is_duplicate = FALSE
ORDER  BY ac.file_path;

SELECT ac.file_path, l.label_name AS label
FROM   audio_clips ac
JOIN   labels l ON l.label_id = ac.label_id
WHERE  ac.split = 'val' AND ac.is_corrupted = FALSE AND ac.is_duplicate = FALSE
ORDER  BY ac.file_path;

SELECT ac.file_path, l.label_name AS label
FROM   audio_clips ac
JOIN   labels l ON l.label_id = ac.label_id
WHERE  ac.split = 'test' AND ac.is_corrupted = FALSE AND ac.is_duplicate = FALSE
ORDER  BY ac.file_path;

-- Song counts per genre x split (verify balance)
SELECT ac.split, l.label_name AS label, COUNT(*) AS songs
FROM   audio_clips ac
JOIN   labels l ON l.label_id = ac.label_id
WHERE  ac.is_corrupted = FALSE AND ac.is_duplicate = FALSE
GROUP  BY ac.split, l.label_name
ORDER  BY ac.split, l.label_name;

-- Top-level summary
SELECT * FROM vw_split_summary;
