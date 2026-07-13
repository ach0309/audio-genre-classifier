# Audio Genre Classifier

An end-to-end machine learning project that classifies music into 10 genres from raw audio. Built by **team4ward** using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), PostgreSQL, librosa, PyTorch, and scikit-learn.

The core pipeline is complete: the project explores and validates the dataset, creates reproducible train/validation/test splits, transforms waveforms into log-mel spectrograms, trains a convolutional neural network, and evaluates the best checkpoint on a held-out test set. A local, user-facing demo (`app/`) puts that pipeline in front of a real person — upload a track and see the model's guess — and is still being iterated on.

## Project overview

Music genre classification sits at an interesting intersection of signal processing and machine learning. A genre is not a single measurable feature: it emerges from rhythm, instrumentation, timbre, production, and cultural context. Our goal was to build a reproducible pipeline that learns useful patterns from audio while remaining honest about the ambiguity of the task and the limitations of a small benchmark dataset.

The completed workflow is:

```text
GTZAN audio
    ↓
data validation and exploratory analysis
    ↓
reproducible PostgreSQL split assignments
    ↓
fixed-length waveforms and log-mel spectrograms
    ↓
training-only augmentation
    ↓
three-block convolutional neural network
    ↓
validation, checkpoint selection, and early stopping
    ↓
held-out test evaluation and single-file inference
```

## Results

The final model reached **34.67% best validation accuracy** and **30.00% test accuracy** across 10 classes. This is above the 10% random-choice baseline, but it also shows that the current CNN is an early baseline rather than a production-ready genre classifier.

| Test-set finding | Result |
|---|---|
| Overall accuracy | 30.00% |
| Strongest F1 scores | metal: 0.537, pop: 0.480, blues: 0.320 |
| Weakest F1 scores | classical: 0.000, rock: 0.129, country: 0.148 |
| Largest confusion | reggae predicted as disco: 11 samples |

Training stopped after four epochs when validation accuracy did not improve for three consecutive epochs. Training loss continued to fall while validation loss rose after the first epoch, indicating limited generalization. Likely contributors include the small dataset, overlapping genre characteristics, GTZAN's known quality issues, and limited hyperparameter tuning.

These results are useful in their own right: the project demonstrates a complete, leakage-aware evaluation process and gives the team a concrete baseline for future experimentation. See the full [evaluation summary](docs/eval_results.md), [training curves](docs/images/training_curves.png), and [confusion matrix](docs/images/confusion_matrix.png).

## What we built

### Data exploration and quality checks

The exploratory notebook examines class balance, clip duration, sample rate, waveforms, mel spectrograms, and MFCCs. It confirmed that the dataset is balanced at 100 songs per genre and identified `jazz.00054.wav` as corrupt. The file remains represented in the database for traceability but is excluded from all modeling.

The analysis also found that `features_3_sec.csv` contains 9,990 rows rather than 10,000 because 10 shorter songs produce only nine complete segments. To prevent leakage between closely related excerpts, all splits are assigned at the parent-song level.

### Reproducible data management

PostgreSQL acts as the source of truth for labels, file paths, data-quality flags, and split membership. `db/populate.py` scans the local audio collection and applies a fixed random seed to create a reproducible 70/15/15 train/validation/test split. The preprocessing and evaluation notebooks query those assignments instead of creating their own splits.

The schema and relationship are documented in the [database ERD](docs/erd.md).

### Audio preprocessing

Each audio file is loaded at 22,050 Hz, padded or truncated to a consistent length, and converted into a 128-band log-mel spectrogram. The resulting model input has shape `[batch, 1, 128, 1292]`.

Augmentation is restricted to the training split and includes gain adjustment, time shifting, additive noise, and frequency masking. Validation and test samples remain unchanged so their metrics are comparable and repeatable.

### CNN training

`AudioCNN` treats each spectrogram as a single-channel image. Its architecture contains:

- Three `Conv2d → BatchNorm2d → ReLU → MaxPool2d` blocks with 32, 64, and 128 channels
- Adaptive average pooling to a fixed `4 × 4` representation
- A 256-unit fully connected layer with 50% dropout
- A 10-logit output layer, one value per genre

The model trains with cross-entropy loss and Adam at a learning rate of `1e-3`. Training runs for up to 10 epochs with early stopping after three epochs without improved validation accuracy. The strongest checkpoint stores the model weights, label mappings, validation score, and training history.

### Evaluation and inference

The final notebook loads the saved checkpoint and evaluates it only on the held-out test split. It reports accuracy, per-genre precision/recall/F1, a confusion matrix, strongest and weakest genres, and the most common confusion pairs. It also contains the inference path used to turn an audio sample into a predicted genre label.

## Project structure

```text
audio-genre-classifier/
├── app/
│   ├── main.py                # FastAPI backend (POST /predict, serves the frontend)
│   ├── inference.py           # Loads models/best_model.pth, runs code/common.py's pipeline
│   ├── README.md              # Demo app architecture, how it connects to the notebooks
│   └── static/
│       ├── index.html         # Single-page frontend (upload, samples, results, spectrogram)
│       └── samples/           # Bundled demo clips, one per genre
├── code/
│   ├── 1_explore.ipynb       # Dataset exploration and integrity checks
│   ├── 2_transform.ipynb     # Preprocessing, augmentation, and DataLoaders
│   ├── 3_model.ipynb         # CNN training, validation, and checkpointing
│   ├── 4_predict.ipynb       # Held-out evaluation and inference
│   └── common.py             # Shared preprocessing, dataset, DB, and model code
├── data/
│   ├── genres_original/      # Raw GTZAN audio (downloaded locally; not committed)
│   ├── features_30_sec.csv   # One pre-extracted feature row per song
│   └── features_3_sec.csv    # Pre-extracted features for three-second segments
├── db/
│   ├── schema.sql            # Tables, indexes, constraints, and summary view
│   ├── populate.py           # File scan and reproducible split population
│   └── queries.sql           # Split and verification queries
├── docs/
│   ├── eval_results.md       # Final model evaluation summary
│   ├── erd.md                # Database design and relationship documentation
│   ├── postgres_primer.md    # PostgreSQL guide for new contributors
│   ├── requirements.txt      # Python dependencies
│   ├── *research.md          # Dataset and audio-feature research
│   └── images/               # EDA, training, evaluation, and ERD figures
├── models/
│   └── best_model.pth        # Generated locally by 3_model.ipynb; not committed
└── scripts/
    └── generate_sample_clips.py  # One-time tool that built app/static/samples/
```

## Dataset

[GTZAN](http://marsyas.info/downloads/datasets.html) is a widely used music information retrieval benchmark introduced by George Tzanetakis and Perry Cook. This project uses the Kaggle distribution linked above.

| Property | Value |
|---|---|
| Genres | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| Songs | 1,000 total; 100 per genre |
| Duration | approximately 30 seconds per song |
| Audio | mono `.wav`, 22,050 Hz |
| Usable files | 999 |
| Excluded file | `jazz.00054.wav` (corrupt) |

GTZAN is useful for benchmarking, but it is not a clean representation of the full musical world. Published research has identified repetitions, mislabels, distortions, and artist repetition in the dataset. Genre labels themselves are subjective and often overlap. We therefore treat the final metrics as a baseline and avoid presenting the model as a definitive judge of genre.

## Run the project

### 1. Clone and install

```bash
git clone git@github.com:ach0309/audio-genre-classifier.git
cd audio-genre-classifier
python -m pip install -r docs/requirements.txt
```

PostgreSQL 14+ must also be installed and running locally.

### 2. Download GTZAN

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place `genres_original/` inside `data/`:

```text
data/genres_original/
├── blues/
├── classical/
├── country/
└── ...
```

### 3. Configure PostgreSQL

```bash
createdb audio_genre_classifier
psql audio_genre_classifier < db/schema.sql
cp .env.example .env
```

Set `DB_USER` in `.env` to your PostgreSQL username, then populate and verify the database:

```bash
python db/populate.py
psql audio_genre_classifier < db/queries.sql
```

See the [PostgreSQL primer](docs/postgres_primer.md) for additional guidance.

### 4. Run the notebooks

Run the notebooks in numerical order. Restart the Jupyter kernel before running them if `code/common.py` has changed.

| Notebook | Purpose | Produces |
|---|---|---|
| `1_explore.ipynb` | Explore and validate GTZAN | EDA figures in `docs/images/` |
| `2_transform.ipynb` | Build the preprocessing and DataLoader pipeline | Verified train/val/test batches |
| `3_model.ipynb` | Train and select the CNN | `models/best_model.pth`, training curves |
| `4_predict.ipynb` | Evaluate and run inference | Metrics, confusion matrix, evaluation summary |

`common.py` must remain beside the notebooks because the final three notebooks import it directly.

### 5. Run the demo app

Requires `models/best_model.pth` to already exist (step 4 above).

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000, drag a track onto the box (or drag/click one of the bundled samples, or use the "choose a file" link), and watch it guess.

## Demo app

**Genre Guesser** — a small local web app that puts the trained model in front of a real person: drag a track onto the box (or drop in one of five bundled sample clips), and watch a game-show-style reveal — the clip's mel spectrogram fades in while genre names roll past in a slot reel, landing on the model's actual top prediction, then the top 3 land as a podium with the winner physically elevated above the #2/#3 runner-ups. The analyzed spectrogram and an autoplaying audio player round out the reveal. See [app/README.md](app/README.md) for how it connects to the rest of the pipeline — the same `models/best_model.pth` checkpoint and `code/common.py` preprocessing/architecture that `3_model.ipynb` and `4_predict.ipynb` use, just run against arbitrary uploaded audio instead of the fixed GTZAN test split.

Upload works by drag-and-drop (including dragging the sample chips themselves) or via an explicit "choose a file" link — the latter is what makes this work on mobile, since there's no drag gesture for files on a touchscreen.

## Project status and next steps

The research, data pipeline, model training, and evaluation phases are complete. The demo (`app/`) is built and working end to end, including the game-show reveal flow and mobile-friendly upload.

With more time, the strongest technical next steps would be:

- Tune learning rate, regularization, batch size, and model capacity
- Train on more diverse and carefully curated audio
- Compare the CNN with transfer learning from a pretrained audio model
- Use stratified, artist-aware splitting where metadata permits
- Add confidence scores and clearer uncertainty handling to the demo

The project’s main outcome is not only a predicted label. It is a complete, reproducible path from imperfect raw audio to an honestly evaluated machine learning system—and a clear foundation for the demo and future model improvements.
