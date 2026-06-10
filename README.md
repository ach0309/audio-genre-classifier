# team4ward

To get started:

**clone this repo on your local environment**
```bash
git clone git@github.com:ach0309/audio-genre-classifier.git
```

**Make sure you’re inside your local copy of the repository, then create your own branch (you can rename it later)**
```bash
git checkout -b <first-name>-<what-youre-working-on> (ex: aeon-eda or noah-spectogram or mike-bivariate)
```

## GTZAN dataset setup

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and stage the GTZAN audio files under the ignored `data/` folder:

```bash
python code/setup_gtzan_dataset.py
```

If you already downloaded GTZAN, point the setup script at the local folder:

```bash
python code/setup_gtzan_dataset.py --source /path/to/gtzan
```

Run the first EDA pass:

```bash
python code/genre_distribution_eda.py
```

The default EDA run validates the expected genre folders and saves a class
distribution chart to `reports/genre_distribution.png`. To load audio files
with `librosa` for duration and sample-rate checks, run:

```bash
python code/genre_distribution_eda.py --audio-summary
```

Plot one MFCC example per genre:

```bash
python code/mfcc_examples_eda.py
```

The MFCC chart is saved to `reports/mfcc_examples_by_genre.png`.
