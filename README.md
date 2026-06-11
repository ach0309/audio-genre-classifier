# team4ward

To get started:

**Clone this repo on your local environment**
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

Place the GTZAN genre folders under the ignored `data/` folder:

```text
data/genres_original/
```

For an interactive notebook view of the genre distribution, audio metadata
sample, and MFCC examples, open `code/1_explore.ipynb`.
