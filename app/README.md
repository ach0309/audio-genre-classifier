# Demo app architecture — Genre Guesser

A local web app that lets someone upload (or try a sample) audio clip and get the
trained model's genre prediction interactively — the Sprint deliverable of putting
the model in front of a real person.

The frontend is a three-stage reveal rather than a static form-and-results page:

1. **Idle** — drag a track onto the box (or drag one of the 5 sample chips onto it,
   or use the "choose a file" fallback link — drag has no equivalent gesture on
   touchscreens, so the fallback is what makes this work on mobile). Clicking a
   sample chip instead of dragging it loads *and* guesses immediately, for the
   fastest path through a live demo.
2. **Guessing** — the analyzed clip's mel spectrogram fades in while genre names
   roll past in a slot-reel window, landing on the model's actual top prediction
   (not a canned animation) after a fixed ~2.6s.
3. **Reveal** — the top pick pops in as a "podium" hero, physically elevated above
   the #2/#3 runner-ups which rise in a beat after, alongside the real spectrogram
   and an audio player that autoplays the clip.

## How it connects to the rest of the pipeline

This app does not train or re-implement anything. It's a new, thin consumer of the
same artifacts the notebooks already produce:

```
code/3_model.ipynb                 code/4_predict.ipynb              app/
─────────────────────              ─────────────────────             ─────────────────────
Trains AudioCNN on the       ──►    Loads the same checkpoint   ┐    Loads the same checkpoint
train/val split, saves the          and evaluates it on the     │    at startup and evaluates
canonical checkpoint:                held-out test split from   │    it on whatever audio a
  models/best_model.pth             the database.                │    person uploads (or one of
  {                                                               │    the bundled samples).
    model_state_dict,                                             │
    label_to_idx,                                                 └──► app/inference.py
    idx_to_label,                                                       load_model()
    train_losses, val_losses,                                           analyze_audio()
    val_accuracy,
  }
```

`code/common.py` is the shared code all three of the above import — `AudioCNN`
(the model architecture), `audio_to_mel_spectrogram`, `pad_or_truncate`,
`SAMPLE_RATE`, `FIXED_NUM_SAMPLES`. The app runs an uploaded clip through the exact
same preprocessing steps `4_predict.ipynb` runs the test split through, so a
prediction here is directly comparable to the notebook's evaluation — same
architecture, same weights, same feature extraction, just a different (arbitrary)
input instead of the fixed GTZAN test split.

**Prerequisite:** `models/best_model.pth` must exist, which means `code/3_model.ipynb`
must have been run at least once. The app fails fast at startup with a clear error
if the checkpoint is missing.

## Request flow

1. Browser sends a `POST /predict` with the audio file (from a manual upload or one
   of the 5 bundled sample chips) as multipart form data.
2. `app/main.py` validates the file extension, writes it to a temp file, and calls
   `app/inference.py`'s `analyze_audio(path)`.
3. `analyze_audio`:
   - Loads the audio with `librosa` at the model's training sample rate (mono, 22,050 Hz)
   - Pads/truncates it to the model's fixed 30-second input window (`pad_or_truncate`)
   - Computes the log-mel spectrogram once (`audio_to_mel_spectrogram`)
   - Runs that spectrogram through `AudioCNN` and softmax to get a probability per genre
   - Renders that *same* spectrogram array to a PNG (so the image shown is exactly
     what the model classified, not a separately-computed visualization)
4. `app/main.py` returns JSON: all 10 genre probabilities (sorted, most likely first)
   plus the spectrogram as a base64 PNG data URI.
5. `app/static/index.html` builds the slot-reel dynamically from that response —
   two cycles of all 10 genres plus the real top prediction as the landing row —
   and holds the reveal back until the reel's CSS transition has fully played, so
   the animation is never cut short regardless of how fast inference returns.
6. The reveal renders the top 3 as a podium (the #1 pick physically elevated above
   #2/#3), sets the spectrogram `<img>` to the returned PNG, and autoplays the
   analyzed clip through a second audio player.

## Files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app. `POST /predict` endpoint; serves the frontend at `/`. |
| `app/inference.py` | Loads `models/best_model.pth` once at startup; runs the pipeline above. |
| `app/static/index.html` | Single-page frontend — no build step, inline CSS/JS. |
| `app/static/samples/` | 5 bundled demo clips (one per genre: blues, classical, hiphop, metal, reggae) so a live demo doesn't require hunting for a file. |
| `scripts/generate_sample_clips.py` | One-time dev tool used to produce the files in `app/static/samples/` from `data/genres_original/`. Only needs re-running if you want to swap which clips are bundled. |

## Running it

```bash
uvicorn app.main:app --reload
```
Then open http://localhost:8000.

## A note on honesty

The model's real test accuracy is roughly 30-34% (see `docs/eval_results.md`) — well
above the 10% random baseline for 10 balanced classes, but far from reliable. The UI
is deliberately framed as "best guess" rather than a definitive answer, and shows the
top-3 breakdown rather than a single hard label, so the demo doesn't overstate what
the model can actually do.
