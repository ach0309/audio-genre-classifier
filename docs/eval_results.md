# Evaluation Results

**Overall test accuracy:** 0.3400 (34.00%).

**Where the model performs well:** The strongest genres by F1 score are metal (F1=0.537), pop (F1=0.519), hiphop (F1=0.462). These classes likely have more distinctive spectrogram patterns or fewer overlaps with neighboring genres in the test split.

**Where the model struggles:** The weakest genres by F1 score are reggae (F1=0.111), rock (F1=0.133), country (F1=0.148). The largest confusion pairs are: true reggae predicted as disco (11); true rock predicted as metal (9); true pop predicted as disco (8).

**Possible reasons why:** Genre boundaries are subjective, and several GTZAN styles share instrumentation, tempo ranges, and production characteristics. Misclassifications are most likely when songs from different genres produce similar mel-spectrogram textures.

**Loss-curve interpretation:** Both losses remain high or fail to improve, which suggests underfitting or an optimization problem.

**Architecture justification:** A CNN is a good fit because log-mel spectrograms behave like time-frequency images. Convolution layers learn local harmonic and rhythmic patterns, pooling makes the classifier less sensitive to small timing shifts, and the classifier head maps the learned audio features to the 10 genre labels.

**Improvement with more time:** Performance could improve with stronger augmentation, more training data, and tuning regularization or model capacity for the genres with the lowest F1 scores.
