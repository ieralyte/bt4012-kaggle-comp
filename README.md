# BT4012 Kaggle Competition 2026: Illicit Bitcoin Transaction Detection

Predicts the probability that each Bitcoin transaction in the test period (time steps 36–49) is illicit. Submissions are scored by ROC AUC.

**Final model (v16):** a 50/50 blend of our earlier v13 ensemble (its saved predictions, the *anchor*) and the mean of three LightGBM models (seeds 2026, 17 and 73). Each of the three models uses 6,000 trees and is trained with masked-feature augmentation: two extra copies of the training rows with 75% of the 165 anonymous feature values hidden, and 8× weight on illicit rows. Its mean ROC AUC on v16's six rolling-origin temporal validation windows is **0.9511**.

**Why v16:** v12 → v13 → v16 each improved on its predecessor in validation (see *Model history* below). A variant of v16 with two extra typicality features scored higher in validation (0.9575 on the same windows) but lower on the public leaderboard, so v16 is the final submission. The report compares all the models we tried.

## Repository contents

| Path | Purpose |
|---|---|
| `e1155885.ipynb` | Final notebook. It validates the inputs, documents preprocessing, feature engineering and the model recipe, recomputes the temporal validation AUCs, and writes `submissions/e1155885.csv`. |
| `fraud_pipeline/` | Helper code used by the notebook: data loading and checks, masked augmentation, and the model refit. |
| `model_assets/lgbm_masked_ensemble/` | The three fitted LightGBM models, the fixed v13 anchor predictions, the saved validation predictions, `release.json` with SHA-256 hashes of every asset, and the originally submitted `submission.csv`. |
| `submissions/e1155885.csv` | The final submission, identical to `model_assets/lgbm_masked_ensemble/submission.csv`. Re-running the notebook rewrites it with the same probabilities. |
| `requirements.txt` | Pinned library versions for the final notebook. |

The competition data is **not** committed. `train.csv` is about 460 MB, over GitHub's 100 MB file limit, and Kaggle competition data should not be redistributed.

## How to reproduce

1. **Get the data.** Download it from the Kaggle competition page, or use the Kaggle CLI:
   ```bash
   kaggle competitions download -c bt-4012-competition-2026
   unzip bt-4012-competition-2026.zip -d bt-4012-competition-2026
   ```
   The notebook checks the four CSV files against the SHA-256 hashes in `release.json`.
2. **Install dependencies.** Use Python 3.10–3.12; numpy 2.0.2 has no wheels for newer versions.
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   On macOS, LightGBM needs the OpenMP runtime. If `import lightgbm` fails, run `brew install libomp`. Use the pinned pandas version: the notebook compares saved medians with pandas 2 dtypes, and pandas 3 fails that check.
3. **Run the notebook from the repository root**, either in Jupyter with *Run All* or from the command line:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace e1155885.ipynb
   ```
   By default it loads the three saved models and writes `submissions/e1155885.csv` in under a minute (about 25 s of compute on 2 CPU cores). Set `REFIT_CORE = True` in the setup cell to retrain the three LightGBM models from the raw data instead; this takes much longer. The v13 anchor is a fixed prediction file. v13's six fitted models were lost, so it cannot be refitted from this repository.

## Method summary

1. **Preprocessing.** Use the 165 supplied features plus `time_step`. Labels recorded as `none` or `unknown` count as missing, and only rows labelled 0 or 1 are used for fitting. Missing feature values are filled with the fitting rows' medians; the supplied data has none. Features that drift with time are kept, because removing them did not help the masked models.
2. **Masked augmentation.** Each training set is tripled: the original rows keep 25% of each row's weight, and two copies take 37.5% each. In each copy, 75% of the anonymous feature values are hidden at random; `time_step` is never hidden. Illicit rows get 8× weight.
3. **Model.** LightGBM with 6,000 trees at learning rate 0.01, 31 leaves, at least 60 rows per leaf, L1 0.2, L2 5, and feature and bagging fractions 0.85. Three seeds are averaged.
4. **Blend.** The final probability is 0.5 × v13 anchor + 0.5 × the mean of the three models.
5. **Validation.** Rolling-origin temporal windows: train on steps ≤ c and score steps c+1 … min(c+14, 35). The notebook recomputes the AUCs from the saved predictions.

## Model history

| Version | Recipe | Mean AUC, v16 dev 6 |
|---|---|---|
| v12 | Earlier submission: our v11 blended with new masked LightGBM models. Only its validation AUCs survive. | 0.9473 |
| v13 | 25% v12 + 37.5% *emphasized75* + 37.5% *positive16*. Each family averages three masked LightGBM models (3,000 trees, two 75%-masked copies); emphasized75 uses 8× illicit weight and 25% of each row's weight on the original rows, positive16 uses 16× and 50%. | 0.9503 |
| **v16** | **50% v13 + 50% the mean of three new masked LightGBM models (6,000 trees, 8× illicit weight, 25% on the original rows).** | **0.9511** |

Expanded, v16 is 12.5% v12 + 18.75% emphasized75 + 18.75% positive16 + 50% new core. v16 does not retrain v13; it blends v13's saved predictions.
