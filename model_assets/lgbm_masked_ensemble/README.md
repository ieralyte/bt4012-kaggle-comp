# Retained prediction assets

This bundle supports `e1155885.ipynb` on a fresh repository checkout. `release.json` contains SHA-256 hashes for all required asset files, hashes for the external competition inputs, the model recipe, the user-reported public score, and eight historical validation receipts.

The three models were fit on labeled periods through 35, using seeds 2026, 17, and 73. Final prediction is 50% fixed v13 anchor plus 50% mean core probability. The fixed anchor and original released submission are included as CSV files. Historical prediction caches and reference CSVs allow ROC-AUC reconstruction against the supplied training labels; they do not refit historical estimators.

Models contain LightGBM estimators with fitting-only medians and feature lists. They are the original local model files, copied without modification. The original full study and provenance archive remain locally under `artifacts/` and are not needed by the notebook.
