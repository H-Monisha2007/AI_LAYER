# DeepForensics Media Datasets

Directory Structure:
- `raw/real/`, `raw/ai/`: Place raw DSLR/phone photos and generated images here.
- `train/`: Balanced, non-overlapping training split.
- `validation/`: Validation split for hyperparameter tuning & probability calibration.
- `test/`: Evaluation test benchmark.
- `unseen_generator_test/`: Test set containing unseen generative models (e.g. FLUX, Midjourney v6).
