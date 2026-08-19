# DeepForensics — Model Training & Dataset Pipeline Guide

This document outlines the procedure for configuring datasets, executing domain model training, fitting probability calibration parameters, and saving model checkpoints.

---

## 1. Dataset Organization

Place raw datasets into `datasets/raw/`:
```
datasets/raw/
├── real/
│   ├── photo_001.jpg
│   └── photo_002.jpg
└── ai/
    ├── sd15_001.jpg
    ├── sdxl_002.jpg
    └── flux_003.jpg
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

---

## 2. Automated Quality Control & Splitting

Run the dataset setup and quality control script:
```bash
python ml/datasets/dataset_setup.py
```

### Quality Control Steps:
1. **Corrupted File Filtering**: Automatically checks PIL decoding and removes broken files.
2. **Resolution Check**: Filters images smaller than $64 \times 64$ pixels.
3. **Duplicate & Leakage Removal**: Computes 64-bit perceptual hashes (pHash) and removes near-duplicates with Hamming distance $\le 4$.
4. **Leakage-Preventing Split**: Generates `train` (60%), `validation` (20%), `test` (20%), and `unseen_generator_test` splits recorded in `datasets/dataset_manifest.json`.

---

## 3. Executing Model Training

Run the master training script:
```bash
python ml/training/train.py
```

This script:
1. Loads dataset splits from `dataset_manifest.json`.
2. Trains `EfficientNetRGBModel`, `ConvNeXtFrequencyModel`, and `SRMResidualModel` independently using AdamW optimizer and CrossEntropyLoss.
3. Evaluates performance on the validation set after each epoch.
4. Fits probability calibration parameters (Temperature & Bias) on validation logit outputs.
5. Saves state dict PyTorch weights and training metadata into `model_weights/`:
   - `model_weights/efficientnet_b4/best.pt`
   - `model_weights/convnext_dct/best.pt`
   - `model_weights/noise_model/best.pt`
   - `model_weights/calibration/calibration.json`
   - `model_weights/training_report.json`
