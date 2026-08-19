# DeepForensics Model Weights Directory

This directory is designed to store trained PyTorch model checkpoints (`.pt` or `.pth` files) for the multidomain inference engine.

## Required Checkpoint Files

| Model Domain | Architecture | Recommended Checkpoint Filename | Domain Flag |
| :--- | :--- | :--- | :--- |
| **RGB Spatial** | EfficientNet-B4 | `efficientnet_b4_rgb.pt` | `spatial_rgb` |
| **Frequency DCT** | ConvNeXt-Tiny | `convnext_dct_freq.pt` | `frequency` |
| **Vision Transformer** | ViT-Base/16 | `vit_b16_spatial.pt` | `spatial_vit` |
| **Noise Residual** | SRM-ResNet18 | `srm_resnet18_residual.pt` | `residual` |
| **Temporal Video** | CNN-LSTM / VideoMAE | `cnn_lstm_temporal.pt` | `temporal` |

## Model Readiness Behavior

- **Without Weights Installed**: DeepForensics strictly operates in `MODEL_NOT_READY` mode. 
  - `GET /api/models` reports `is_active: false` / `NOT_READY`.
  - Detection endpoints return status `MODEL_NOT_READY`, primary prediction `UNCERTAIN`, and confidence `null`.
  - **No fake predictions or fabricated metrics are generated under any circumstances.**

- **With Weights Installed**: Once a valid checkpoint is placed in `model_weights/` and its path configured in `.env` or `backend/core/config.py`:
  - DeepForensics automatically loads model parameters on startup.
  - `GET /api/models` reports `is_active: true` / `READY`.
  - Production inference runs full forward passes across all ready domain branches.

## How to Train & Export Weights

1. Organize training dataset frames with video/identity-aware splits using `ml/datasets/forensic_dataset.py`.
2. Run training script:
   ```bash
   python -m ml.training.trainer --model efficientnet --dataset ffpp --epochs 25 --save-path model_weights/efficientnet_b4_rgb.pt
   ```
3. Verify readiness using PyTest:
   ```bash
   python -m pytest tests/test_ml_models.py
   ```
