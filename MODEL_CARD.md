# DeepForensics — Model Card & Ethical Usage Statement

## Model Overview
- **Model Name**: DeepForensics Multidomain Media Classifier
- **Model Version**: 1.0.0
- **Model Types**: EfficientNet-B4 (RGB Spatial), ConvNeXt-Tiny (2D DCT Frequency), SRM-ResNet18 (Noise Residual)
- **Primary Task**: Binary classification of images and videos into `REAL`, `AI_GENERATED`, or `UNCERTAIN`.

---

## Intended Use & Applications
- **Intended Uses**: Digital forensics, media verification, newsroom content verification, platform safety, research benchmarks.
- **Out-of-Scope Uses**: Automated legal evidence without expert human verification, surveillance, or harassment.

---

## Limitations & Ethical Considerations
1. **Unseen Generative Architectures**: Models may show degraded performance when encountering novel generative paradigms not included in training (e.g., futuristic diffusion samplers).
2. **Aggressive Media Compression**: Heavy re-compression (e.g. repeated WhatsApp/Telegram re-encodes) can attenuate high-frequency forensic cues.
3. **Calibrated Thresholds**: The system employs an explicit `UNCERTAIN` zone ($0.35 < P(\text{AI}) < 0.65$) to avoid over-confident false accusations when evidence is ambiguous.
