# DeepForensics — Evaluation & Robustness Benchmark Suite

DeepForensics evaluates model performance using standardized forensic verification metrics across clean, unseen, and degraded image media.

---

## 1. Core Evaluation Metrics

| Metric | Definition & Purpose |
| :--- | :--- |
| **Accuracy** | Overall proportion of correct classifications $\frac{TP + TN}{TP + TN + FP + FN}$. |
| **Precision** | Reliability of AI generation predictions $\frac{TP}{TP + FP}$. |
| **Recall (TPR)** | Capability to catch synthetic media $\frac{TP}{TP + FN}$. |
| **F1-Score** | Harmonic mean of precision and recall. |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve. |
| **PR-AUC** | Area under the Precision-Recall curve (critical for imbalanced datasets). |
| **EER (Equal Error Rate)** | Operating point where False Positive Rate equals False Negative Rate ($FPR = FNR$). |

---

## 2. Robustness Testing Suite

Evaluates model resilience under common digital media transformations:

```bash
python ml/evaluation/robustness_test.py
```

### Tested Degradation Modes:
1. **JPEG Compression**: Quality factors 95, 75, and 50.
2. **Resolution Downsampling**: 50% spatial bilinear downscaling followed by upscaling.
3. **Gaussian Blur**: Kernel radius = 1.5.
4. **Gaussian Noise**: Additive zero-mean Gaussian noise ($\sigma = 15$).
5. **Screenshot Distortion**: Contrast boost + brightness reduction to emulate social media screenshotting.

---

## 3. Dataset Leakage Audit

Run the cross-split perceptual duplicate audit:
```bash
python ml/evaluation/leakage_test.py
```

Checks for pHash collisions between `train`, `validation`, and `test` splits to guarantee zero data leakage.
