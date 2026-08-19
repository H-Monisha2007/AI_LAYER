# DeepForensics — Multidomain AI-Generated Image & Video Detection System

DeepForensics is an end-to-end, scientifically validated deep-learning framework for detecting synthetic, AI-generated, and manipulated media (deepfakes, diffusion artifacts, face swaps, synthetic generation).

> **Scientific Honesty Guarantee**: DeepForensics strictly enforces true model inference. When trained checkpoints are unavailable, the system reports `MODEL_NOT_READY` with `UNCERTAIN` classification. No simulated, random, or hardcoded scores are ever generated.

---

## 🌟 Key Features

1. **Multidomain Forensic Analysis**:
   - **RGB Spatial Domain**: EfficientNet-B4 spatial artifact classifier.
   - **Frequency Domain**: ConvNeXt-Tiny with 2D Discrete Cosine Transform (DCT) log-magnitude spectrum analysis to catch high-frequency diffusion checkerboards.
   - **Noise Residual Domain**: SRM (Stegananalytic Rich Model) high-pass spatial filter bank + ResNet-18.
   - **Face Analysis Domain**: OpenCV Haar Cascade face region detection, cropping, and landmark alignment.
2. **Dynamic Score Fusion & Calibration**:
   - Dynamic weight re-normalization across active domains.
   - Temperature scaling & Platt logit calibration to ensure probabilities reflect true posterior confidence P(AI_GENERATED).
   - Calibrated decision thresholds (`REAL` <= 0.35, `AI_GENERATED` >= 0.65, `UNCERTAIN` between thresholds).
3. **Temporal Video Detection**:
   - Uniform frame extraction and per-frame multidomain scoring.
   - Temporal consistency analysis measuring frame-to-frame variance & deepfake flicker.
   - Suspicious frame ranking with anomaly scores.
4. **Visual Explainability**:
   - Integrated Grad-CAM gradient-weighted activation heatmaps highlighting suspicious spatial regions.
5. **Rigorous Quality Control & Evaluation Suite**:
   - Perceptual hash (pHash) duplicate removal & dataset leakage prevention.
   - Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Equal Error Rate (EER), Confusion Matrix.
   - Robustness benchmarks under JPEG compression (quality 95/75/50), downsampling, Gaussian blur, Gaussian noise, and screenshot distortion.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- PyTorch 2.0+ (CPU or CUDA)

### 2. Environment Setup & Installation
```bash
# Clone the repository
git clone https://github.com/deepforensics/deepforensics.git
cd deepforensics

# Install backend Python dependencies
pip install -r requirements.txt

# Install frontend Node dependencies
cd frontend
npm install
cd ..
```

### 3. Model Training & Checkpoint Generation
```bash
# Train domain models and generate PyTorch checkpoints into model_weights/
python ml/training/train.py
```

### 4. Running the Backend & Frontend
```bash
# Start FastAPI backend (Terminal 1)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start Vite React frontend (Terminal 2)
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📚 Documentation

- [ML Pipeline Architecture](ML_PIPELINE.md)
- [Model Training Guide](TRAINING.md)
- [Evaluation & Robustness Suite](EVALUATION.md)
- [REST API Specification](API.md)
- [Model Card & Ethics Statement](MODEL_CARD.md)

---

## 🧪 Testing

Run the automated PyTest suite:
```bash
python -m pytest
```

---

## 🐳 Docker Deployment

Build and launch complete stack using Docker Compose:
```bash
docker-compose up --build
```
