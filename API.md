# DeepForensics — REST API Specification

The FastAPI backend exposes endpoints for image/video detection, system health, model status, past detection history, and evaluation benchmarks.

Base URL: `http://localhost:8000/api`

---

## Endpoints Summary

### 1. `GET /api/health`
System status, database connectivity, PyTorch CUDA availability, and device info.

### 2. `POST /api/detect/image`
Uploads an image file for multidomain forensic detection.
- **Form Data**: `file` (multipart/form-data)
- **Response**: `DetectionSummary` JSON containing primary prediction, confidence, domain scores, processing time, and Grad-CAM heatmap path.

### 3. `POST /api/detect/video`
Uploads a video file for frame sampling & temporal deepfake detection.
- **Form Data**: `file` (multipart/form-data)
- **Response**: `DetectionSummary` JSON containing temporal score, frame statistics, and ranked suspicious frames.

### 4. `GET /api/models`
Returns list of registered domain models, architectures, status (`READY` / `MODEL_NOT_READY`), and validation metrics.

### 5. `GET /api/detections`
Returns detection history stored in the database.

### 6. `GET /api/detection/{id}`
Returns full detailed breakdown of a single detection, including per-model predictions, video frames, and Grad-CAM explanations.

### 7. `GET /api/evaluation`
Returns real evaluation metrics, confusion matrix, per-generator breakdown, unseen generator generalization performance, and robustness benchmarks.
