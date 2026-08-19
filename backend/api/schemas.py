import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    database_connected: bool
    cuda_available: bool
    device: str
    timestamp: datetime.datetime


class ModelInfoResponse(BaseModel):
    id: str
    name: str
    version: str
    domain_type: str
    architecture: str
    accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    eer: Optional[float] = None
    is_active: bool


class VerdictWarning(BaseModel):
    code: str
    title: str
    message: str


class VerdictDetail(BaseModel):
    verdict_key: str
    display_label: str
    category_icon: str
    ai_score: float
    real_score: float
    confidence: float
    reliability: float
    model_agreement: float
    ood_status: str
    user_explanation: str


class DetectionSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    media_file_id: str
    original_filename: str
    media_type: str
    status: str = "completed"
    primary_prediction: str
    decision: Optional[str] = None
    verdict: Optional[VerdictDetail] = None
    confidence: Optional[float] = None
    reliability: Optional[float] = None
    calibrated_probability_ai: Optional[float] = None
    probability_real: Optional[float] = None
    uncertainty: Optional[float] = None
    is_ood: Optional[bool] = False
    ood_status: Optional[str] = None
    model_agreement: Optional[float] = None
    user_explanation: Optional[str] = None
    warnings: Optional[List[VerdictWarning]] = []
    rgb_score: Optional[float] = None
    frequency_score: Optional[float] = None
    residual_score: Optional[float] = None
    face_score: Optional[float] = None
    temporal_score: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None
    forensic_flags: Optional[List[str]] = []
    processing_time_ms: Optional[float] = None
    message: Optional[str] = None
    analysis_status: Optional[str] = "VALID"
    created_at: datetime.datetime


class FrameDetectionSummary(BaseModel):
    id: str
    frame_index: int
    timestamp_seconds: float
    frame_path: str
    heatmap_path: Optional[str] = None
    prediction: str
    anomaly_score: float
    face_detected: bool


class ExplanationSummary(BaseModel):
    method: str
    gradcam_heatmap_path: Optional[str] = None
    frequency_spectrum_path: Optional[str] = None
    residual_noise_path: Optional[str] = None
    explanation_summary: Optional[str] = None


class DetectionDetailResponse(DetectionSummary):
    model_predictions: List[Dict[str, Any]] = []
    frames: List[FrameDetectionSummary] = []
    explanations: List[ExplanationSummary] = []


class SystemMetricsResponse(BaseModel):
    total_detections: int
    real_count: int
    ai_generated_count: int
    ai_manipulated_count: int
    uncertain_count: int
    avg_processing_time_ms: float
    models_active: int
    latest_evaluations: List[Dict[str, Any]] = []
