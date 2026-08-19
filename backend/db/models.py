import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from backend.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String(20), default="researcher")  # researcher, admin, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    detections = relationship("Detection", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class MediaFile(Base):
    __tablename__ = "media_files"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    media_type = Column(String(20), nullable=False)  # image, video
    mime_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash_sha256 = Column(String(64), index=True, nullable=False)
    
    # Metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    detections = relationship("Detection", back_populates="media_file")
    video_frames = relationship("VideoFrame", back_populates="media_file", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, index=True)
    media_file_id = Column(String, ForeignKey("media_files.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    status = Column(String(20), default="completed")  # pending, processing, completed, failed
    primary_prediction = Column(String(50), nullable=False)  # REAL, AI_GENERATED, AI_MANIPULATED, UNCERTAIN
    confidence = Column(Float, nullable=True)  # 0.0 to 1.0, null if MODEL_NOT_READY
    is_multidomain_fused = Column(Boolean, default=True)
    
    # Domain-specific scores
    rgb_score = Column(Float, nullable=True)
    frequency_score = Column(Float, nullable=True)
    residual_score = Column(Float, nullable=True)
    face_score = Column(Float, nullable=True)
    temporal_score = Column(Float, nullable=True)
    
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    media_file = relationship("MediaFile", back_populates="detections")
    user = relationship("User", back_populates="detections")
    model_predictions = relationship("ModelPrediction", back_populates="detection", cascade="all, delete-orphan")
    video_frames = relationship("VideoFrame", back_populates="detection", cascade="all, delete-orphan")
    explanations = relationship("Explanation", back_populates="detection", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    domain_type = Column(String(50), nullable=False)  # spatial_rgb, frequency, residual, face, temporal, fusion
    architecture = Column(String(100), nullable=False)  # EfficientNet, ConvNeXt, ViT, CNN-LSTM, VideoMAE
    weights_path = Column(String(500), nullable=False)
    accuracy = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    eer = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("ModelPrediction", back_populates="model_version")


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(String, primary_key=True, index=True)
    detection_id = Column(String, ForeignKey("detections.id"), nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    
    prediction = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=True)
    raw_scores = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    detection = relationship("Detection", back_populates="model_predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")


class VideoFrame(Base):
    __tablename__ = "video_frames"

    id = Column(String, primary_key=True, index=True)
    detection_id = Column(String, ForeignKey("detections.id"), nullable=False)
    media_file_id = Column(String, ForeignKey("media_files.id"), nullable=False)
    
    frame_index = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    frame_path = Column(String(500), nullable=False)
    heatmap_path = Column(String(500), nullable=True)
    
    prediction = Column(String(50), nullable=False)
    anomaly_score = Column(Float, nullable=False)
    face_detected = Column(Boolean, default=False)
    face_bounding_box = Column(JSON, nullable=True)

    detection = relationship("Detection", back_populates="video_frames")
    media_file = relationship("MediaFile", back_populates="video_frames")


class Explanation(Base):
    __tablename__ = "explanations"

    id = Column(String, primary_key=True, index=True)
    detection_id = Column(String, ForeignKey("detections.id"), nullable=False)
    
    method = Column(String(50), nullable=False)  # Grad-CAM, Frequency-Spectrum, Residual-Noise
    gradcam_heatmap_path = Column(String(500), nullable=True)
    frequency_spectrum_path = Column(String(500), nullable=True)
    residual_noise_path = Column(String(500), nullable=True)
    explanation_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    detection = relationship("Detection", back_populates="explanations")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True)
    experiment_name = Column(String(100), nullable=False)
    ablation_config = Column(String(100), nullable=False)  # RGB_ONLY, RGB_FREQ, FULL_FUSION, etc.
    dataset_name = Column(String(100), nullable=False)  # FaceForensics++, Celeb-DF, DFDC
    
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    eer = Column(Float, nullable=False)
    fpr = Column(Float, nullable=False)
    fnr = Column(Float, nullable=False)
    
    confusion_matrix = Column(JSON, nullable=True)
    metrics_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=False)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")
