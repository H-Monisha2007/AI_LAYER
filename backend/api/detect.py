import os
import time
import uuid
import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from backend.api.schemas import DetectionSummary
from backend.db.session import get_db
from backend.db.models import MediaFile, Detection, ModelPrediction, VideoFrame, Explanation
from backend.services.storage import storage_manager
from backend.core.logging import logger

# ML dependencies are optional.
# The API can start even when PyTorch/model weights are unavailable.
try:
    from ml.models.efficientnet_rgb import EfficientNetRGBModel
    from ml.models.convnext_frequency import ConvNeXtFrequencyModel
    from ml.models.srm_residual import SRMResidualModel
    from inference.image.pipeline import ImageInferencePipeline, InferenceNotReadyError
    from inference.video.pipeline import VideoInferencePipeline

    ML_AVAILABLE = True
    ML_IMPORT_ERROR = None

except (ImportError, ModuleNotFoundError) as e:
    EfficientNetRGBModel = None
    ConvNeXtFrequencyModel = None
    SRMResidualModel = None
    ImageInferencePipeline = None
    VideoInferencePipeline = None

    class InferenceNotReadyError(Exception):
        pass

    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(e)
router = APIRouter()

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model_weights"))

# Checkpoint paths
rgb_path = os.path.join(WEIGHTS_DIR, "efficientnet_b4", "best.pt")
freq_path = os.path.join(WEIGHTS_DIR, "convnext_dct", "best.pt")
noise_path = os.path.join(WEIGHTS_DIR, "noise_model", "best.pt")

# Initialize models
rgb_model = EfficientNetRGBModel(weights_path=rgb_path if os.path.exists(rgb_path) else None)
if os.path.exists(rgb_path):
    try:
        rgb_model.load()
    except Exception as e:
        logger.warning(f"Failed to load RGB weights: {e}")

freq_model = ConvNeXtFrequencyModel(weights_path=freq_path if os.path.exists(freq_path) else None)
if os.path.exists(freq_path):
    try:
        freq_model.load()
    except Exception as e:
        logger.warning(f"Failed to load Frequency weights: {e}")

noise_model = SRMResidualModel(weights_path=noise_path if os.path.exists(noise_path) else None)
if os.path.exists(noise_path):
    try:
        noise_model.load()
    except Exception as e:
        logger.warning(f"Failed to load Noise weights: {e}")

# Initialize Pipelines
image_pipeline = ImageInferencePipeline(
    rgb_model=rgb_model,
    frequency_model=freq_model,
    residual_model=noise_model,
)
video_pipeline = VideoInferencePipeline(image_pipeline=image_pipeline)


@router.post("/detect/image", response_model=DetectionSummary)
async def detect_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()
    logger.info("[DETECTION] Request received for image upload")

    # 1. Secure Storage & MIME Validation
    try:
        file_info = await storage_manager.save_uploaded_file(file, media_type="image")
        logger.info(f"[DETECTION] File validated & saved: {file_info['original_filename']}")
    except HTTPException as e:
        logger.warning(f"[DETECTION ERROR] File validation failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"[DETECTION ERROR] Failed to save upload: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file upload: {str(e)}")

    # 2. Database Record - Media File
    db_media = MediaFile(
        id=file_info["id"],
        filename=file_info["filename"],
        original_filename=file_info["original_filename"],
        media_type="image",
        mime_type=file_info["mime_type"],
        file_path=file_info["file_path"],
        file_size_bytes=file_info["file_size_bytes"],
        file_hash_sha256=file_info["file_hash_sha256"],
        created_at=datetime.datetime.utcnow()
    )
    db.add(db_media)

    detection_id = str(uuid.uuid4())

    # 3. Real Inference Pipeline Execution
    results = {}
    try:
        logger.info("[DETECTION] Running multi-signal inference pipeline...")
        results = image_pipeline.run(file_info["file_path"], generate_gradcam=True)
        processing_time_ms = float(results.get("processing_time_ms", round((time.time() - start_time) * 1000, 2)))
        logger.info(f"[DETECTION] Inference completed in {processing_time_ms} ms. Decision: {results['primary_prediction']}")

        db_detection = Detection(
            id=detection_id,
            media_file_id=db_media.id,
            status="completed",
            primary_prediction=results["primary_prediction"],
            confidence=float(results["confidence"]) if results.get("confidence") is not None else None,
            rgb_score=float(results["rgb_score"]) if results.get("rgb_score") is not None else None,
            frequency_score=float(results["frequency_score"]) if results.get("frequency_score") is not None else None,
            residual_score=float(results["residual_score"]) if results.get("residual_score") is not None else None,
            face_score=float(results["face_score"]) if results.get("face_score") is not None else None,
            processing_time_ms=processing_time_ms,
            created_at=datetime.datetime.utcnow()
        )
        message = None

        MODEL_VERSION_MAP = {
            "rgb_spatial": "effnet-b4-rgb",
            "frequency": "convnext-dct-freq",
            "noise_residual": "srm-noise-residual",
            "face_analysis": "opencv-face-detector",
        }

        # Store model predictions breakdown safely
        for dom, pred_info in results.get("forensic_signals", {}).items():
            conf_val = pred_info.get("ai_probability")
            if conf_val is not None:
                conf_val = float(conf_val)
            db_mp = ModelPrediction(
                id=str(uuid.uuid4()),
                detection_id=detection_id,
                model_version_id=MODEL_VERSION_MAP.get(dom, "effnet-b4-rgb"),
                prediction=results["primary_prediction"],
                confidence=conf_val,
                raw_scores=pred_info,
                created_at=datetime.datetime.utcnow()
            )
            db.add(db_mp)

        # Store Grad-CAM Explanation if generated
        if results.get("gradcam_heatmap_path"):
            db_exp = Explanation(
                id=str(uuid.uuid4()),
                detection_id=detection_id,
                method="Grad-CAM",
                gradcam_heatmap_path=results.get("gradcam_heatmap_path"),
                explanation_summary="Visual saliency map highlighting key spatial image regions influencing model inference.",
                created_at=datetime.datetime.utcnow()
            )
            db.add(db_exp)

    except InferenceNotReadyError as err:
        logger.warning(f"[DETECTION] Model weights not loaded: {err}")
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        message = "Trained model weights are not loaded. Train or add PyTorch checkpoints to model_weights/ directory."

        db_detection = Detection(
            id=detection_id,
            media_file_id=db_media.id,
            status="MODEL_NOT_READY",
            primary_prediction="UNCERTAIN",
            confidence=None,
            rgb_score=None,
            frequency_score=None,
            residual_score=None,
            face_score=None,
            processing_time_ms=processing_time_ms,
            created_at=datetime.datetime.utcnow()
        )

    db.add(db_detection)
    await db.commit()
    logger.info("[DETECTION] Detection record successfully committed to database")

    return DetectionSummary(
        id=db_detection.id,
        media_file_id=db_media.id,
        original_filename=db_media.original_filename,
        media_type="image",
        status=db_detection.status,
        primary_prediction=db_detection.primary_prediction,
        decision=db_detection.primary_prediction,
        verdict=results.get("verdict") if db_detection.status == "completed" else None,
        confidence=db_detection.confidence,
        reliability=float(results.get("reliability")) if results.get("reliability") is not None else None,
        calibrated_probability_ai=float(results.get("calibrated_probability_ai")) if results.get("calibrated_probability_ai") is not None else None,
        probability_real=float(results.get("real_probability")) if results.get("real_probability") is not None else None,
        uncertainty=float(results.get("uncertainty")) if results.get("uncertainty") is not None else None,
        is_ood=bool(results.get("is_ood", False)) if db_detection.status == "completed" else False,
        ood_status=results.get("ood_status") if db_detection.status == "completed" else None,
        model_agreement=float(results.get("model_agreement")) if results.get("model_agreement") is not None else None,
        user_explanation=results.get("user_explanation") if db_detection.status == "completed" else None,
        warnings=results.get("warnings", []) if db_detection.status == "completed" else [],
        rgb_score=db_detection.rgb_score,
        frequency_score=db_detection.frequency_score,
        residual_score=db_detection.residual_score,
        face_score=db_detection.face_score,
        temporal_score=None,
        evidence=results.get("evidence") if db_detection.status == "completed" else None,
        forensic_flags=results.get("forensic_flags", []) if db_detection.status == "completed" else [],
        processing_time_ms=processing_time_ms,
        message=message,
        analysis_status=results.get("analysis_status", "VALID") if db_detection.status == "completed" else "MODEL_NOT_READY",
        created_at=db_detection.created_at
    )


@router.post("/detect/video", response_model=DetectionSummary)
async def detect_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()

    # 1. Secure Storage & MIME Validation
    file_info = await storage_manager.save_uploaded_file(file, media_type="video")

    # 2. Database Record - Media File
    db_media = MediaFile(
        id=file_info["id"],
        filename=file_info["filename"],
        original_filename=file_info["original_filename"],
        media_type="video",
        mime_type=file_info["mime_type"],
        file_path=file_info["file_path"],
        file_size_bytes=file_info["file_size_bytes"],
        file_hash_sha256=file_info["file_hash_sha256"],
        created_at=datetime.datetime.utcnow()
    )
    db.add(db_media)

    detection_id = str(uuid.uuid4())

    try:
        results = video_pipeline.run(file_info["file_path"], num_samples=16)
        processing_time_ms = results.get("processing_time_ms", round((time.time() - start_time) * 1000, 2))

        db_detection = Detection(
            id=detection_id,
            media_file_id=db_media.id,
            status="completed",
            primary_prediction=results["primary_prediction"],
            confidence=results["confidence"],
            temporal_score=results.get("temporal_consistency"),
            processing_time_ms=processing_time_ms,
            created_at=datetime.datetime.utcnow()
        )
        message = None

        # Store suspicious frame records
        for sf in results.get("suspicious_frames", []):
            db_frame = VideoFrame(
                id=str(uuid.uuid4()),
                detection_id=detection_id,
                frame_index=sf["frame_index"],
                timestamp_seconds=sf["timestamp_seconds"],
                frame_path=file_info["file_path"],
                prediction=sf["prediction"],
                anomaly_score=sf["anomaly_score"],
                face_detected=sf["face_detected"],
                created_at=datetime.datetime.utcnow()
            )
            db.add(db_frame)

    except InferenceNotReadyError as err:
        logger.warning(f"Video inference model not ready: {err}")
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        message = "Trained model weights are not loaded. Train or add PyTorch checkpoints to model_weights/ directory."

        db_detection = Detection(
            id=detection_id,
            media_file_id=db_media.id,
            status="MODEL_NOT_READY",
            primary_prediction="UNCERTAIN",
            confidence=None,
            rgb_score=None,
            frequency_score=None,
            residual_score=None,
            face_score=None,
            temporal_score=None,
            processing_time_ms=processing_time_ms,
            created_at=datetime.datetime.utcnow()
        )

    db.add(db_detection)
    await db.commit()

    return DetectionSummary(
        id=db_detection.id,
        media_file_id=db_media.id,
        original_filename=db_media.original_filename,
        media_type="video",
        status=db_detection.status,
        primary_prediction=db_detection.primary_prediction,
        confidence=db_detection.confidence,
        rgb_score=None,
        frequency_score=None,
        residual_score=None,
        face_score=None,
        temporal_score=db_detection.temporal_score,
        processing_time_ms=processing_time_ms,
        message=message,
        created_at=db_detection.created_at
    )
