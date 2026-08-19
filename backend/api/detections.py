from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.api.schemas import DetectionSummary, DetectionDetailResponse, FrameDetectionSummary
from backend.db.session import get_db
from backend.db.models import Detection, MediaFile, VideoFrame, Explanation, ModelPrediction

router = APIRouter()

@router.get("/detections", response_model=List[DetectionSummary])
async def list_detections(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Detection, MediaFile.original_filename, MediaFile.media_type)
        .join(MediaFile, Detection.media_file_id == MediaFile.id)
        .order_by(Detection.created_at.desc())
        .limit(50)
    )
    result = await db.execute(stmt)
    rows = result.all()

    summaries = []
    for detection, orig_filename, media_type in rows:
        summaries.append(
            DetectionSummary(
                id=detection.id,
                media_file_id=detection.media_file_id,
                original_filename=orig_filename,
                media_type=media_type,
                primary_prediction=detection.primary_prediction,
                confidence=detection.confidence,
                rgb_score=detection.rgb_score,
                frequency_score=detection.frequency_score,
                residual_score=detection.residual_score,
                face_score=detection.face_score,
                temporal_score=detection.temporal_score,
                processing_time_ms=detection.processing_time_ms,
                created_at=detection.created_at
            )
        )
    return summaries


@router.get("/detection/{detection_id}", response_model=DetectionDetailResponse)
async def get_detection(detection_id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Detection, MediaFile.original_filename, MediaFile.media_type)
        .join(MediaFile, Detection.media_file_id == MediaFile.id)
        .where(Detection.id == detection_id)
    )
    result = await db.execute(stmt)
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail=f"Detection with ID '{detection_id}' not found.")

    detection, orig_filename, media_type = row

    # Fetch frames
    frames_res = await db.execute(
        select(VideoFrame).where(VideoFrame.detection_id == detection_id).order_by(VideoFrame.frame_index)
    )
    frames = frames_res.scalars().all()

    # Fetch explanations
    exp_res = await db.execute(
        select(Explanation).where(Explanation.detection_id == detection_id)
    )
    explanations = exp_res.scalars().all()

    # Fetch model predictions
    pred_res = await db.execute(
        select(ModelPrediction).where(ModelPrediction.detection_id == detection_id)
    )
    model_preds = pred_res.scalars().all()

    return DetectionDetailResponse(
        id=detection.id,
        media_file_id=detection.media_file_id,
        original_filename=orig_filename,
        media_type=media_type,
        primary_prediction=detection.primary_prediction,
        confidence=detection.confidence,
        rgb_score=detection.rgb_score,
        frequency_score=detection.frequency_score,
        residual_score=detection.residual_score,
        face_score=detection.face_score,
        temporal_score=detection.temporal_score,
        processing_time_ms=detection.processing_time_ms,
        created_at=detection.created_at,
        model_predictions=[
            {
                "id": mp.id,
                "model_version_id": mp.model_version_id,
                "prediction": mp.prediction,
                "confidence": mp.confidence,
                "raw_scores": mp.raw_scores
            }
            for mp in model_preds
        ],
        frames=[
            FrameDetectionSummary(
                id=f.id,
                frame_index=f.frame_index,
                timestamp_seconds=f.timestamp_seconds,
                frame_path=f.frame_path,
                heatmap_path=f.heatmap_path,
                prediction=f.prediction,
                anomaly_score=f.anomaly_score,
                face_detected=f.face_detected
            )
            for f in frames
        ],
        explanations=[
            {
                "method": e.method,
                "gradcam_heatmap_path": e.gradcam_heatmap_path,
                "frequency_spectrum_path": e.frequency_spectrum_path,
                "residual_noise_path": e.residual_noise_path,
                "explanation_summary": e.explanation_summary
            }
            for e in explanations
        ]
    )


@router.get("/detections/{detection_id}/frames", response_model=List[FrameDetectionSummary])
async def get_detection_frames(detection_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(VideoFrame).where(VideoFrame.detection_id == detection_id).order_by(VideoFrame.frame_index)
    )
    frames = result.scalars().all()
    return [
        FrameDetectionSummary(
            id=f.id,
            frame_index=f.frame_index,
            timestamp_seconds=f.timestamp_seconds,
            frame_path=f.frame_path,
            heatmap_path=f.heatmap_path,
            prediction=f.prediction,
            anomaly_score=f.anomaly_score,
            face_detected=f.face_detected
        )
        for f in frames
    ]
