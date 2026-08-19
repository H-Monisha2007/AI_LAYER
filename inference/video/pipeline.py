"""
Video Inference Pipeline

Decodes videos, extracts frame samples, performs frame-level multidomain forensic analysis,
runs face detection & tracking, calculates temporal consistency, ranks suspicious frames,
and aggregates frame predictions into a calibrated video classification.
"""
from typing import Dict, Any, List, Optional
import os
import time
import cv2
import numpy as np
import torch

from ml.models.face_detector import face_detector
from ml.preprocessing import compute_phash
from ml.fusion import default_fusion_engine
from ml.calibration import default_calibrator
from ml.decision import default_decision_engine
from inference.image.pipeline import ImageInferencePipeline, InferenceNotReadyError
from PIL import Image


class VideoInferencePipeline:
    """
    Multidomain temporal video deepfake & manipulation detector.
    """

    def __init__(self, image_pipeline: ImageInferencePipeline, temporal_model=None):
        self.image_pipeline = image_pipeline
        self.temporal_model = temporal_model

    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        return {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration_seconds": round(duration_sec, 2),
        }

    def sample_frames(self, video_path: str, num_samples: int = 16) -> List[Dict[str, Any]]:
        meta = self.extract_metadata(video_path)
        total_frames = meta["total_frames"]
        fps = meta["fps"]

        if total_frames <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        cap = cv2.VideoCapture(video_path)

        sampled_frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = round(frame_idx / fps, 2) if fps > 0 else 0.0

            sampled_frames.append({
                "frame_index": int(frame_idx),
                "timestamp_seconds": timestamp,
                "frame_rgb": frame_rgb
            })

        cap.release()
        return sampled_frames

    def run(self, video_path: str, num_samples: int = 16) -> Dict[str, Any]:
        t_start = time.time()

        if not self.image_pipeline.active_domains:
            raise InferenceNotReadyError(
                "No trained model checkpoints are loaded in model_weights/. "
                "Train or provide checkpoints before performing video inference."
            )

        meta = self.extract_metadata(video_path)
        sampled = self.sample_frames(video_path, num_samples=num_samples)

        if not sampled:
            raise ValueError("No frames could be extracted from the video file.")

        frame_results = []
        ai_probs = []

        # Save temporary frames to run through image pipeline
        temp_dir = os.path.dirname(video_path)

        for sf in sampled:
            frame_filename = f"temp_frame_{sf['frame_index']}.jpg"
            frame_tmp_path = os.path.join(temp_dir, frame_filename)

            try:
                pil_frame = Image.fromarray(sf["frame_rgb"])
                pil_frame.save(frame_tmp_path, quality=95)

                # Run per-frame multidomain inference
                frame_res = self.image_pipeline.run(frame_tmp_path, generate_gradcam=False)

                ai_prob = frame_res["ai_probability"]
                ai_probs.append(ai_prob)

                # Check face in frame
                _, bbox = face_detector.crop_primary_face(sf["frame_rgb"])

                frame_results.append({
                    "frame_index": sf["frame_index"],
                    "timestamp_seconds": sf["timestamp_seconds"],
                    "prediction": frame_res["primary_prediction"],
                    "anomaly_score": round(ai_prob, 4),
                    "ai_probability": round(ai_prob, 4),
                    "face_detected": bbox is not None,
                    "face_bounding_box": bbox,
                    "frame_path": frame_tmp_path,
                })
            except Exception as e:
                # Log frame evaluation error
                continue
            finally:
                if os.path.exists(frame_tmp_path):
                    try:
                        os.remove(frame_tmp_path)
                    except Exception:
                        pass

        if not ai_probs:
            raise RuntimeError("Frame-level evaluation failed for all sampled frames.")

        # Temporal analysis: variance across frame predictions indicates temporal instability/flicker
        avg_ai_prob = float(np.mean(ai_probs))
        frame_std = float(np.std(ai_probs))
        
        # Temporal consistency: 1.0 = highly consistent across time, lower = temporal artifacts / deepfake flicker
        temporal_consistency = round(float(np.clip(1.0 - frame_std * 2.5, 0.0, 1.0)), 4)

        # Calibrate aggregated video P(AI)
        calibrated_v_ai = default_calibrator.calibrate_probability(avg_ai_prob)

        # Apply threshold decision logic
        decision = default_decision_engine.classify(calibrated_v_ai, uncertainty_score=round(frame_std, 4))

        # Rank top suspicious frames (highest P(AI) anomaly scores)
        suspicious = sorted(frame_results, key=lambda f: f["anomaly_score"], reverse=True)[:5]

        # Cleanup frame paths from response objects
        for sf in suspicious:
            sf.pop("frame_path", None)

        t_end = time.time()
        total_ms = round((t_end - t_start) * 1000, 2)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        frames_ai_count = sum(1 for p in ai_probs if p >= 0.5)
        frames_real_count = len(ai_probs) - frames_ai_count

        return {
            "status": "completed",
            "primary_prediction": decision["classification"],
            "confidence": decision["confidence"],
            "ai_probability": decision["ai_probability"],
            "real_probability": decision["real_probability"],
            "uncertainty": decision["uncertainty"],
            "temporal_consistency": temporal_consistency,
            "metadata": meta,
            "frames_analyzed": len(ai_probs),
            "frames_detected_as_ai": frames_ai_count,
            "frames_detected_as_real": frames_real_count,
            "suspicious_frames": suspicious,
            "processing": {
                "total_ms": total_ms,
                "device": device_str,
            },
            "processing_time_ms": total_ms,
        }
