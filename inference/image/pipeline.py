"""
Image Forensic Inference Pipeline

Orchestrates multi-signal forensic evaluation for a single image:
1. Centralized Image Preprocessing & Orientation
2. Auxiliary Metadata & EXIF Tag Inspection (ml.metadata)
3. 2D-DCT / 2D-FFT Frequency Spectrum Analysis (ml.frequency)
4. SRM High-Pass Noise Residual & Noise Consistency Analysis (ml.noise)
5. Multi-Crop High-Resolution Patch Analysis (ml.patch_forensics)
6. RGB Spatial Model (EfficientNet-B4)
7. Frequency DCT Model (ConvNeXt-Tiny)
8. SRM Noise Residual Model (SRM-ResNet18)
9. Face Crop Detection & Analysis (OpenCV Haar Cascade)
10. Multi-View Patch Logit Aggregation
11. Out-of-Distribution (OOD) & Uncertainty Evaluation (ml.ood)
12. Score Fusion Engine (ml.fusion)
13. Probability Calibration Engine (ml.calibration)
14. Calibrated 3-Way Decision Engine (ml.decision)
15. Grad-CAM Visual Heatmap Saliency Generator (ml.explainability)

Rule: If no trained checkpoints are loaded, pipeline raises InferenceNotReadyError.
It NEVER generates random, hardcoded, or simulated results.
"""
from typing import Optional, Dict, Any, List
import time
import os
import torch
import numpy as np
from PIL import Image

from ml.preprocessing import default_preprocessor, load_and_orient_image
from ml.models.base import BaseForensicModel
from ml.models.face_detector import face_detector
from ml.metadata import analyze_image_metadata
from ml.frequency import compute_spectral_residual_features
from ml.noise import compute_noise_residual_statistics
from ml.patch_forensics import extract_patches, aggregate_patch_scores
from ml.ood import default_ood_detector
from ml.fusion import default_fusion_engine
from ml.calibration import default_calibrator
from ml.decision import default_decision_engine
from ml.explainability.gradcam import GradCAM, overlay_heatmap
from backend.core.logging import logger


class InferenceNotReadyError(Exception):
    """Raised when no trained model checkpoints are loaded."""
    pass


class ImageInferencePipeline:
    """
    Multidomain multi-signal image inference pipeline.
    """

    MODEL_VERSIONS = {
        "rgb_spatial": "1.0.0",
        "frequency": "1.0.0",
        "noise_residual": "1.0.0",
        "face_analysis": "1.0.0",
        "fusion": "1.0.0",
    }

    def __init__(
        self,
        rgb_model: Optional[BaseForensicModel] = None,
        frequency_model: Optional[BaseForensicModel] = None,
        residual_model: Optional[BaseForensicModel] = None,
        face_model: Optional[BaseForensicModel] = None,
    ):
        self.rgb_model = rgb_model
        self.frequency_model = frequency_model
        self.residual_model = residual_model
        self.face_model = face_model

    @property
    def active_domains(self) -> List[str]:
        active = []
        if self.rgb_model and self.rgb_model.is_ready:
            active.append("rgb")
        if self.frequency_model and self.frequency_model.is_ready:
            active.append("frequency")
        if self.residual_model and self.residual_model.is_ready:
            active.append("residual")
        if self.face_model and self.face_model.is_ready:
            active.append("face")
        return active

    def run(self, image_path: str, generate_gradcam: bool = True) -> Dict[str, Any]:
        """
        Run multidomain forensic inference on image_path.
        Returns complete, evidence-rich response dict.
        Raises InferenceNotReadyError if zero models are ready.
        """
        t_start = time.time()

        if not self.active_domains:
            raise InferenceNotReadyError(
                "No trained model checkpoints are loaded in model_weights/. "
                "Train or provide checkpoints before performing inference."
            )

        # 1. Centralized Image Preprocessing & Hashes
        t_prep_start = time.time()
        prep_data = default_preprocessor.preprocess_image(image_path)
        pil_img = prep_data["pil_image"]
        img_np = np.array(pil_img)
        t_prep_end = time.time()

        # 2. Extract Multi-Signal Forensic Evidence
        metadata_evidence = analyze_image_metadata(image_path)
        freq_evidence = compute_spectral_residual_features(img_np)
        noise_evidence = compute_noise_residual_statistics(img_np)

        # 3. Multi-Crop Patch Analysis
        patch_records = extract_patches(pil_img, patch_size=(224, 224), max_patches=5)
        patch_probs: List[float] = []

        # 4. Domain Inference
        t_inf_start = time.time()
        domain_results: Dict[str, Dict[str, Any]] = {}
        domain_scores: Dict[str, Optional[float]] = {
            "rgb": None,
            "frequency": None,
            "residual": None,
            "face": None,
        }

        # RGB Spatial Model (Full Image & Patches)
        if "rgb" in self.active_domains:
            try:
                rgb_res = self.rgb_model.predict_from_input(pil_img)
                domain_results["rgb"] = rgb_res
                domain_scores["rgb"] = rgb_res.get("raw_scores", {}).get("AI_GENERATED")

                # Evaluate patches using RGB model
                for p_rec in patch_records:
                    if p_rec["label"] != "full_image":
                        p_res = self.rgb_model.predict_from_input(p_rec["image"])
                        p_score = p_res.get("raw_scores", {}).get("AI_GENERATED")
                        if p_score is not None:
                            patch_probs.append(p_score)
            except Exception as e:
                logger.error(f"RGB Spatial model inference failed: {e}")

        # Frequency Domain Model
        if "frequency" in self.active_domains:
            try:
                freq_res = self.frequency_model.predict_from_input(pil_img)
                domain_results["frequency"] = freq_res
                domain_scores["frequency"] = freq_res.get("raw_scores", {}).get("AI_GENERATED")
            except Exception as e:
                logger.error(f"Frequency model inference failed: {e}")

        # SRM Noise Residual Model
        if "residual" in self.active_domains:
            try:
                res_res = self.residual_model.predict_from_input(pil_img)
                domain_results["residual"] = res_res
                domain_scores["residual"] = res_res.get("raw_scores", {}).get("AI_GENERATED")
            except Exception as e:
                logger.error(f"Residual model inference failed: {e}")

        # Face Analysis Model
        face_status = "not_applicable"
        if "face" in self.active_domains:
            try:
                cropped_face, bbox = face_detector.crop_primary_face(img_np)
                if bbox is not None:
                    face_res = self.face_model.predict_from_input(cropped_face)
                    domain_results["face"] = face_res
                    domain_scores["face"] = face_res.get("raw_scores", {}).get("AI_GENERATED")
                    face_status = "analyzed"
                else:
                    face_status = "no_face_detected"
            except Exception as e:
                logger.error(f"Face model inference failed: {e}")
                face_status = "inconclusive"
        elif face_detector.cascade is not None:
            _, bbox = face_detector.crop_primary_face(img_np)
            if bbox is not None:
                face_status = "face_detected_no_model"

        t_inf_end = time.time()

        # 5. Patch Evidence Aggregation
        patch_aggregation = aggregate_patch_scores(patch_probs, method="logit_max_pool")

        # 6. OOD & Uncertainty Evaluation
        ood_result = default_ood_detector.evaluate(
            domain_scores=domain_scores,
            img_np=img_np,
            patch_variance=patch_aggregation["patch_variance"]
        )

        # 7. Multidomain Score Fusion
        fused = default_fusion_engine.fuse_domain_scores(domain_scores)
        raw_p_ai = fused["ai_probability"]

        # Forensic flags aggregation
        forensic_flags = list(metadata_evidence.get("forensic_flags", []))
        if ood_result["is_ood"]:
            forensic_flags.extend(ood_result["reasons"])

        # 8. Probability Calibration
        calibrated_p_ai = default_calibrator.calibrate_probability(raw_p_ai)

        # 9. Calibrated 5-Category Verdict & Decision Engine
        decision = default_decision_engine.classify(
            calibrated_p_ai=calibrated_p_ai,
            uncertainty_score=fused["uncertainty"],
            ood_result=ood_result,
            domain_scores=domain_scores,
            raw_flags=forensic_flags
        )

        # 10. Optional Grad-CAM Heatmap Generation
        heatmap_path = None
        if generate_gradcam and "rgb" in self.active_domains and hasattr(self.rgb_model, "model"):
            try:
                target_layer = list(self.rgb_model.model.features.children())[-1]
                gcam = GradCAM(self.rgb_model.model, target_layer)
                input_tensor = prep_data["tensor"].to(self.rgb_model.device)
                heatmap_2d = gcam.generate(input_tensor, target_class=1)
                overlay = overlay_heatmap(img_np, heatmap_2d)

                save_dir = os.path.dirname(image_path)
                heatmap_filename = f"gradcam_{os.path.basename(image_path)}"
                heatmap_path = os.path.join(save_dir, heatmap_filename)
                load_and_orient_image(overlay).save(heatmap_path)
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")

        t_end = time.time()

        prep_ms = round((t_prep_end - t_prep_start) * 1000, 2)
        inf_ms = round((t_inf_end - t_inf_start) * 1000, 2)
        total_ms = round((t_end - t_start) * 1000, 2)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "status": "completed",
            "primary_prediction": decision["primary_prediction"],
            "decision": decision["primary_prediction"],
            "verdict": {
                "verdict_key": decision["verdict_key"],
                "display_label": decision["display_label"],
                "category_icon": decision["category_icon"],
                "ai_score": decision["ai_score"],
                "real_score": decision["real_score"],
                "confidence": decision["confidence"],
                "reliability": decision["reliability"],
                "model_agreement": decision["model_agreement"],
                "ood_status": decision["ood_status"],
                "user_explanation": decision["user_explanation"],
            },
            "confidence": decision["confidence"],
            "reliability": decision["reliability"],
            "ai_probability": decision["ai_score"],
            "calibrated_probability_ai": decision["ai_score"],
            "real_probability": decision["real_score"],
            "uncertainty": decision["uncertainty"],
            "is_ood": decision["is_ood"],
            "ood_status": decision["ood_status"],
            "model_agreement": decision["model_agreement"],
            "user_explanation": decision["user_explanation"],
            "warnings": decision["warnings"],
            "decision_reasons": decision["decision_reasons"],
            "rgb_score": domain_scores["rgb"],
            "frequency_score": domain_scores["frequency"],
            "residual_score": domain_scores["residual"],
            "face_score": domain_scores["face"],
            "face_status": face_status,
            "evidence": {
                "visual": {
                    "rgb_score": domain_scores["rgb"],
                    "model": "EfficientNet-B4",
                },
                "frequency": freq_evidence,
                "residual": noise_evidence,
                "patch": patch_aggregation,
                "metadata": metadata_evidence,
            },
            "forensic_flags": forensic_flags,
            "forensic_signals": {
                "rgb_spatial": {"ai_probability": domain_scores["rgb"]},
                "frequency": {"ai_probability": domain_scores["frequency"]},
                "noise_residual": {"ai_probability": domain_scores["residual"]},
                "face_analysis": {"status": face_status, "ai_probability": domain_scores["face"]},
            },
            "model_versions": self.MODEL_VERSIONS,
            "gradcam_heatmap_path": heatmap_path,
            "processing": {
                "preprocessing_ms": prep_ms,
                "inference_ms": inf_ms,
                "total_ms": total_ms,
                "device": device_str,
            },
            "processing_time_ms": total_ms,
            "active_domains": self.active_domains,
            "analysis_status": "VALID",
        }
