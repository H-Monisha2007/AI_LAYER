"""
DeepForensics Centralized Decision Engine & Verdict Builder

Determines 5-category human-readable verdicts, separates model score from reliability,
calculates model agreement, and formats human-readable forensic explanations and warnings.

5 Categories:
- HIGH_CHANCE_AI_GENERATED (🔴 HIGH CHANCE OF AI-GENERATED)
- MODERATE_CHANCE_AI_GENERATED (🟠 MODERATE CHANCE OF AI-GENERATED)
- UNCERTAIN (🟡 UNCERTAIN / INSUFFICIENT EVIDENCE)
- MODERATE_CHANCE_REAL (🟢 MODERATE CHANCE OF REAL)
- HIGH_CHANCE_REAL (🟢 HIGH CHANCE OF REAL)
"""
from typing import Dict, Any, List, Optional
import numpy as np


class ForensicDecisionEngine:
    """
    Centralized forensic decision engine converting calibrated probabilities, domain scores,
    patch aggregation, and OOD signals into structured, calibrated 5-category verdicts.
    """

    def __init__(
        self,
        high_ai_thresh: float = 0.75,
        mod_ai_thresh: float = 0.60,
        mod_real_thresh: float = 0.40,
        high_real_thresh: float = 0.25
    ):
        self.high_ai_thresh = high_ai_thresh
        self.mod_ai_thresh = mod_ai_thresh
        self.mod_real_thresh = mod_real_thresh
        self.high_real_thresh = high_real_thresh

    def calculate_model_agreement(self, domain_scores: Dict[str, Optional[float]]) -> float:
        """
        Calculates normalized model agreement metric [0.0 to 1.0] across active domains.
        """
        valid_scores = [v for v in domain_scores.values() if v is not None]
        if len(valid_scores) < 2:
            return 1.0  # Single domain active -> default high agreement

        score_range = max(valid_scores) - min(valid_scores)
        agreement = max(0.0, 1.0 - (score_range * 1.25))
        return round(float(agreement), 4)

    def format_human_warnings(
        self,
        raw_flags: List[str],
        ood_result: Optional[Dict[str, Any]],
        model_agreement: float
    ) -> List[Dict[str, str]]:
        """
        Converts internal flags and codes into clear, user-friendly warnings.
        """
        warnings = []
        seen = set()

        if model_agreement < 0.60 and "model_disagreement" not in seen:
            seen.add("model_disagreement")
            warnings.append({
                "code": "MODEL_DISAGREEMENT",
                "title": "Model Disagreement",
                "message": "Independent forensic domains produced varying signals, reducing overall prediction reliability."
            })

        for flag in raw_flags:
            if "EXIF_METADATA_STRIPPED" in flag and "exif_absent" not in seen:
                seen.add("exif_absent")
                warnings.append({
                    "code": "EXIF_METADATA_ABSENT",
                    "title": "Metadata Absent",
                    "message": "EXIF metadata is absent. Many web platforms remove metadata automatically; this is not evidence of AI generation."
                })
            elif "EXTREME_LOW_VARIANCE" in flag and "low_variance" not in seen:
                seen.add("low_variance")
                warnings.append({
                    "code": "LOW_VARIANCE_IMAGE",
                    "title": "Uniform / Low-Variance Image",
                    "message": "Image contains uniform color or low spatial variation, which reduces noise residual signal clarity."
                })

        if ood_result and ood_result.get("is_ood") and "ood_warning" not in seen:
            seen.add("ood_warning")
            warnings.append({
                "code": "DOMAIN_SHIFT_WARNING",
                "title": "Partial Domain Shift Detected",
                "message": "Image exhibits structural properties near the boundary of standard calibration. Score reflects model estimate with reduced reliability."
            })

        return warnings

    def classify(
        self,
        calibrated_p_ai: float,
        uncertainty_score: float = 0.0,
        ood_result: Optional[Dict[str, Any]] = None,
        domain_scores: Optional[Dict[str, Optional[float]]] = None,
        raw_flags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Builds complete 5-category verdict object.
        """
        p_ai = float(np.clip(calibrated_p_ai, 0.0, 1.0))
        p_real = float(1.0 - p_ai)

        domain_scores = domain_scores or {}
        raw_flags = raw_flags or []
        model_agreement = self.calculate_model_agreement(domain_scores)

        # 1. Compute raw model confidence
        confidence = float(max(p_ai, p_real))

        # 2. Compute reliability score (decoupled from classification)
        ood_flag = ood_result.get("is_ood", False) if ood_result else False
        reliability = confidence * (0.50 + 0.50 * model_agreement)
        if ood_flag:
            reliability *= 0.80
        if uncertainty_score > 0.35:
            reliability *= 0.85
        reliability = round(float(np.clip(reliability, 0.05, 1.0)), 4)

        # 3. Determine 5-Category Verdict
        if reliability < 0.25:
            verdict_key = "UNCERTAIN"
            display_label = "UNCERTAIN / INSUFFICIENT EVIDENCE"
            category_icon = "🟡"
            user_explanation = "The available forensic evidence does not reliably distinguish between AI-generated and real imagery due to low signal reliability."
            primary_prediction = "UNCERTAIN"
        elif p_ai >= self.high_ai_thresh:
            verdict_key = "HIGH_CHANCE_AI_GENERATED"
            display_label = "HIGH CHANCE OF AI-GENERATED"
            category_icon = "🔴"
            user_explanation = "Multiple independent forensic signals (RGB spatial, frequency spectrum, noise residual) strongly favor AI-generated imagery."
            primary_prediction = "AI_GENERATED"
        elif p_ai >= self.mod_ai_thresh:
            verdict_key = "MODERATE_CHANCE_AI_GENERATED"
            display_label = "MODERATE CHANCE OF AI-GENERATED"
            category_icon = "🟠"
            user_explanation = "Forensic evidence leans toward AI generation, but features are not strong enough for a high-confidence conclusion."
            primary_prediction = "AI_GENERATED"
        elif p_ai <= self.high_real_thresh:
            verdict_key = "HIGH_CHANCE_REAL"
            display_label = "HIGH CHANCE OF REAL"
            category_icon = "🟢"
            user_explanation = "Forensic signals detect natural camera noise profiles, authentic frequency spectrum distribution, and organic pixel structures typical of genuine photography."
            primary_prediction = "REAL"
        elif p_ai <= self.mod_real_thresh:
            verdict_key = "MODERATE_CHANCE_REAL"
            display_label = "MODERATE CHANCE OF REAL"
            category_icon = "🟢"
            user_explanation = "Forensic analysis indicates characteristics more consistent with real photography than AI generation."
            primary_prediction = "REAL"
        else:
            verdict_key = "UNCERTAIN"
            display_label = "UNCERTAIN / INSUFFICIENT EVIDENCE"
            category_icon = "🟡"
            user_explanation = "Forensic signals produced balanced or boundary indicators. Evidence is insufficient for a definitive decision."
            primary_prediction = "UNCERTAIN"

        ood_status = "WARNING" if ood_flag else ("CAUTION" if model_agreement < 0.60 else "OK")
        human_warnings = self.format_human_warnings(raw_flags, ood_result, model_agreement)

        return {
            "primary_prediction": primary_prediction,
            "classification": primary_prediction,
            "verdict_key": verdict_key,
            "display_label": display_label,
            "category_icon": category_icon,
            "ai_score": round(p_ai, 4),
            "real_score": round(p_real, 4),
            "calibrated_probability_ai": round(p_ai, 4),
            "real_probability": round(p_real, 4),
            "confidence": round(confidence, 4),
            "reliability": reliability,
            "model_agreement": model_agreement,
            "ood_status": ood_status,
            "is_ood": ood_flag,
            "uncertainty": round(float(uncertainty_score), 4),
            "user_explanation": user_explanation,
            "warnings": human_warnings,
            "decision_reasons": [user_explanation] + [w["message"] for w in human_warnings],
            "thresholds": {
                "high_ai": self.high_ai_thresh,
                "mod_ai": self.mod_ai_thresh,
                "mod_real": self.mod_real_thresh,
                "high_real": self.high_real_thresh,
            }
        }


default_decision_engine = ForensicDecisionEngine()
