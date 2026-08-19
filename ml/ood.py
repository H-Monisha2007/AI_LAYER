"""
DeepForensics Out-of-Distribution (OOD) & Uncertainty Detection Engine

Detects images that fall outside the detector's reliable operating distribution
(e.g., non-photographic graphics, corrupt/extreme resolution, heavy synthetic filters, domain shifts).

If OOD conditions or severe inter-domain model disagreements occur,
forces the decision engine to output INCONCLUSIVE with an explicit diagnostic reason.
"""
from typing import Dict, Any, List
import numpy as np


class OODDetector:
    """
    Evaluates out-of-distribution indicators:
    1. Inter-domain model disagreement (high variance between RGB, Frequency, Residual models)
    2. Image quality anomalies (extreme darkness, zero variance, extreme noise ratio)
    3. Spatial patch conflict score
    """

    def __init__(self, disagreement_threshold: float = 0.35, min_image_std: float = 2.0):
        self.disagreement_threshold = disagreement_threshold
        self.min_image_std = min_image_std

    def evaluate(
        self,
        domain_scores: Dict[str, float],
        img_np: np.ndarray,
        patch_variance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluates whether an input image is Out-Of-Distribution (OOD).
        """
        is_ood = False
        reasons: List[str] = []

        # 1. Image Quality / Naturalness Sanity Check
        if img_np.ndim == 3:
            gray = np.mean(img_np, axis=2)
        else:
            gray = img_np

        img_std = float(np.std(gray))
        if img_std < self.min_image_std:
            is_ood = True
            reasons.append("EXTREME_LOW_VARIANCE_BLANK_OR_UNIFORM_IMAGE")

        # 2. Inter-domain Model Disagreement Check
        valid_scores = [v for k, v in domain_scores.items() if v is not None and not np.isnan(v)]
        domain_range = 0.0
        domain_var = 0.0

        if len(valid_scores) >= 2:
            domain_range = float(np.max(valid_scores) - np.min(valid_scores))
            domain_var = float(np.var(valid_scores))

            if domain_range > self.disagreement_threshold:
                is_ood = True
                reasons.append(f"INTER_DOMAIN_MODEL_DISAGREEMENT (Max score diff: {domain_range:.2f})")

        # 3. Patch Conflict Check
        if patch_variance > 0.15:
            is_ood = True
            reasons.append(f"HIGH_LOCAL_PATCH_FORENSIC_DISCREPANCY (Variance: {patch_variance:.3f})")

        return {
            "is_ood": is_ood,
            "reasons": reasons,
            "domain_disagreement_range": round(domain_range, 4),
            "domain_variance": round(domain_var, 4),
            "patch_variance": round(patch_variance, 4),
            "image_std": round(img_std, 4),
        }


default_ood_detector = OODDetector()
