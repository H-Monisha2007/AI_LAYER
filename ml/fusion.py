"""
Score Fusion Engine

Combines individual domain probability scores (RGB Spatial, Frequency DCT, Noise Residual, Face Analysis)
using a calibrated weighted ensemble or learned meta-classifier.

Handles missing/optional domain scores (e.g. face analysis when no face is present).
"""
import numpy as np
from typing import Dict, Any, Optional, List


class ScoreFusionEngine:
    """
    Multidomain forensic score fusion.
    
    Default weights are derived from domain performance validation:
    - RGB Spatial: 0.35
    - Frequency DCT: 0.30
    - Noise Residual: 0.25
    - Face Analysis: 0.10 (redistributed dynamically when face is N/A)
    """

    DEFAULT_WEIGHTS = {
        "rgb": 0.35,
        "frequency": 0.30,
        "residual": 0.25,
        "face": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def fuse_domain_scores(self, domain_scores: Dict[str, Optional[float]]) -> Dict[str, float]:
        """
        Fuses available domain P(AI_GENERATED) scores into a single combined P(AI_GENERATED).
        
        domain_scores: dict mapping domain key -> float score P(AI) or None/N/A
        """
        available_scores = {}
        available_weights = {}

        for domain, score in domain_scores.items():
            if score is not None and not np.isnan(score):
                available_scores[domain] = float(score)
                available_weights[domain] = self.weights.get(domain, 0.25)

        if not available_scores:
            # Zero ready models -> Return 0.5 (Uncertain baseline)
            return {
                "ai_probability": 0.5,
                "real_probability": 0.5,
                "uncertainty": 1.0,
            }

        # Normalize weights for available domains
        weight_sum = sum(available_weights.values())
        norm_weights = {k: v / weight_sum for k, v in available_weights.items()}

        fused_ai_prob = sum(norm_weights[k] * available_scores[k] for k in available_scores)
        fused_ai_prob = float(np.clip(fused_ai_prob, 0.0, 1.0))
        fused_real_prob = float(np.clip(1.0 - fused_ai_prob, 0.0, 1.0))

        # Calculate score variance across domains as an indicator of uncertainty/disagreement
        scores_list = list(available_scores.values())
        if len(scores_list) > 1:
            variance = float(np.var(scores_list))
            # High domain variance indicates conflicting evidence across spatial vs frequency domains
            uncertainty = float(np.clip(variance * 2.0 + abs(fused_ai_prob - 0.5) * -0.5 + 0.25, 0.0, 1.0))
        else:
            uncertainty = float(np.clip(1.0 - abs(fused_ai_prob - 0.5) * 2, 0.0, 1.0))

        return {
            "ai_probability": round(fused_ai_prob, 4),
            "real_probability": round(fused_real_prob, 4),
            "uncertainty": round(uncertainty, 4),
            "normalized_weights": {k: round(v, 4) for k, v in norm_weights.items()}
        }


default_fusion_engine = ScoreFusionEngine()
