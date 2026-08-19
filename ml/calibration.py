"""
Probability Calibration Engine

Applies Platt Scaling / Temperature Scaling to map raw model logit outputs
into well-calibrated probabilities that faithfully represent true posterior confidence P(AI_GENERATED).
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple


class ProbabilityCalibrator:
    """
    Applies Platt Scaling (logistic calibration) or Temperature Scaling.
    
    Calibrated probability P_calibrated = 1 / (1 + exp(-(A * logit + B)))
    where A (temperature) and B (bias) are fitted on validation data.
    """

    def __init__(self, calibration_path: Optional[str] = None):
        self.temperature = 1.0
        self.bias = 0.0
        self.is_calibrated = False

        if calibration_path and os.path.exists(calibration_path):
            self.load_calibration(calibration_path)

    def load_calibration(self, json_path: str):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            self.temperature = data.get("temperature", 1.0)
            self.bias = data.get("bias", 0.0)
            self.is_calibrated = data.get("is_calibrated", True)
        except Exception:
            self.temperature = 1.0
            self.bias = 0.0
            self.is_calibrated = False

    def calibrate_probability(self, raw_p_ai: float) -> float:
        """
        Calibrates raw probability P(AI) into calibrated P(AI).
        Converts probability to log-odds, applies temperature/bias scaling, and applies sigmoid.
        """
        # Convert probability to log-odds (logit) safely
        p_clamped = np.clip(raw_p_ai, 1e-6, 1.0 - 1e-6)
        logit = np.log(p_clamped / (1.0 - p_clamped))

        # Apply scaling: logit_cal = logit / temperature + bias
        calibrated_logit = (logit / max(self.temperature, 1e-4)) + self.bias

        # Sigmoidal transform
        calibrated_p = 1.0 / (1.0 + np.exp(-calibrated_logit))
        return float(np.clip(calibrated_p, 0.0, 1.0))

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Fits Platt scaling parameters (temperature, bias) on validation predictions.
        """
        from scipy.optimize import minimize

        raw_probs = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(raw_probs / (1.0 - raw_probs))

        def loss_fn(params):
            temp, b = params
            temp = max(temp, 1e-4)
            cal_logits = (logits / temp) + b
            cal_p = 1.0 / (1.0 + np.exp(-cal_logits))
            cal_p = np.clip(cal_p, 1e-6, 1.0 - 1e-6)
            # Binary cross-entropy
            bce = -np.mean(y_true * np.log(cal_p) + (1.0 - y_true) * np.log(1.0 - cal_p))
            return bce

        res = minimize(loss_fn, x0=[1.0, 0.0], bounds=[(0.1, 10.0), (-5.0, 5.0)])
        self.temperature = float(res.x[0])
        self.bias = float(res.x[1])
        self.is_calibrated = True

        return {
            "temperature": round(self.temperature, 4),
            "bias": round(self.bias, 4),
            "bce_loss": round(float(res.fun), 4),
        }


default_calibrator = ProbabilityCalibrator()
