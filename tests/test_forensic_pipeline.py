"""
Comprehensive Test Suite for DeepForensics ML Pipeline & API
"""
import os
import pytest
import numpy as np
from PIL import Image

from ml.preprocessing import ImagePreprocessor, compute_phash, compute_dhash, hamming_distance
from ml.fusion import ScoreFusionEngine
from ml.calibration import ProbabilityCalibrator
from ml.decision import ForensicDecisionEngine
from inference.image.pipeline import ImageInferencePipeline
from ml.evaluation.evaluator import evaluate_predictions, compute_equal_error_rate
from ml.evaluation.leakage_test import audit_dataset_leakage


def test_image_preprocessor():
    preprocessor = ImagePreprocessor()
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    res = preprocessor.preprocess_image(img)

    assert "pil_image" in res
    assert "tensor" in res
    assert res["tensor"].shape == (1, 3, 380, 380)
    assert len(res["phash"]) == 16
    assert len(res["dhash"]) == 16


def test_perceptual_hash_similarity():
    img1 = Image.new("RGB", (256, 256), color=(100, 150, 200))
    img2 = Image.new("RGB", (256, 256), color=(100, 150, 205))  # Slightly different

    hash1 = compute_phash(img1)
    hash2 = compute_phash(img2)
    dist = hamming_distance(hash1, hash2)

    assert dist < 5  # Highly similar images have small Hamming distance


def test_score_fusion():
    fusion = ScoreFusionEngine()
    domain_scores = {
        "rgb": 0.85,
        "frequency": 0.90,
        "residual": 0.80,
        "face": None,  # No face detected
    }
    res = fusion.fuse_domain_scores(domain_scores)

    assert "ai_probability" in res
    assert "real_probability" in res
    assert res["ai_probability"] > 0.70
    assert abs(res["ai_probability"] + res["real_probability"] - 1.0) < 1e-3


def test_probability_calibration():
    calibrator = ProbabilityCalibrator()
    raw_p = 0.80
    calibrated_p = calibrator.calibrate_probability(raw_p)

    assert 0.0 <= calibrated_p <= 1.0


def test_decision_engine():
    decision_engine = ForensicDecisionEngine()

    res_ai = decision_engine.classify(0.85)
    assert res_ai["primary_prediction"] == "AI_GENERATED"

    res_real = decision_engine.classify(0.15)
    assert res_real["primary_prediction"] == "REAL"

    res_unc = decision_engine.classify(0.50)
    assert res_unc["primary_prediction"] == "UNCERTAIN"

    # Verify 5-category verdicts
    high_ai = decision_engine.classify(0.90)
    assert high_ai["verdict_key"] == "HIGH_CHANCE_AI_GENERATED"

    mod_ai = decision_engine.classify(0.65)
    assert mod_ai["verdict_key"] == "MODERATE_CHANCE_AI_GENERATED"

    high_real = decision_engine.classify(0.10)
    assert high_real["verdict_key"] == "HIGH_CHANCE_REAL"

    mod_real = decision_engine.classify(0.35)
    assert mod_real["verdict_key"] == "MODERATE_CHANCE_REAL"


def test_evaluation_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])

    metrics = evaluate_predictions(y_true, y_probs)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["eer"] == 0.0
