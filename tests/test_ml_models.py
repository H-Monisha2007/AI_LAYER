import pytest
import numpy as np
from PIL import Image
import torch

from ml.models.base import BaseForensicModel
from ml.models.efficientnet_rgb import EfficientNetRGBModel
from ml.models.convnext_frequency import ConvNeXtFrequencyModel, compute_dct_spectrum
from ml.models.vit_spatial import ViTSpatialModel
from ml.models.srm_residual import SRMResidualModel, extract_srm_residuals
from ml.models.face_detector import face_detector
from ml.evaluation.evaluator import evaluate_predictions, compute_equal_error_rate
from inference.image.pipeline import ImageInferencePipeline, InferenceNotReadyError
from inference.video.pipeline import VideoInferencePipeline


def test_efficientnet_rgb_initialization():
    model = EfficientNetRGBModel(weights_path=None, device="cpu")
    assert model.device == "cpu"
    assert model.is_ready is False
    assert model.get_info()["domain_type"] == "spatial_rgb"


def test_convnext_frequency_dct_transform():
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    spectrum = compute_dct_spectrum(img_array)
    assert spectrum.shape == (100, 100)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0


def test_vit_spatial_initialization():
    model = ViTSpatialModel(weights_path=None, device="cpu")
    assert model.is_ready is False
    assert model.get_info()["architecture"] == "ViT-Base/16"


def test_srm_residual_extraction():
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    res = extract_srm_residuals(img_array)
    assert res.shape == (100, 100, 3)
    assert res.dtype == np.uint8


def test_face_detector():
    # Test face detector on synthetic image (no faces expected)
    dummy_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    crop, bbox = face_detector.crop_primary_face(dummy_img)
    assert bbox is None
    assert crop.shape == (200, 200, 3)


def test_pipeline_raises_not_ready_when_unweighted():
    rgb_model = EfficientNetRGBModel(weights_path=None, device="cpu")
    freq_model = ConvNeXtFrequencyModel(weights_path=None, device="cpu")
    
    pipeline = ImageInferencePipeline(rgb_model=rgb_model, frequency_model=freq_model)
    assert len(pipeline.active_domains) == 0

    with pytest.raises(InferenceNotReadyError):
        pipeline.run("dummy_path.jpg")


def test_evaluation_metrics_calculator():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    
    metrics = evaluate_predictions(y_true, y_probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert "confusion_matrix" in metrics
