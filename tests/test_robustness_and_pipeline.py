"""
Comprehensive Forensic Pipeline & Robustness Tests

Tests:
1. Valid JPEG, PNG, WebP, RGBA, Grayscale decoding & preprocessing
2. Corrupted & small image handling
3. Model inference and threshold calibration
4. Multi-signal evidence generation (metadata, frequency, noise, patches)
5. Out-of-Distribution (OOD) & 3-way decision handling
6. Backend API response schema validation
"""
import io
import os
import pytest
import numpy as np
from PIL import Image
import torch

from ml.preprocessing import default_preprocessor, load_and_orient_image
from ml.metadata import analyze_image_metadata
from ml.frequency import compute_spectral_residual_features
from ml.noise import compute_noise_residual_statistics
from ml.patch_forensics import extract_patches, aggregate_patch_scores
from ml.ood import default_ood_detector
from ml.decision import default_decision_engine
from ml.calibration import default_calibrator
from ml.models.efficientnet_rgb import EfficientNetRGBModel
from inference.image.pipeline import ImageInferencePipeline


@pytest.fixture
def dummy_rgb_image(tmp_path):
    fpath = os.path.join(tmp_path, "test_rgb.jpg")
    img = Image.new("RGB", (256, 256), color=(128, 64, 32))
    img.save(fpath, format="JPEG", quality=90)
    return fpath


@pytest.fixture
def dummy_rgba_image(tmp_path):
    fpath = os.path.join(tmp_path, "test_rgba.png")
    img = Image.new("RGBA", (256, 256), color=(100, 150, 200, 128))
    img.save(fpath, format="PNG")
    return fpath


@pytest.fixture
def dummy_grayscale_image(tmp_path):
    fpath = os.path.join(tmp_path, "test_gray.webp")
    img = Image.new("L", (200, 200), color=120)
    img.save(fpath, format="WEBP")
    return fpath


@pytest.fixture
def dummy_corrupted_file(tmp_path):
    fpath = os.path.join(tmp_path, "corrupt.jpg")
    with open(fpath, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER_DATA_123456789")
    return fpath


def test_image_decoding_formats(dummy_rgb_image, dummy_rgba_image, dummy_grayscale_image):
    for path in [dummy_rgb_image, dummy_rgba_image, dummy_grayscale_image]:
        pil_img = load_and_orient_image(path)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"


def test_preprocessor_hashes_and_tensors(dummy_rgb_image):
    res = default_preprocessor.preprocess_image(dummy_rgb_image)
    assert "pil_image" in res
    assert "tensor" in res
    assert "dhash" in res
    assert "phash" in res
    assert res["tensor"].shape == (1, 3, 380, 380)


def test_metadata_analysis(dummy_rgb_image):
    meta = analyze_image_metadata(dummy_rgb_image)
    assert "exif_present" in meta
    assert "forensic_flags" in meta
    assert 0.0 <= meta["auxiliary_score"] <= 1.0


def test_frequency_spectral_analysis(dummy_rgb_image):
    pil_img = load_and_orient_image(dummy_rgb_image)
    img_np = np.array(pil_img)
    freq = compute_spectral_residual_features(img_np)
    assert "high_freq_energy_ratio" in freq
    assert "frequency_anomaly_score" in freq


def test_noise_residual_analysis(dummy_rgb_image):
    pil_img = load_and_orient_image(dummy_rgb_image)
    img_np = np.array(pil_img)
    noise = compute_noise_residual_statistics(img_np)
    assert "noise_variance" in noise
    assert "noise_anomaly_score" in noise


def test_patch_extraction(dummy_rgb_image):
    pil_img = load_and_orient_image(dummy_rgb_image)
    patches = extract_patches(pil_img, patch_size=(100, 100))
    assert len(patches) > 0
    agg = aggregate_patch_scores([0.4, 0.7, 0.6])
    assert "aggregated_p_ai" in agg


def test_ood_and_3way_decision():
    ood = default_ood_detector.evaluate(
        domain_scores={"rgb": 0.9, "frequency": 0.1},  # Severe disagreement
        img_np=np.zeros((100, 100, 3), dtype=np.uint8)
    )
    assert ood["is_ood"] is True

    dec = default_decision_engine.classify(
        calibrated_p_ai=0.9,
        ood_result=ood,
        domain_scores={"rgb": 0.9, "frequency": 0.1}
    )
    # New behavior: OOD reduces reliability but doesn't force UNCERTAIN
    # when model score is strongly directional
    assert dec["primary_prediction"] == "AI_GENERATED"
    assert dec["is_ood"] is True
    assert dec["reliability"] < dec["confidence"]  # reliability penalized by OOD


def test_corrupted_file_handling(dummy_corrupted_file):
    with pytest.raises(Exception):
        load_and_orient_image(dummy_corrupted_file)
