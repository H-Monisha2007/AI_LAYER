"""
Robustness Testing Suite

Evaluates detection performance under visual transformations & attacks:
- JPEG Compression (Quality 95, 75, 50)
- Resolution Downsampling (50%)
- Gaussian Blur
- Additive Gaussian Noise
- Contrast / Brightness Distortion (Social Media / Screenshot artifact simulation)
"""
import io
import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any, List

from ml.evaluation.evaluator import evaluate_predictions


def apply_transformation(img: Image.Image, transform_name: str) -> Image.Image:
    """Applies target visual transformation to a PIL Image."""
    if transform_name == "original":
        return img.copy()

    elif transform_name == "jpeg_95":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    elif transform_name == "jpeg_75":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    elif transform_name == "jpeg_50":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    elif transform_name == "resize_50":
        w, h = img.size
        small = img.resize((max(32, w // 2), max(32, h // 2)), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)

    elif transform_name == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))

    elif transform_name == "gaussian_noise":
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 15.0, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    elif transform_name == "screenshot_distortion":
        # Simulates screenshotting: contrast boost + slight brightness shift
        enh = ImageEnhance.Contrast(img).enhance(1.2)
        enh = ImageEnhance.Brightness(enh).enhance(0.95)
        return enh

    return img.copy()


def evaluate_robustness(pipeline: Any, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs robustness evaluation across all transformation types.
    """
    transformations = [
        "original",
        "jpeg_95",
        "jpeg_75",
        "jpeg_50",
        "resize_50",
        "gaussian_blur",
        "gaussian_noise",
        "screenshot_distortion",
    ]

    results = {}

    for tr_name in transformations:
        y_true = []
        y_probs = []

        for sample in test_samples:
            path = sample["path"]
            label_int = 0 if sample["label"] == "REAL" else 1

            if not os.path.exists(path):
                continue

            try:
                orig_img = Image.open(path).convert("RGB")
                trans_img = apply_transformation(orig_img, tr_name)

                # Save temporary image for pipeline evaluation
                tmp_dir = os.path.dirname(path)
                tmp_path = os.path.join(tmp_dir, f"robust_temp_{os.path.basename(path)}")
                trans_img.save(tmp_path, quality=95)

                try:
                    res = pipeline.run(tmp_path)
                    ai_prob = res["ai_probability"]
                    y_probs.append(ai_prob)
                    y_true.append(label_int)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception:
                continue

        if y_true and y_probs:
            metrics = evaluate_predictions(np.array(y_true), np.array(y_probs))
            results[tr_name] = metrics
        else:
            results[tr_name] = {"error": "No valid predictions"}

    return results
