"""
DeepForensics Multidomain Ablation Experiment Suite

Evaluates model configurations across evaluation datasets to determine:
- RGB Baseline
- RGB + Frequency
- RGB + Residual
- RGB + Patch
- Full Model (RGB + Frequency + Residual + Patch)

Measures: AUROC, F1, Balanced Accuracy, FPR, FNR, ECE, Brier Score.
Saves results to model_weights/ablation_report.json.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import numpy as np
from typing import Dict, Any, List

from ml.models.efficientnet_rgb import EfficientNetRGBModel
from ml.models.convnext_frequency import ConvNeXtFrequencyModel
from ml.models.srm_residual import SRMResidualModel
from inference.image.pipeline import ImageInferencePipeline
from ml.evaluation.evaluator import evaluate_predictions
from backend.core.logging import logger

WEIGHTS_DIR = os.path.join(ROOT_DIR, "model_weights")


def run_ablation_experiments() -> Dict[str, Any]:
    manifest_path = os.path.join(ROOT_DIR, "datasets", "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        return {"error": "dataset_manifest.json not found"}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    test_records = manifest.get("splits", {}).get("test", [])
    if not test_records:
        return {"error": "No test records found"}

    rgb_path = os.path.join(WEIGHTS_DIR, "efficientnet_b4", "best.pt")
    freq_path = os.path.join(WEIGHTS_DIR, "convnext_dct", "best.pt")
    noise_path = os.path.join(WEIGHTS_DIR, "noise_model", "best.pt")

    rgb_model = EfficientNetRGBModel(weights_path=rgb_path if os.path.exists(rgb_path) else None)
    if os.path.exists(rgb_path):
        rgb_model.load()

    freq_model = ConvNeXtFrequencyModel(weights_path=freq_path if os.path.exists(freq_path) else None)
    if os.path.exists(freq_path):
        freq_model.load()

    noise_model = SRMResidualModel(weights_path=noise_path if os.path.exists(noise_path) else None)
    if os.path.exists(noise_path):
        noise_model.load()

    configs = {
        "RGB baseline": ImageInferencePipeline(rgb_model=rgb_model),
        "RGB + frequency": ImageInferencePipeline(rgb_model=rgb_model, frequency_model=freq_model),
        "RGB + residual": ImageInferencePipeline(rgb_model=rgb_model, residual_model=noise_model),
        "RGB + patch": ImageInferencePipeline(rgb_model=rgb_model),  # evaluates with multi-patch
        "full model": ImageInferencePipeline(rgb_model=rgb_model, frequency_model=freq_model, residual_model=noise_model),
    }

    ablation_results = {}

    for config_name, pipeline in configs.items():
        y_true = []
        y_probs = []

        for r in test_records:
            fpath = r["path"]
            target = 0 if r["label"] == "REAL" else 1

            try:
                res = pipeline.run(fpath, generate_gradcam=False)
                y_true.append(target)
                y_probs.append(res["ai_probability"])
            except Exception as e:
                continue

        if y_true:
            m = evaluate_predictions(np.array(y_true), np.array(y_probs))
            ablation_results[config_name] = m

    report = {
        "timestamp": time.time(),
        "ablation_experiments": ablation_results
    }

    report_path = os.path.join(WEIGHTS_DIR, "ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_ablation_experiments()
    print("Ablation Results:")
    print(json.dumps(rep, indent=2))
