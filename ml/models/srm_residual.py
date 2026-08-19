"""
SRM Noise Residual Extraction + Classifier Adapter

Domain: High-frequency noise residual analysis (Steganalysis filters)
Architecture: High-Pass SRM Filter Bank + ResNet-18
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import scipy.signal

from ml.models.base import BaseForensicModel

# 3 Basic SRM 5x5 High-Pass Residual Filters
SRM_FILTERS = np.array([
    # 1st order
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    # 2nd order
    [[0, 0, 0, 0, 0],
     [0, -1, 2, -1, 0],
     [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0],
     [0, 0, 0, 0, 0]],
    # 3rd order
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]]
], dtype=np.float32) / 12.0


def extract_srm_residuals(img_array: np.ndarray) -> np.ndarray:
    """
    Extract SRM high-frequency noise residual channels.
    Camera sensor noise follows physical Poisson-Gaussian distributions, whereas
    AI generative pipelines produce unnaturally smooth or correlated residuals.
    """
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2).astype(np.float32)
    else:
        gray = img_array.astype(np.float32)

    channels = []
    for kernel in SRM_FILTERS:
        res = scipy.signal.convolve2d(gray, kernel, mode='same', boundary='symm')
        res = np.clip(res, -255.0, 255.0)
        res = (res + 255.0) / 510.0  # Normalize to [0, 1]
        channels.append(res)

    residual_rgb = np.stack(channels, axis=2)  # [H, W, 3]
    return (residual_rgb * 255).astype(np.uint8)


class SRMResidualModel(BaseForensicModel):
    """
    Model trained on SRM noise residuals to detect generative sensor artifacts.
    """

    UNCERTAIN_THRESHOLD = 0.15

    def __init__(self, weights_path=None, device=None, num_classes=2):
        super().__init__(weights_path=weights_path, device=device)
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def build_model(self) -> nn.Module:
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model

    def preprocess(self, input_data: Any) -> torch.Tensor:
        if isinstance(input_data, str):
            img = np.array(Image.open(input_data).convert("RGB"))
        elif isinstance(input_data, Image.Image):
            img = np.array(input_data.convert("RGB"))
        elif isinstance(input_data, np.ndarray):
            img = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        res_img = extract_srm_residuals(img)
        pil_res = Image.fromarray(res_img)
        return self.transform(pil_res).unsqueeze(0)

    def predict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        p_real = probs[0].item()
        p_fake = probs[1].item()

        if abs(p_fake - 0.5) < self.UNCERTAIN_THRESHOLD:
            prediction = "UNCERTAIN"
            confidence = 1 - abs(p_fake - 0.5) * 2
        elif p_fake > 0.5:
            prediction = "AI_GENERATED"
            confidence = p_fake
        else:
            prediction = "REAL"
            confidence = p_real

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "raw_scores": {"REAL": round(float(p_real), 4), "AI_GENERATED": round(float(p_fake), 4)},
            "domain": "noise_residual",
            "model": "SRM-ResNet18"
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "architecture": "SRM-ResNet18",
            "domain_type": "residual",
            "preprocessing": "3-Filter SRM High-Pass Residual Extraction",
        }
