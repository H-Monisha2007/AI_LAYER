"""
EfficientNet-B4 RGB Spatial Model Adapter

Domain: RGB spatial feature detection
Architecture: EfficientNet-B4 (torchvision)
Binary output: REAL / AI_GENERATED (with UNCERTAIN threshold)
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import numpy as np

from ml.models.base import BaseForensicModel


class EfficientNetRGBModel(BaseForensicModel):
    """
    EfficientNet-B4 classifier for RGB spatial forgery detection.
    
    This model detects AI-generation artifacts in RGB color space:
    texture inconsistencies, color distribution anomalies, blending
    artifacts, and pixel-level noise patterns.
    """

    UNCERTAIN_THRESHOLD = 0.15  # |P(fake) - 0.5| < threshold → UNCERTAIN

    def __init__(self, weights_path=None, device=None, num_classes=2):
        super().__init__(weights_path=weights_path, device=device)
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.Resize((380, 380)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def build_model(self) -> nn.Module:
        # EfficientNet-B4 — replace the head with our binary classifier
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT if self.weights_path is None else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )
        return model

    def preprocess(self, input_data: Any) -> torch.Tensor:
        if isinstance(input_data, str):
            img = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(input_data).convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        return self.transform(img).unsqueeze(0)  # [1, C, H, W]

    def predict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        logits = self.model(tensor)  # [1, 2]
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
            "raw_scores": {
                "REAL": round(float(p_real), 4),
                "AI_GENERATED": round(float(p_fake), 4),
            },
            "domain": "rgb_spatial",
            "model": "EfficientNet-B4"
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "architecture": "EfficientNet-B4",
            "domain_type": "spatial_rgb",
            "input_size": (380, 380),
            "num_classes": self.num_classes,
        }
