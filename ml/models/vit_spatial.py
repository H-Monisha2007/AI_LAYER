"""
Vision Transformer (ViT-Base/16) Spatial Model Adapter

Domain: Spatial RGB / Global self-attention anomaly detection
Architecture: ViT-Base/16 (torchvision)
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vit_b_16
from PIL import Image
import numpy as np

from ml.models.base import BaseForensicModel


class ViTSpatialModel(BaseForensicModel):
    """
    Vision Transformer (ViT-Base/16) model for spatial forensic detection.
    
    Self-attention mechanisms capture long-range spatial correlations across the entire
    image (e.g. global lighting inconsistencies, eye symmetry, background distortion)
    which localized CNN convolutional kernels may miss.
    """

    UNCERTAIN_THRESHOLD = 0.15

    def __init__(self, weights_path=None, device=None, num_classes=2):
        super().__init__(weights_path=weights_path, device=device)
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def build_model(self) -> nn.Module:
        model = vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, self.num_classes)
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
        return self.transform(img).unsqueeze(0)

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
            "domain": "spatial_vit",
            "model": "ViT-Base/16"
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "architecture": "ViT-Base/16",
            "domain_type": "spatial_rgb",
            "input_size": (224, 224),
        }
