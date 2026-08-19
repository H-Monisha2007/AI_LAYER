"""
Frequency Domain Preprocessing + ConvNeXt-Tiny Model Adapter

Domain: Frequency spectrum (DCT/FFT) analysis
Architecture: ConvNeXt-Tiny
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import numpy as np
from scipy.fftpack import dct

from ml.models.base import BaseForensicModel


def compute_dct_spectrum(img_array: np.ndarray) -> np.ndarray:
    """
    Compute 2D DCT spectrum of a grayscale image.
    GAN generators and diffusion models leave distinct DCT artifacts
    (periodic checkerboard patterns from transposed convolutions).
    """
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(np.float32)
    
    # Apply 2D DCT (separable row/column DCT-II)
    dct_rows = dct(gray.astype(np.float64), axis=1, norm='ortho')
    dct_2d = dct(dct_rows, axis=0, norm='ortho')
    
    # Log-magnitude for numerical stability
    magnitude = np.log(np.abs(dct_2d) + 1e-8)
    
    # Normalize to [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)


class ConvNeXtFrequencyModel(BaseForensicModel):
    """
    ConvNeXt-Tiny model operating on DCT frequency spectra.
    
    AI-generated images have distinctive DCT artifacts invisible to human eyes:
    - GANs: periodic peaks from transposed convolution upsampling
    - Diffusion models: different spectral noise profiles
    This model learns to detect these frequency-space signatures.
    """

    UNCERTAIN_THRESHOLD = 0.15

    def __init__(self, weights_path=None, device=None, num_classes=2):
        super().__init__(weights_path=weights_path, device=device)
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def build_model(self) -> nn.Module:
        model = convnext_tiny(weights=None)
        # Adjust first conv layer to accept single-channel DCT input
        original_first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, original_first.out_channels,
            kernel_size=original_first.kernel_size,
            stride=original_first.stride,
            padding=original_first.padding
        )
        # Replace head
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, self.num_classes)
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

        dct_map = compute_dct_spectrum(img)
        pil_dct = Image.fromarray((dct_map * 255).astype(np.uint8))
        return self.transform(pil_dct).unsqueeze(0)

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
            "domain": "frequency_dct",
            "model": "ConvNeXt-Tiny"
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            **super().get_info(),
            "architecture": "ConvNeXt-Tiny",
            "domain_type": "frequency",
            "preprocessing": "2D-DCT log-magnitude spectrum",
            "input_size": (224, 224),
        }
