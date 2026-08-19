"""
DeepForensics ML Model Adapter Interface

All model adapters must implement this abstract base class.
This ensures models can be swapped without rewriting the application.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import numpy as np


class BaseForensicModel(ABC):
    """
    Abstract base class for all DeepForensics domain models.
    
    Every model adapter (EfficientNet-RGB, ConvNeXt-Freq, ViT-Residual, etc.) 
    must implement this interface for consistent integration into the inference pipeline.
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.model: Optional[torch.nn.Module] = None
        self._is_loaded = False

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Build and return the raw PyTorch model (un-initialized weights)."""
        ...

    @abstractmethod
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Preprocess raw input (numpy array / PIL Image / file path) into a model tensor."""
        ...

    def load(self) -> None:
        """
        Loads the model weights from self.weights_path.
        If no weights path is given, the model runs in random (untrained) mode
        and the caller must handle this by checking is_ready.
        """
        self.model = self.build_model().to(self.device)
        if self.weights_path:
            try:
                state = torch.load(self.weights_path, map_location=self.device)
                # Support both raw state dicts and checkpoint dicts
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to load weights from {self.weights_path}: {e}")
        self.model.eval()

    @property
    def is_ready(self) -> bool:
        """Returns True only if model weights were successfully loaded."""
        return self._is_loaded

    @abstractmethod
    def predict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Run forward pass on a preprocessed tensor.

        Returns a dict with AT MINIMUM:
            {
                "prediction": str,        # REAL, AI_GENERATED, AI_MANIPULATED, UNCERTAIN
                "confidence": float,      # 0.0 – 1.0
                "raw_scores": dict        # class label → probability
            }
        """
        ...

    def predict_from_input(self, input_data: Any) -> Dict[str, Any]:
        """Full pipeline: preprocess → predict."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        tensor = self.preprocess(input_data).to(self.device)
        with torch.no_grad():
            return self.predict(tensor)

    def extract_features(self, tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Optional: extract intermediate feature embeddings for fusion.
        Override in subclasses that support feature extraction.
        """
        return None

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata for the model registry API."""
        return {
            "class": self.__class__.__name__,
            "device": self.device,
            "weights_path": self.weights_path,
            "is_loaded": self._is_loaded,
        }
