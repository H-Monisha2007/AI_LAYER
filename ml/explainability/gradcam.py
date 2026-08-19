"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Generator

Generates activation heatmaps and visual overlay images highlighting
the spatial regions that contributed most strongly to the model's prediction.
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class GradCAM:
    """
    Hook-based Grad-CAM implementation for PyTorch CNN modules.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate a normalized 2D Grad-CAM heatmap array [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check target_layer.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay a 2D Grad-CAM heatmap [0, 1] on top of an RGB original image array.
    """
    H, W = original_img.shape[:2]
    resized_heatmap = cv2.resize(heatmap_2d, (W, H))
    heatmap_uint8 = (resized_heatmap * 255).astype(np.uint8)
    
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
    return overlay
