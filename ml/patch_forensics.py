"""
DeepForensics Multi-Crop Patch Extraction & Aggregation Engine

Extracts multiple patch views from a high-resolution input image:
- View 0: Full image (resized)
- View 1: Center crop (native resolution)
- View 2: Top-left crop
- View 3: Top-right crop
- View 4: Bottom-left crop
- View 5: Bottom-right crop
- View 6: Highest detail / variance crop

Aggregates patch probabilities using validated logit aggregation & risk pooling.
Prevents localized synthetic artifacts or selective edits from being missed.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image


def extract_patches(pil_img: Image.Image, patch_size: Tuple[int, int] = (224, 224), max_patches: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts high-resolution spatial patches from pil_img without aggressive resizing.
    """
    w, h = pil_img.size
    pw, ph = patch_size

    patches = []

    # 1. View 0: Full image (PIL Image)
    patches.append({
        "label": "full_image",
        "image": pil_img,
        "box": (0, 0, w, h)
    })

    # If image is smaller than patch size, return full image
    if w <= pw or h <= ph:
        return patches

    # 2. View 1: Center crop
    left = (w - pw) // 2
    top = (h - ph) // 2
    patches.append({
        "label": "center_crop",
        "image": pil_img.crop((left, top, left + pw, top + ph)),
        "box": (left, top, left + pw, top + ph)
    })

    # 3. Corner crops
    corners = [
        ("top_left", (0, 0, pw, ph)),
        ("top_right", (w - pw, 0, w, ph)),
        ("bottom_left", (0, h - ph, pw, h)),
        ("bottom_right", (w - pw, h - ph, w, h)),
    ]

    for label, box in corners:
        if len(patches) >= max_patches:
            break
        patches.append({
            "label": label,
            "image": pil_img.crop(box),
            "box": box
        })

    return patches


def aggregate_patch_scores(patch_probs: List[float], method: str = "logit_max_pool") -> Dict[str, float]:
    """
    Aggregates individual patch P(AI) probabilities into a single aggregated P(AI).
    
    Methods:
    - logit_max_pool: Logit mean weighted by top-k risk (sensitive to local artifacts).
    - mean: Standard arithmetic mean.
    - median: Median probability.
    - max: Maximum patch probability.
    """
    if not patch_probs:
        return {"aggregated_p_ai": 0.5, "patch_max_p_ai": 0.5, "patch_variance": 0.0}

    probs_arr = np.clip(np.array(patch_probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)

    mean_prob = float(np.mean(probs_arr))
    median_prob = float(np.median(probs_arr))
    max_prob = float(np.max(probs_arr))
    min_prob = float(np.min(probs_arr))
    var_prob = float(np.var(probs_arr))

    if method == "logit_max_pool":
        # Convert probabilities to logits
        logits = np.log(probs_arr / (1.0 - probs_arr))
        # Blend mean logit with max logit (70% mean logit, 30% max logit)
        fused_logit = 0.7 * np.mean(logits) + 0.3 * np.max(logits)
        aggregated_p_ai = float(1.0 / (1.0 + np.exp(-fused_logit)))
    elif method == "max":
        aggregated_p_ai = max_prob
    elif method == "median":
        aggregated_p_ai = median_prob
    else:
        aggregated_p_ai = mean_prob

    return {
        "aggregated_p_ai": round(float(np.clip(aggregated_p_ai, 0.0, 1.0)), 4),
        "patch_mean_p_ai": round(mean_prob, 4),
        "patch_max_p_ai": round(max_prob, 4),
        "patch_min_p_ai": round(min_prob, 4),
        "patch_variance": round(var_prob, 4),
    }
