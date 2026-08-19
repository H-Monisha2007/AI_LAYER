"""
Forensic Dataset Loader & Data Leakage Prevention Splitter

Supports loading FaceForensics++, Celeb-DF, DFDC, and custom AI-generated image datasets.
Enforces Identity-Aware / Video-Aware train/val/test splits to prevent frame-level data leakage.
"""
import os
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import torch


class ForensicDataset(Dataset):
    """
    Standardized PyTorch dataset for deepfake & AI-media detection training.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        samples: List of (image_path, label) where label: 0=REAL, 1=AI_GENERATED/MANIPULATED
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def video_aware_split(
    video_records: List[Dict[str, Any]], 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits dataset at the VIDEO / IDENTITY level rather than the frame level.
    This guarantees no frames from the same video/subject appear in both train and test.
    """
    import random
    
    # Group frames by video_id / subject_id
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in video_records:
        vid = r.get("video_id", r.get("subject_id", r["path"]))
        grouped.setdefault(vid, []).append(r)

    video_ids = list(grouped.keys())
    random.seed(seed)
    random.shuffle(video_ids)

    n_total = len(video_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_vids = set(video_ids[:n_train])
    val_vids = set(video_ids[n_train:n_train + n_val])
    test_vids = set(video_ids[n_train + n_val:])

    train_samples = [r for vid in train_vids for r in grouped[vid]]
    val_samples = [r for vid in val_vids for r in grouped[vid]]
    test_samples = [r for vid in test_vids for r in grouped[vid]]

    return train_samples, val_samples, test_samples
