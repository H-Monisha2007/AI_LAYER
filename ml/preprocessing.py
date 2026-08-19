"""
DeepForensics Centralized Image Preprocessing Pipeline

Provides a single, standard preprocessing pipeline shared across:
- Training
- Validation
- Testing
- Inference
- Robustness evaluation

Handles:
- Image decoding (filepath, bytes, PIL Image, numpy array)
- EXIF orientation auto-correction
- RGB color-space conversion
- Dynamic resizing with specified interpolation
- Perceptual hash generation (dHash & pHash) for quality control / duplicate detection
- Normalization and PyTorch tensor conversion
- Corrupted & invalid image error handling
"""
import io
import os
import math
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFile
import torch
import torchvision.transforms as T

# Allow loading truncated images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_and_orient_image(input_data: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
    """
    Decodes input_data into a PIL Image and applies EXIF orientation auto-correction.
    Guarantees RGB color mode.
    """
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Image file not found: {input_data}")
        img = Image.open(input_data)
    elif isinstance(input_data, bytes):
        img = Image.open(io.BytesIO(input_data))
    elif isinstance(input_data, Image.Image):
        img = input_data
    elif isinstance(input_data, np.ndarray):
        img = Image.fromarray(input_data)
    else:
        raise TypeError(f"Unsupported input type for image decoding: {type(input_data)}")

    # Auto-correct EXIF orientation
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Ensure 3-channel RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Computes difference hash (dHash) of an image for fast perceptual similarity & duplicate detection.
    """
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.int32)
    # Compare adjacent pixels in each row
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert boolean array to hex string
    binary_str = "".join("1" if val else "0" for val in diff.flatten())
    hex_str = f"{int(binary_str, 2):0{hash_size * hash_size // 4}x}"
    return hex_str


def compute_phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    Computes perceptual hash (pHash) using 2D DCT for robust duplicate & leakage detection.
    """
    img_size = hash_size * highfreq_factor
    resized = img.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)

    # 2D DCT via scipy.fftpack if available, else standard DCT matrix
    try:
        from scipy.fftpack import dct
        dct_rows = dct(pixels, axis=1, norm='ortho')
        dct_2d = dct(dct_rows, axis=0, norm='ortho')
    except ImportError:
        # Basic DCT-II calculation
        N = img_size
        n = np.arange(N)
        k = n.reshape((N, 1))
        dct_1d = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        dct_2d = np.dot(np.dot(dct_1d, pixels), dct_1d.T)

    # Extract low frequency 8x8 block (excluding DC coefficient at 0,0)
    dct_low = dct_2d[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])
    diff = dct_low > med
    binary_str = "".join("1" if val else "0" for val in diff.flatten())
    return f"{int(binary_str, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculates Hamming distance between two hex hash strings."""
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    return bin(val1 ^ val2).count("1")


class ImagePreprocessor:
    """
    Centralized image preprocessing transformer.
    """

    def __init__(self, target_size: Tuple[int, int] = (380, 380), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.target_size = target_size
        self.mean = mean
        self.std = std

        self.transform = T.Compose([
            T.Resize(self.target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess_image(self, input_data: Union[str, bytes, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Preprocesses an image and returns a dict containing:
        - PIL Image (oriented, RGB)
        - PyTorch tensor [1, 3, H, W]
        - Image dimensions
        - Perceptual hashes (dhash, phash)
        """
        pil_img = load_and_orient_image(input_data)
        width, height = pil_img.size

        # Hashes for quality check & deduplication
        dhash_str = compute_dhash(pil_img)
        phash_str = compute_phash(pil_img)

        # Transform to tensor
        tensor = self.transform(pil_img).unsqueeze(0)  # Batch dimension [1, C, H, W]

        return {
            "pil_image": pil_img,
            "tensor": tensor,
            "width": width,
            "height": height,
            "dhash": dhash_str,
            "phash": phash_str,
        }


# Default global instance
default_preprocessor = ImagePreprocessor()
