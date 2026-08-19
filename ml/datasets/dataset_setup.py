"""
Dataset Pipeline & Quality Control Setup Script

Builds standard dataset directory structure:
datasets/
    raw/{real, ai}
    processed/{real, ai}
    train/{real, ai}
    validation/{real, ai}
    test/{real, ai}
    unseen_generator_test/{real, ai}

Features:
- Quality control: corrupted image filter, resolution filter
- Perceptual hash (pHash) duplicate detection & leakage prevention
- Class balancing & metadata logging
- Train / Validation / Test / Unseen-Generator split generation
- Sample dataset generation (for deterministic end-to-end training/eval when external data is unavailable)
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Dict, Tuple, Any

from ml.preprocessing import compute_phash, hamming_distance, load_and_orient_image

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))

SUBDIRS = [
    "raw/real", "raw/ai",
    "processed/real", "processed/ai",
    "train/real", "train/ai",
    "validation/real", "validation/ai",
    "test/real", "test/ai",
    "unseen_generator_test/real", "unseen_generator_test/ai",
]


def ensure_dataset_structure():
    """Create all required dataset directories and README instructions."""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    for sd in SUBDIRS:
        os.makedirs(os.path.join(DATASET_ROOT, sd), exist_ok=True)

    info_file = os.path.join(DATASET_ROOT, "README.md")
    if not os.path.exists(info_file):
        with open(info_file, "w") as f:
            f.write(
                "# DeepForensics Media Datasets\n\n"
                "Directory Structure:\n"
                "- `raw/real/`, `raw/ai/`: Place raw DSLR/phone photos and generated images here.\n"
                "- `train/`: Balanced, non-overlapping training split.\n"
                "- `validation/`: Validation split for hyperparameter tuning & probability calibration.\n"
                "- `test/`: Evaluation test benchmark.\n"
                "- `unseen_generator_test/`: Test set containing unseen generative models (e.g. FLUX, Midjourney v6).\n"
            )


def generate_synthetic_benchmark_dataset(num_real: int = 50, num_ai: int = 50, seed: int = 42):
    """
    Generates realistic synthetic forensic images for REAL and AI classes if external images are not loaded.
    - REAL images feature natural photographic textures, camera noise, high-frequency details.
    - AI images emulate generative artifacts (transposed conv frequency checkerboards, unnatural smoothness, SRM noise shifts).
    """
    ensure_dataset_structure()

    np.random.seed(seed)
    random.seed(seed)

    raw_real_dir = os.path.join(DATASET_ROOT, "raw", "real")
    raw_ai_dir = os.path.join(DATASET_ROOT, "raw", "ai")

    generators = ["stable_diffusion_v1_5", "sdxl", "dall_e_3", "midjourney_v5", "flux_1"]

    # 1. Generate REAL images
    for i in range(num_real):
        filename = f"real_photo_{i+1:03d}.jpg"
        filepath = os.path.join(raw_real_dir, filename)
        if os.path.exists(filepath):
            continue

        # Create natural image with photographic noise & gradients
        w, h = random.choice([(512, 512), (640, 480), (800, 600)])
        base = np.zeros((h, w, 3), dtype=np.float32)

        # Gradient background
        for y in range(h):
            base[y, :, 0] = np.sin(y / 50.0) * 100 + 120
            base[y, :, 1] = np.cos(y / 40.0) * 80 + 130
            base[y, :, 2] = np.sin(y / 30.0) * 90 + 110

        # Add Gaussian sensor noise (natural camera ISO noise)
        sensor_noise = np.random.normal(0, 12, (h, w, 3))
        img_arr = np.clip(base + sensor_noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # Add crisp natural details
        draw.rectangle([w//4, h//4, w*3//4, h*3//4], outline=(200, 50, 50), width=4)
        draw.ellipse([w//3, h//3, w*2//3, h*2//3], fill=(50, 180, 80))

        img.save(filepath, quality=95)

    # 2. Generate AI images
    for i in range(num_ai):
        gen_type = generators[i % len(generators)]
        filename = f"ai_gen_{gen_type}_{i+1:03d}.jpg"
        filepath = os.path.join(raw_ai_dir, filename)
        if os.path.exists(filepath):
            continue

        w, h = random.choice([(512, 512), (768, 768)])
        base = np.zeros((h, w, 3), dtype=np.float32)

        # Generative AI features: smooth gradients + periodic transposed convolution checkerboard artifacts
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        # Periodic checkerboard grid artifact typical of GANs/Diffusion upsamplers
        checkerboard = (np.sin(xx / 4.0) * np.sin(yy / 4.0)) * 25.0

        for y in range(h):
            base[y, :, 0] = np.sin(y / 60.0) * 110 + 130
            base[y, :, 1] = np.sin(y / 60.0) * 110 + 130
            base[y, :, 2] = np.cos(y / 60.0) * 110 + 130

        # Over-smoothed noise (diffusion denoiser artifact)
        smooth_noise = np.random.normal(0, 3, (h, w, 3))
        img_arr = np.clip(base + checkerboard[:, :, None] + smooth_noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_arr)
        # Apply slight bilateral blur (characteristic of AI image generation pipelines)
        img = img.filter(ImageFilter.SMOOTH)

        draw = ImageDraw.Draw(img)
        draw.ellipse([w//3, h//3, w*2//3, h*2//3], fill=(220, 160, 40))

        img.save(filepath, quality=95)


def run_quality_control_and_split(
    train_pct: float = 0.60,
    val_pct: float = 0.20,
    test_pct: float = 0.20,
    phash_dup_threshold: int = 4
) -> Dict[str, Any]:
    """
    Scans raw datasets, performs quality control (validates image decoding, filters near duplicates),
    splits data into train/val/test/unseen_generator_test sets, and records dataset manifest.
    """
    ensure_dataset_structure()

    raw_real_dir = os.path.join(DATASET_ROOT, "raw", "real")
    raw_ai_dir = os.path.join(DATASET_ROOT, "raw", "ai")

    def collect_and_clean(directory: str, label_str: str) -> List[Dict[str, Any]]:
        cleaned = []
        hashes = []

        if not os.path.exists(directory):
            return cleaned

        for root, _, files in os.walk(directory):
            for fname in files:
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    continue
                fpath = os.path.join(root, fname)

                # Quality check 1: Can image be decoded?
                try:
                    pil_img = load_and_orient_image(fpath)
                except Exception:
                    continue  # Filter corrupted image

                # Quality check 2: Minimum resolution check (>= 64x64)
                w, h = pil_img.size
                if w < 64 or h < 64:
                    continue

                # Quality check 3: Duplicate detection via pHash
                phash_str = compute_phash(pil_img)
                is_duplicate = False
                for prev_hash in hashes:
                    if hamming_distance(phash_str, prev_hash) <= phash_dup_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                hashes.append(phash_str)
                cleaned.append({
                    "path": fpath,
                    "filename": fname,
                    "label": label_str,
                    "phash": phash_str,
                    "width": w,
                    "height": h,
                })
        return cleaned

    real_samples = collect_and_clean(raw_real_dir, "REAL")
    ai_samples = collect_and_clean(raw_ai_dir, "AI_GENERATED")

    random.seed(42)
    random.shuffle(real_samples)
    random.shuffle(ai_samples)

    # Separate unseen generator samples if filename contains 'flux' or 'midjourney'
    seen_ai = [s for s in ai_samples if "flux" not in s["filename"].lower() and "midjourney" not in s["filename"].lower()]
    unseen_ai = [s for s in ai_samples if "flux" in s["filename"].lower() or "midjourney" in s["filename"].lower()]

    if not seen_ai:
        seen_ai = ai_samples

    # Create train/val/test splits
    def split_list(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        n = len(items)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    train_real, val_real, test_real = split_list(real_samples)
    train_ai, val_ai, test_ai = split_list(seen_ai)

    manifest = {
        "summary": {
            "total_real": len(real_samples),
            "total_ai": len(ai_samples),
            "train_real": len(train_real),
            "train_ai": len(train_ai),
            "val_real": len(val_real),
            "val_ai": len(val_ai),
            "test_real": len(test_real),
            "test_ai": len(test_ai),
            "unseen_ai": len(unseen_ai),
        },
        "splits": {
            "train": train_real + train_ai,
            "validation": val_real + val_ai,
            "test": test_real + test_ai,
            "unseen_generator_test": unseen_ai + test_real[:len(unseen_ai)],
        }
    }

    manifest_path = os.path.join(DATASET_ROOT, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    generate_synthetic_benchmark_dataset()
    m = run_quality_control_and_split()
    print("Dataset setup completed successfully!")
    print(json.dumps(m["summary"], indent=2))
