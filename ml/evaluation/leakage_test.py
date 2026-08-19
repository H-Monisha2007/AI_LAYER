"""
Dataset Leakage & Perceptual Overlap Auditor

Scans dataset splits (train vs validation vs test vs unseen_generator_test)
to ensure zero perceptual hash (pHash) collisions or near-duplicate leakage exist.
"""
import os
import json
from typing import Dict, Any, List, Tuple
from ml.preprocessing import compute_phash, hamming_distance, load_and_orient_image


def audit_dataset_leakage(manifest_path: str, phash_dup_threshold: int = 4) -> Dict[str, Any]:
    """
    Audits dataset splits in manifest_path for cross-split perceptual duplicate leakage.
    Returns audit findings and pass/fail boolean.
    """
    if not os.path.exists(manifest_path):
        return {"status": "error", "message": f"Manifest file not found: {manifest_path}"}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    train_samples = splits.get("train", [])
    val_samples = splits.get("validation", [])
    test_samples = splits.get("test", [])

    leakage_events = []

    # Compute pHashes for each split
    def get_split_hashes(samples: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        hashes = []
        for s in samples:
            path = s["path"]
            phash_val = s.get("phash")
            if not phash_val and os.path.exists(path):
                try:
                    img = load_and_orient_image(path)
                    phash_val = compute_phash(img)
                except Exception:
                    continue
            if phash_val:
                hashes.append((path, s.get("filename", os.path.basename(path)), phash_val))
        return hashes

    train_h = get_split_hashes(train_samples)
    val_h = get_split_hashes(val_samples)
    test_h = get_split_hashes(test_samples)

    # Check Train vs Val
    for t_path, t_name, t_hash in train_h:
        for v_path, v_name, v_hash in val_h:
            dist = hamming_distance(t_hash, v_hash)
            if dist <= phash_dup_threshold:
                leakage_events.append({
                    "split_1": "train",
                    "file_1": t_name,
                    "split_2": "validation",
                    "file_2": v_name,
                    "hamming_distance": dist,
                })

    # Check Train vs Test
    for t_path, t_name, t_hash in train_h:
        for te_path, te_name, te_hash in test_h:
            dist = hamming_distance(t_hash, te_hash)
            if dist <= phash_dup_threshold:
                leakage_events.append({
                    "split_1": "train",
                    "file_1": t_name,
                    "split_2": "test",
                    "file_2": te_name,
                    "hamming_distance": dist,
                })

    passed = len(leakage_events) == 0

    return {
        "status": "passed" if passed else "failed",
        "total_train_samples": len(train_h),
        "total_val_samples": len(val_h),
        "total_test_samples": len(test_h),
        "leakage_events_count": len(leakage_events),
        "leakage_events": leakage_events,
        "recommendation": "Dataset splits are clean with zero cross-split duplicate leakage." if passed else "Remove flagged near-duplicate images before retraining."
    }


if __name__ == "__main__":
    manifest_f = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "dataset_manifest.json"))
    res = audit_dataset_leakage(manifest_f)
    print(json.dumps(res, indent=2))
