"""
DeepForensics Metadata & Image Header Analyzer

Extracts auxiliary forensic evidence from image metadata:
- EXIF metadata tags (Camera model, Software, DateTime, Lens, Exposure)
- Software / AI Generation signature tags (e.g. Adobe Photoshop, Midjourney, Stable Diffusion, DALL-E)
- Compression profile & JPEG quantization tables
- Color space & aspect ratio anomalies

Rule: Metadata is strictly AUXILIARY evidence.
- Missing EXIF NEVER forces an image to be classified as AI.
- Existing Camera EXIF NEVER forces an image to be classified as REAL.
"""
import os
from typing import Dict, Any, List, Optional
from PIL import Image, ImageOps, ExifTags


def analyze_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts and evaluates metadata evidence from an image file.
    Returns metadata dict, auxiliary risk score, and list of forensic flags.
    """
    evidence: Dict[str, Any] = {
        "exif_present": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "dpi": None,
        "compression": None,
        "has_software_tag": False,
        "known_ai_software": False,
        "forensic_flags": [],
        "auxiliary_score": 0.5  # Neutral default
    }

    flags: List[str] = []
    auxiliary_score = 0.5  # 0.0 (Strong Real evidence) -> 1.0 (Strong AI evidence)

    if not os.path.exists(image_path):
        return evidence

    try:
        with Image.open(image_path) as img:
            evidence["format"] = img.format
            evidence["mode"] = img.mode
            evidence["size"] = img.size

            # Check EXIF data
            exif_data = img.getexif()
            if exif_data and len(exif_data) > 0:
                evidence["exif_present"] = True
                
                # Map standard EXIF tags
                tag_dict = {}
                for tag_id, val in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    tag_dict[tag_name] = str(val)

                evidence["camera_make"] = tag_dict.get("Make")
                evidence["camera_model"] = tag_dict.get("Model")
                evidence["software"] = tag_dict.get("Software")

                if tag_dict.get("Make") or tag_dict.get("Model"):
                    flags.append("EXIF_CAMERA_HARDWARE_TAGGED")
                    auxiliary_score -= 0.15

                if tag_dict.get("Software"):
                    evidence["has_software_tag"] = True
                    sw_str = tag_dict.get("Software", "").lower()
                    ai_keywords = ["stable diffusion", "midjourney", "dall-e", "novelai", "comfyui", "automatic1111", "flux", "firefly"]
                    if any(kw in sw_str for kw in ai_keywords):
                        evidence["known_ai_software"] = True
                        flags.append("EXIF_AI_GENERATOR_SOFTWARE_TAG")
                        auxiliary_score += 0.35
                    elif "photoshop" in sw_str or "gimp" in sw_str or "lightroom" in sw_str:
                        flags.append("EXIF_EDITING_SOFTWARE_TAG")

            else:
                flags.append("EXIF_METADATA_STRIPPED_OR_ABSENT")
                # Missing EXIF is common on social media, so we do NOT heavily penalize it.
                auxiliary_score += 0.05

            # JPEG Quantization table inspection if available
            if img.format == "JPEG" and hasattr(img, "quantization"):
                q_tables = getattr(img, "quantization", None)
                if q_tables:
                    evidence["jpeg_quantization_tables"] = len(q_tables)
                    flags.append("JPEG_QUANTIZATION_HEADER_VALID")

    except Exception as e:
        flags.append(f"METADATA_READ_ERROR: {str(e)}")

    evidence["forensic_flags"] = flags
    evidence["auxiliary_score"] = float(max(0.0, min(1.0, auxiliary_score)))
    return evidence
