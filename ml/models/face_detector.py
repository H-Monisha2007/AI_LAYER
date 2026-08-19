"""
Face Detector Abstraction

Supports face detection, tracking, cropping, and alignment.
Uses OpenCV Haar Cascades for lightweight face detection with fallback.
"""
from typing import List, Tuple, Optional
import cv2
import numpy as np


class FaceDetector:
    """
    OpenCV-based face detection and cropping module.
    """

    def __init__(self):
        self.cascade = None
        try:
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.cascade = None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in an image.
        Returns a list of (x, y, w, h) bounding boxes.
        """
        if self.cascade is None or getattr(self.cascade, "empty", lambda: True)():
            return []

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [tuple(f) for f in faces]

    def crop_primary_face(self, image: np.ndarray, margin_pct: float = 0.2) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Crop the largest face detected in the image with a percentage margin.
        If no face is detected, returns the original full image and None.
        """
        boxes = self.detect_faces(image)
        if not boxes:
            return image, None

        # Pick the largest box by area
        boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = boxes_sorted[0]

        # Add margin
        mx = int(w * margin_pct)
        my = int(h * margin_pct)

        H, W = image.shape[:2]
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(W, x + w + mx)
        y2 = min(H, y + h + my)

        cropped = image[y1:y2, x1:x2]
        return cropped, (x1, y1, x2 - x1, y2 - y1)


# Global instance
face_detector = FaceDetector()
