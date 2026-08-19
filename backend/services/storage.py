import os
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from fastapi import UploadFile, HTTPException, status
from backend.core.config import settings
from backend.core.logging import logger

class MediaStorageManager:
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or settings.UPLOAD_DIR
        self.images_dir = self.upload_dir / "images"
        self.videos_dir = self.upload_dir / "videos"
        self.frames_dir = self.upload_dir / "frames"
        self.explanations_dir = self.upload_dir / "explanations"
        self.temp_dir = self.upload_dir / "temp"

        self._ensure_directories()

    def _ensure_directories(self):
        for directory in [self.upload_dir, self.images_dir, self.videos_dir, self.frames_dir, self.explanations_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_file(self, file: UploadFile, expected_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Validates uploaded file MIME type, extension, size, and sanitizes file name.
        Prevents path traversal and unsafe execution.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")

        # Sanitize filename (prevent path traversal)
        safe_filename = Path(file.filename).name
        ext = safe_filename.split(".")[-1].lower() if "." in safe_filename else ""

        is_image = ext in settings.allowed_image_ext_list
        is_video = ext in settings.allowed_video_ext_list

        if not (is_image or is_video):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '.{ext}'. Allowed images: {settings.allowed_image_ext_list}, videos: {settings.allowed_video_ext_list}"
            )

        detected_type = "image" if is_image else "video"

        if expected_type and detected_type != expected_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file is a {detected_type}, but {expected_type} was expected."
            )

        return ext, detected_type

    async def save_uploaded_file(self, file: UploadFile, media_type: str) -> Dict[str, Any]:
        """
        Saves uploaded file securely to target storage directory and calculates SHA256 hash.
        """
        ext, _ = self.validate_file(file, expected_type=media_type)
        
        file_uuid = str(uuid.uuid4())
        stored_filename = f"{file_uuid}.{ext}"
        target_dir = self.images_dir if media_type == "image" else self.videos_dir
        target_path = target_dir / stored_filename

        sha256_hash = hashlib.sha256()
        file_size = 0
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        try:
            with open(target_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    file_size += len(chunk)
                    if file_size > max_bytes:
                        # Clean up partial file
                        buffer.close()
                        if target_path.exists():
                            target_path.unlink()
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
                        )
                    sha256_hash.update(chunk)
                    buffer.write(chunk)
        except Exception as e:
            if target_path.exists():
                target_path.unlink()
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file securely.")

        logger.info(f"Saved {media_type} file {stored_filename} ({file_size} bytes, SHA256: {sha256_hash.hexdigest()[:8]}...)")

        return {
            "id": file_uuid,
            "filename": stored_filename,
            "original_filename": Path(file.filename).name,
            "file_path": str(target_path),
            "file_size_bytes": file_size,
            "file_hash_sha256": sha256_hash.hexdigest(),
            "media_type": media_type,
            "mime_type": file.content_type or f"{media_type}/{ext}"
        }

    def cleanup_temp_files(self):
        """Clean up temporary execution files."""
        for item in self.temp_dir.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Could not delete temp file {item}: {e}")

storage_manager = MediaStorageManager()
