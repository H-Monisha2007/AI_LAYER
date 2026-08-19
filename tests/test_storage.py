import pytest
from io import BytesIO
from fastapi import UploadFile, HTTPException
from backend.services.storage import MediaStorageManager

def test_media_storage_validation():
    storage = MediaStorageManager()
    
    # Valid image
    file_valid = UploadFile(filename="sample_test.png", file=BytesIO(b"fake_image_bytes"))
    ext, media_type = storage.validate_file(file_valid)
    assert ext == "png"
    assert media_type == "image"

    # Path traversal attempt
    file_traversal = UploadFile(filename="../../../etc/passwd.jpg", file=BytesIO(b"fake_bytes"))
    ext, media_type = storage.validate_file(file_traversal)
    assert ext == "jpg"

    # Invalid extension
    file_invalid = UploadFile(filename="malicious.exe", file=BytesIO(b"executable"))
    with pytest.raises(HTTPException) as exc_info:
        storage.validate_file(file_invalid)
    assert exc_info.value.status_code == 400
