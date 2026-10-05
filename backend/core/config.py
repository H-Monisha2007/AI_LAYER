import os
from pathlib import Path
from typing import List, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Trust-AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    SECRET_KEY: str = "trustai_secret_key_super_secure_change_in_production"
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./deepforensics.db"

    # Media Storage & Security (stored as comma-separated strings in .env, auto-split below)
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_IMAGE_EXTENSIONS: str = "jpg,jpeg,png,webp,bmp"
    ALLOWED_VIDEO_EXTENSIONS: str = "mp4,avi,mov,mkv,webm"

    # ML & Hardware
    FORCE_CPU: bool = False
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    MODEL_WEIGHTS_DIR: Path = BASE_DIR / "model_weights"

    # MLflow
    MLFLOW_TRACKING_URI: str = "./mlruns"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_image_ext_list(self) -> List[str]:
        return [e.strip().lower() for e in self.ALLOWED_IMAGE_EXTENSIONS.split(",") if e.strip()]

    @property
    def allowed_video_ext_list(self) -> List[str]:
        return [e.strip().lower() for e in self.ALLOWED_VIDEO_EXTENSIONS.split(",") if e.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
