import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from backend.core.config import settings
from backend.core.logging import logger
from backend.db.base import Base
from backend.db.models import ModelVersion

# Ensure SQLite async driver format if standard sqlite is passed
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite://"):
    db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")

engine = create_async_engine(
    db_url,
    echo=settings.DEBUG,
    future=True
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def init_db():
    logger.info("Initializing database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")

    # Seed model versions if table is empty
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ModelVersion))
        existing = result.scalars().all()
        if not existing:
            default_models = [
                ModelVersion(
                    id="effnet-b4-rgb",
                    name="EfficientNet-B4 RGB Spatial Classifier",
                    version="1.0.0",
                    domain_type="spatial_rgb",
                    architecture="EfficientNet-B4",
                    weights_path=os.path.join(settings.MODEL_WEIGHTS_DIR, "efficientnet_b4", "best.pt"),
                    accuracy=0.9091,
                    roc_auc=0.9333,
                    eer=0.0909,
                    is_active=True
                ),
                ModelVersion(
                    id="convnext-dct-freq",
                    name="ConvNeXt Frequency DCT Analyzer",
                    version="1.0.0",
                    domain_type="frequency",
                    architecture="ConvNeXt-Tiny",
                    weights_path=os.path.join(settings.MODEL_WEIGHTS_DIR, "convnext_dct", "best.pt"),
                    accuracy=0.7778,
                    roc_auc=0.9444,
                    eer=0.1667,
                    is_active=True
                ),
                ModelVersion(
                    id="srm-noise-residual",
                    name="SRM High-Pass Noise Residual Model",
                    version="1.0.0",
                    domain_type="residual",
                    architecture="SRM-ResNet18",
                    weights_path=os.path.join(settings.MODEL_WEIGHTS_DIR, "noise_model", "best.pt"),
                    accuracy=0.3333,
                    roc_auc=0.5000,
                    eer=0.5000,
                    is_active=True
                ),
                ModelVersion(
                    id="opencv-face-detector",
                    name="Haar Cascade Face Region Alignment",
                    version="1.0.0",
                    domain_type="face_analysis",
                    architecture="Haar-Cascade-Detector",
                    weights_path="haarcascade_frontalface_default.xml",
                    accuracy=None,
                    roc_auc=None,
                    eer=None,
                    is_active=True
                ),
            ]
            for m in default_models:
                session.add(m)
            await session.commit()
            logger.info("Seeded 4 default ModelVersion records.")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
