import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from backend.api.schemas import HealthResponse
from backend.core.config import settings
from backend.db.session import get_db

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health(db: AsyncSession = Depends(get_db)):
    db_connected = False

    try:
        await db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    # PyTorch is optional for the health endpoint.
    # The backend must be able to start even when ML
    # dependencies/model weights are unavailable.
    try:
        import torch

        cuda_available = (
            torch.cuda.is_available()
            and not settings.FORCE_CPU
        )
    except ImportError:
        cuda_available = False

    device = "cuda" if cuda_available else "cpu"

    return HealthResponse(
        status="ok" if db_connected else "degraded",
        app_name=settings.APP_NAME,
        version="1.0.0",
        environment=settings.APP_ENV,
        database_connected=db_connected,
        cuda_available=cuda_available,
        device=device,
        timestamp=datetime.datetime.utcnow()
    )