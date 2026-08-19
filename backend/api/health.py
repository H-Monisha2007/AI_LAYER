import datetime
import torch
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

    cuda_available = torch.cuda.is_available() and not settings.FORCE_CPU
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
