import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.core.logging import logger
from backend.db.session import init_db
from backend.services.storage import storage_manager

from backend.api.health import router as health_router
from backend.api.detect import router as detect_router
from backend.api.models import router as models_router
from backend.api.metrics import router as metrics_router
from backend.api.detections import router as detections_router
from backend.api.evaluation import router as evaluation_router
from backend.api.trust_audit import router as trust_audit_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} server...")
    storage_manager._ensure_directories()
    await init_db()
    yield
    logger.info(f"Shutting down {settings.APP_NAME} server...")
    storage_manager.cleanup_temp_files()

app = FastAPI(
    title=settings.APP_NAME,
    description="Multidomain Deep Learning Framework for AI-Generated Image & Video Detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Uploads directory for static media inspection
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include Routers
app.include_router(health_router, prefix=settings.API_PREFIX, tags=["Health"])
app.include_router(detect_router, prefix=settings.API_PREFIX, tags=["Detection"])
app.include_router(models_router, prefix=settings.API_PREFIX, tags=["Models"])
app.include_router(metrics_router, prefix=settings.API_PREFIX, tags=["Metrics"])
app.include_router(detections_router, prefix=settings.API_PREFIX, tags=["Detections"])
app.include_router(evaluation_router, prefix=settings.API_PREFIX, tags=["Evaluation"])
app.include_router(trust_audit_router, prefix=settings.API_PREFIX, tags=["Trust Audit"])

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
