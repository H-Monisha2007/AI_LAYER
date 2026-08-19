from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from backend.api.schemas import SystemMetricsResponse
from backend.db.session import get_db
from backend.db.models import Detection, ModelVersion, Experiment

router = APIRouter()

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_metrics(db: AsyncSession = Depends(get_db)):
    # Query database for statistics
    total_res = await db.execute(select(func.count(Detection.id)))
    total_detections = total_res.scalar() or 0

    real_res = await db.execute(select(func.count(Detection.id)).where(Detection.primary_prediction == "REAL"))
    real_count = real_res.scalar() or 0

    ai_gen_res = await db.execute(select(func.count(Detection.id)).where(Detection.primary_prediction == "AI_GENERATED"))
    ai_gen_count = ai_gen_res.scalar() or 0

    ai_man_res = await db.execute(select(func.count(Detection.id)).where(Detection.primary_prediction == "AI_MANIPULATED"))
    ai_man_count = ai_man_res.scalar() or 0

    unc_res = await db.execute(select(func.count(Detection.id)).where(Detection.primary_prediction == "UNCERTAIN"))
    uncertain_count = unc_res.scalar() or 0

    avg_time_res = await db.execute(select(func.avg(Detection.processing_time_ms)))
    avg_processing_time_ms = float(avg_time_res.scalar() or 0.0)

    models_res = await db.execute(select(func.count(ModelVersion.id)).where(ModelVersion.is_active == True))
    models_active = models_res.scalar() or 4

    exp_res = await db.execute(select(Experiment).order_by(Experiment.created_at.desc()).limit(5))
    experiments = exp_res.scalars().all()

    latest_evals = [
        {
            "id": exp.id,
            "experiment_name": exp.experiment_name,
            "ablation_config": exp.ablation_config,
            "dataset_name": exp.dataset_name,
            "accuracy": exp.accuracy,
            "precision": exp.precision,
            "recall": exp.recall,
            "f1_score": exp.f1_score,
            "roc_auc": exp.roc_auc,
            "eer": exp.eer
        }
        for exp in experiments
    ]

    return SystemMetricsResponse(
        total_detections=total_detections,
        real_count=real_count,
        ai_generated_count=ai_gen_count,
        ai_manipulated_count=ai_man_count,
        uncertain_count=uncertain_count,
        avg_processing_time_ms=avg_processing_time_ms,
        models_active=models_active,
        latest_evaluations=latest_evals
    )
