"""
TRUST-AI Independent Audit API

Takes a completed DL detection and independently audits whether the prediction
should be trusted. The DL model makes the prediction; TRUST-AI only determines
if that prediction is reliable enough to act upon.

Trust Decision Categories:
- TRUST: Prediction is reliable, all quality checks pass
- REVIEW REQUIRED: Some risk factors detected, human review recommended
- UNSAFE TO AUTOMATE: High risk, prediction should not be acted upon automatically
"""
import uuid
import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any

from backend.db.session import get_db
from backend.db.models import Detection, TrustAudit
from backend.core.logging import logger

router = APIRouter()


# ═══════ Schemas ═══════

class TrustAuditRequest(BaseModel):
    detection_id: str
    safety_mode: str = "standard"  # standard, safety_critical, permissive


class RiskFactor(BaseModel):
    category: str
    severity: str  # low, medium, high, critical
    description: str
    score: float


class AuditEvidenceResponse(BaseModel):
    data_quality_score: float
    ood_risk_score: float
    model_agreement_score: float
    explainability_score: float
    robustness_score: float
    safety_score: float


class TrustAuditResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    detection_id: str
    dl_prediction: str
    dl_confidence: Optional[float] = None
    trust_decision: str  # TRUST, REVIEW_REQUIRED, UNSAFE_TO_AUTOMATE
    trust_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL

    # Audit Evidence Scores
    data_quality_score: float
    ood_risk_score: float
    model_agreement_score: float
    explainability_score: float
    robustness_score: float
    safety_score: float

    # Risk Factors
    risk_factors: List[RiskFactor] = []
    recommendations: List[str] = []
    audit_summary: str

    # Metadata
    safety_mode: str
    audited_at: str


class TrustAuditHistoryItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    detection_id: str
    dl_prediction: str
    dl_confidence: Optional[float] = None
    trust_decision: str
    trust_score: float
    risk_level: str
    safety_mode: str
    audited_at: str
    original_filename: Optional[str] = None
    media_type: Optional[str] = None


class TrustReportResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_audits: int
    trust_count: int
    review_count: int
    unsafe_count: int
    avg_trust_score: float
    risk_distribution: Dict[str, int]
    recent_audits: List[TrustAuditHistoryItem]
    safety_modes_used: Dict[str, int]


# ═══════ Audit Engine ═══════

class TrustAuditEngine:
    """
    Independent audit engine that evaluates whether a DL prediction
    should be trusted, requires review, or is unsafe to automate.

    This engine does NOT make predictions. It only audits the quality,
    reliability, and safety of existing DL predictions.
    """

    # Thresholds for trust decisions
    TRUST_THRESHOLDS = {
        "standard": {"trust": 0.75, "review": 0.45},
        "safety_critical": {"trust": 0.85, "review": 0.60},
        "permissive": {"trust": 0.60, "review": 0.30},
    }

    def audit(self, detection: Detection, safety_mode: str = "standard") -> Dict[str, Any]:
        """Run independent audit on a completed detection."""

        thresholds = self.TRUST_THRESHOLDS.get(safety_mode, self.TRUST_THRESHOLDS["standard"])

        # 1. Data Quality Assessment
        data_quality = self._assess_data_quality(detection)

        # 2. OOD Risk Assessment
        ood_risk = self._assess_ood_risk(detection)

        # 3. Model Agreement Assessment
        model_agreement = self._assess_model_agreement(detection)

        # 4. Explainability Assessment
        explainability = self._assess_explainability(detection)

        # 5. Robustness Assessment
        robustness = self._assess_robustness(detection)

        # 6. Safety Assessment
        safety = self._assess_safety(detection, safety_mode)

        # Compute overall trust score (weighted average)
        weights = {
            "data_quality": 0.15,
            "ood_risk": 0.20,
            "model_agreement": 0.25,
            "explainability": 0.10,
            "robustness": 0.15,
            "safety": 0.15,
        }

        trust_score = (
            weights["data_quality"] * data_quality["score"]
            + weights["ood_risk"] * ood_risk["score"]
            + weights["model_agreement"] * model_agreement["score"]
            + weights["explainability"] * explainability["score"]
            + weights["robustness"] * robustness["score"]
            + weights["safety"] * safety["score"]
        )
        trust_score = round(min(max(trust_score, 0.0), 1.0), 4)

        # Collect risk factors
        risk_factors = []
        for assessment in [data_quality, ood_risk, model_agreement, explainability, robustness, safety]:
            risk_factors.extend(assessment.get("risk_factors", []))

        # Determine trust decision
        if trust_score >= thresholds["trust"] and not any(rf["severity"] == "critical" for rf in risk_factors):
            trust_decision = "TRUST"
            risk_level = "LOW"
        elif trust_score >= thresholds["review"] and not any(rf["severity"] == "critical" for rf in risk_factors):
            trust_decision = "REVIEW_REQUIRED"
            risk_level = "MEDIUM" if trust_score >= 0.55 else "HIGH"
        else:
            trust_decision = "UNSAFE_TO_AUTOMATE"
            risk_level = "HIGH" if trust_score >= 0.30 else "CRITICAL"

        # Override for critical risk factors
        critical_factors = [rf for rf in risk_factors if rf["severity"] == "critical"]
        if critical_factors:
            trust_decision = "UNSAFE_TO_AUTOMATE"
            risk_level = "CRITICAL"

        # Generate recommendations
        recommendations = self._generate_recommendations(trust_decision, risk_factors, safety_mode)

        # Generate audit summary
        audit_summary = self._generate_summary(
            detection, trust_decision, trust_score, risk_level, risk_factors
        )

        return {
            "trust_decision": trust_decision,
            "trust_score": trust_score,
            "risk_level": risk_level,
            "data_quality_score": data_quality["score"],
            "ood_risk_score": ood_risk["score"],
            "model_agreement_score": model_agreement["score"],
            "explainability_score": explainability["score"],
            "robustness_score": robustness["score"],
            "safety_score": safety["score"],
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "audit_summary": audit_summary,
        }

    def _assess_data_quality(self, detection: Detection) -> Dict[str, Any]:
        """Assess the quality of input data that fed the DL prediction."""
        score = 1.0
        risk_factors = []

        # Check if detection was completed successfully
        if detection.status != "completed":
            score = 0.1
            risk_factors.append({
                "category": "Data Quality",
                "severity": "critical",
                "description": f"Detection did not complete successfully (status: {detection.status})",
                "score": 0.1,
            })
            return {"score": score, "risk_factors": risk_factors}

        # Check confidence availability
        if detection.confidence is None:
            score *= 0.4
            risk_factors.append({
                "category": "Data Quality",
                "severity": "high",
                "description": "No confidence score available from DL models",
                "score": 0.4,
            })

        # Check domain score completeness
        available_scores = sum(1 for s in [
            detection.rgb_score, detection.frequency_score,
            detection.residual_score, detection.face_score
        ] if s is not None)

        if available_scores == 0:
            score *= 0.2
            risk_factors.append({
                "category": "Data Quality",
                "severity": "critical",
                "description": "No domain-level scores available",
                "score": 0.2,
            })
        elif available_scores < 2:
            score *= 0.6
            risk_factors.append({
                "category": "Data Quality",
                "severity": "medium",
                "description": f"Only {available_scores}/4 domain models produced scores",
                "score": 0.6,
            })
        elif available_scores < 3:
            score *= 0.85
            risk_factors.append({
                "category": "Data Quality",
                "severity": "low",
                "description": f"Only {available_scores}/4 domain models produced scores",
                "score": 0.85,
            })

        return {"score": round(min(max(score, 0.0), 1.0), 4), "risk_factors": risk_factors}

    def _assess_ood_risk(self, detection: Detection) -> Dict[str, Any]:
        """Assess out-of-distribution risk."""
        score = 0.85  # Default: moderate confidence
        risk_factors = []

        # Check domain score variance (proxy for OOD)
        scores = [s for s in [
            detection.rgb_score, detection.frequency_score,
            detection.residual_score
        ] if s is not None]

        if len(scores) >= 2:
            import numpy as np
            variance = float(np.var(scores))
            score_range = max(scores) - min(scores)

            if score_range > 0.5:
                score = 0.2
                risk_factors.append({
                    "category": "OOD Risk",
                    "severity": "high",
                    "description": f"Extreme domain score divergence (range: {score_range:.2f}), likely OOD input",
                    "score": 0.2,
                })
            elif score_range > 0.35:
                score = 0.45
                risk_factors.append({
                    "category": "OOD Risk",
                    "severity": "medium",
                    "description": f"Significant domain score divergence (range: {score_range:.2f}), possible domain shift",
                    "score": 0.45,
                })
            elif score_range > 0.2:
                score = 0.7
                risk_factors.append({
                    "category": "OOD Risk",
                    "severity": "low",
                    "description": f"Moderate domain score spread (range: {score_range:.2f})",
                    "score": 0.7,
                })
            else:
                score = 0.95

        return {"score": round(score, 4), "risk_factors": risk_factors}

    def _assess_model_agreement(self, detection: Detection) -> Dict[str, Any]:
        """Assess agreement across different model domains."""
        score = 0.8
        risk_factors = []

        scores = [s for s in [
            detection.rgb_score, detection.frequency_score,
            detection.residual_score
        ] if s is not None]

        if len(scores) < 2:
            score = 0.5
            risk_factors.append({
                "category": "Model Agreement",
                "severity": "medium",
                "description": "Insufficient models for agreement analysis",
                "score": 0.5,
            })
            return {"score": score, "risk_factors": risk_factors}

        # Check if all models agree on the direction (above/below 0.5)
        ai_votes = sum(1 for s in scores if s > 0.5)
        real_votes = len(scores) - ai_votes

        agreement_ratio = max(ai_votes, real_votes) / len(scores)

        if agreement_ratio == 1.0:
            score = 0.95
        elif agreement_ratio >= 0.67:
            score = 0.7
            risk_factors.append({
                "category": "Model Agreement",
                "severity": "low",
                "description": f"Partial model disagreement ({ai_votes} AI vs {real_votes} Real)",
                "score": 0.7,
            })
        else:
            score = 0.3
            risk_factors.append({
                "category": "Model Agreement",
                "severity": "high",
                "description": f"Strong model disagreement ({ai_votes} AI vs {real_votes} Real)",
                "score": 0.3,
            })

        return {"score": round(score, 4), "risk_factors": risk_factors}

    def _assess_explainability(self, detection: Detection) -> Dict[str, Any]:
        """Assess whether the prediction has sufficient explainability evidence."""
        score = 0.7
        risk_factors = []

        # Check if we have domain-level breakdowns
        has_rgb = detection.rgb_score is not None
        has_freq = detection.frequency_score is not None
        has_noise = detection.residual_score is not None

        evidence_count = sum([has_rgb, has_freq, has_noise])

        if evidence_count >= 3:
            score = 0.95  # Full explainability
        elif evidence_count >= 2:
            score = 0.75
        elif evidence_count >= 1:
            score = 0.5
            risk_factors.append({
                "category": "Explainability",
                "severity": "medium",
                "description": "Limited forensic evidence available for explanation",
                "score": 0.5,
            })
        else:
            score = 0.2
            risk_factors.append({
                "category": "Explainability",
                "severity": "high",
                "description": "No forensic evidence available to explain prediction",
                "score": 0.2,
            })

        return {"score": round(score, 4), "risk_factors": risk_factors}

    def _assess_robustness(self, detection: Detection) -> Dict[str, Any]:
        """Assess prediction robustness based on confidence and score margins."""
        score = 0.7
        risk_factors = []

        confidence = detection.confidence
        if confidence is None:
            score = 0.3
            risk_factors.append({
                "category": "Robustness",
                "severity": "high",
                "description": "No confidence value available for robustness assessment",
                "score": 0.3,
            })
            return {"score": score, "risk_factors": risk_factors}

        # High-confidence predictions are more robust
        if confidence >= 0.90:
            score = 0.95
        elif confidence >= 0.75:
            score = 0.80
        elif confidence >= 0.60:
            score = 0.60
            risk_factors.append({
                "category": "Robustness",
                "severity": "low",
                "description": f"Moderate confidence ({confidence:.1%}) suggests borderline prediction",
                "score": 0.60,
            })
        elif confidence >= 0.50:
            score = 0.35
            risk_factors.append({
                "category": "Robustness",
                "severity": "medium",
                "description": f"Low confidence ({confidence:.1%}) indicates uncertain prediction",
                "score": 0.35,
            })
        else:
            score = 0.15
            risk_factors.append({
                "category": "Robustness",
                "severity": "high",
                "description": f"Very low confidence ({confidence:.1%}) suggests unreliable prediction",
                "score": 0.15,
            })

        return {"score": round(score, 4), "risk_factors": risk_factors}

    def _assess_safety(self, detection: Detection, safety_mode: str) -> Dict[str, Any]:
        """Assess safety implications of the prediction."""
        score = 0.85
        risk_factors = []

        # In safety_critical mode, require higher standards
        if safety_mode == "safety_critical":
            if detection.confidence and detection.confidence < 0.80:
                score *= 0.6
                risk_factors.append({
                    "category": "Safety",
                    "severity": "high",
                    "description": "Safety-critical mode requires >80% confidence",
                    "score": score,
                })

            if detection.primary_prediction == "UNCERTAIN":
                score = 0.1
                risk_factors.append({
                    "category": "Safety",
                    "severity": "critical",
                    "description": "UNCERTAIN prediction in safety-critical context",
                    "score": 0.1,
                })

        # Check for ambiguous predictions
        if detection.primary_prediction == "UNCERTAIN":
            score *= 0.5
            risk_factors.append({
                "category": "Safety",
                "severity": "medium",
                "description": "DL model produced UNCERTAIN prediction",
                "score": score,
            })

        return {"score": round(min(max(score, 0.0), 1.0), 4), "risk_factors": risk_factors}

    def _generate_recommendations(
        self, decision: str, risk_factors: List[Dict], safety_mode: str
    ) -> List[str]:
        """Generate actionable recommendations based on audit results."""
        recs = []

        if decision == "TRUST":
            recs.append("Prediction meets quality and safety thresholds for automated use.")
            if safety_mode == "safety_critical":
                recs.append("Safety-critical checks passed. Prediction is suitable for high-stakes decisions.")

        elif decision == "REVIEW_REQUIRED":
            recs.append("Human review is recommended before acting on this prediction.")
            high_risks = [rf for rf in risk_factors if rf["severity"] in ("high", "critical")]
            if high_risks:
                recs.append(f"Address {len(high_risks)} high-severity risk factor(s) identified.")
            recs.append("Consider running additional forensic analysis on the input media.")

        else:  # UNSAFE_TO_AUTOMATE
            recs.append("DO NOT use this prediction for automated decision-making.")
            critical = [rf for rf in risk_factors if rf["severity"] == "critical"]
            if critical:
                recs.append(f"{len(critical)} critical risk factor(s) detected. Manual expert review required.")
            recs.append("Reprocess with higher-quality input or additional models if possible.")

        return recs

    def _generate_summary(
        self, detection: Detection, decision: str, trust_score: float,
        risk_level: str, risk_factors: List[Dict]
    ) -> str:
        """Generate a human-readable audit summary."""
        pred = detection.primary_prediction or "UNKNOWN"
        conf = f"{detection.confidence:.1%}" if detection.confidence else "N/A"

        summary_parts = [
            f"TRUST-AI Independent Audit of DL prediction '{pred}' (confidence: {conf}).",
            f"Trust Score: {trust_score:.1%} | Decision: {decision} | Risk Level: {risk_level}.",
        ]

        if risk_factors:
            high_count = sum(1 for rf in risk_factors if rf["severity"] in ("high", "critical"))
            if high_count:
                summary_parts.append(f"{high_count} significant risk factor(s) identified.")

        return " ".join(summary_parts)


# Global engine instance
trust_engine = TrustAuditEngine()


# ═══════ API Endpoints ═══════

@router.post("/trust/audit", response_model=TrustAuditResponse)
async def run_trust_audit(
    request: TrustAuditRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Run an independent TRUST-AI audit on a completed DL detection.
    The DL model makes the prediction; TRUST-AI audits whether it should be trusted.
    """
    logger.info(f"[TRUST-AI] Audit requested for detection: {request.detection_id}")

    # Fetch the detection
    result = await db.execute(
        select(Detection).where(Detection.id == request.detection_id)
    )
    detection = result.scalar_one_or_none()

    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection {request.detection_id} not found"
        )

    # Run independent audit
    audit_result = trust_engine.audit(detection, safety_mode=request.safety_mode)

    # Persist audit record
    audit_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()

    db_audit = TrustAudit(
        id=audit_id,
        detection_id=detection.id,
        dl_prediction=detection.primary_prediction,
        dl_confidence=detection.confidence,
        trust_decision=audit_result["trust_decision"],
        trust_score=audit_result["trust_score"],
        risk_level=audit_result["risk_level"],
        data_quality_score=audit_result["data_quality_score"],
        ood_risk_score=audit_result["ood_risk_score"],
        model_agreement_score=audit_result["model_agreement_score"],
        explainability_score=audit_result["explainability_score"],
        robustness_score=audit_result["robustness_score"],
        safety_score=audit_result["safety_score"],
        risk_factors=audit_result["risk_factors"],
        recommendations=audit_result["recommendations"],
        audit_summary=audit_result["audit_summary"],
        safety_mode=request.safety_mode,
        audited_at=now,
    )
    db.add(db_audit)
    await db.commit()

    logger.info(f"[TRUST-AI] Audit completed: {audit_result['trust_decision']} (score: {audit_result['trust_score']:.2f})")

    return TrustAuditResponse(
        id=audit_id,
        detection_id=detection.id,
        dl_prediction=detection.primary_prediction,
        dl_confidence=detection.confidence,
        trust_decision=audit_result["trust_decision"],
        trust_score=audit_result["trust_score"],
        risk_level=audit_result["risk_level"],
        data_quality_score=audit_result["data_quality_score"],
        ood_risk_score=audit_result["ood_risk_score"],
        model_agreement_score=audit_result["model_agreement_score"],
        explainability_score=audit_result["explainability_score"],
        robustness_score=audit_result["robustness_score"],
        safety_score=audit_result["safety_score"],
        risk_factors=[RiskFactor(**rf) for rf in audit_result["risk_factors"]],
        recommendations=audit_result["recommendations"],
        audit_summary=audit_result["audit_summary"],
        safety_mode=request.safety_mode,
        audited_at=now.isoformat(),
    )


@router.get("/trust/history", response_model=List[TrustAuditHistoryItem])
async def get_trust_audit_history(db: AsyncSession = Depends(get_db)):
    """Get all TRUST-AI audit history."""
    result = await db.execute(
        select(TrustAudit).order_by(desc(TrustAudit.audited_at))
    )
    audits = result.scalars().all()

    items = []
    for a in audits:
        # Try to get the original filename from the associated detection
        det_result = await db.execute(
            select(Detection).where(Detection.id == a.detection_id)
        )
        det = det_result.scalar_one_or_none()

        original_filename = None
        media_type = None
        if det:
            from backend.db.models import MediaFile
            mf_result = await db.execute(
                select(MediaFile).where(MediaFile.id == det.media_file_id)
            )
            mf = mf_result.scalar_one_or_none()
            if mf:
                original_filename = mf.original_filename
                media_type = mf.media_type

        items.append(TrustAuditHistoryItem(
            id=a.id,
            detection_id=a.detection_id,
            dl_prediction=a.dl_prediction,
            dl_confidence=a.dl_confidence,
            trust_decision=a.trust_decision,
            trust_score=a.trust_score,
            risk_level=a.risk_level,
            safety_mode=a.safety_mode,
            audited_at=a.audited_at.isoformat() if a.audited_at else "",
            original_filename=original_filename,
            media_type=media_type,
        ))

    return items


@router.get("/trust/report", response_model=TrustReportResponse)
async def get_trust_report(db: AsyncSession = Depends(get_db)):
    """Generate a TRUST-AI audit report summary."""
    result = await db.execute(
        select(TrustAudit).order_by(desc(TrustAudit.audited_at))
    )
    audits = result.scalars().all()

    total = len(audits)
    trust_count = sum(1 for a in audits if a.trust_decision == "TRUST")
    review_count = sum(1 for a in audits if a.trust_decision == "REVIEW_REQUIRED")
    unsafe_count = sum(1 for a in audits if a.trust_decision == "UNSAFE_TO_AUTOMATE")

    avg_score = sum(a.trust_score for a in audits) / total if total > 0 else 0.0

    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    safety_modes = {}
    for a in audits:
        risk_dist[a.risk_level] = risk_dist.get(a.risk_level, 0) + 1
        safety_modes[a.safety_mode] = safety_modes.get(a.safety_mode, 0) + 1

    # Build recent items
    recent = []
    for a in audits[:20]:
        det_result = await db.execute(
            select(Detection).where(Detection.id == a.detection_id)
        )
        det = det_result.scalar_one_or_none()
        original_filename = None
        media_type = None
        if det:
            from backend.db.models import MediaFile
            mf_result = await db.execute(
                select(MediaFile).where(MediaFile.id == det.media_file_id)
            )
            mf = mf_result.scalar_one_or_none()
            if mf:
                original_filename = mf.original_filename
                media_type = mf.media_type

        recent.append(TrustAuditHistoryItem(
            id=a.id,
            detection_id=a.detection_id,
            dl_prediction=a.dl_prediction,
            dl_confidence=a.dl_confidence,
            trust_decision=a.trust_decision,
            trust_score=a.trust_score,
            risk_level=a.risk_level,
            safety_mode=a.safety_mode,
            audited_at=a.audited_at.isoformat() if a.audited_at else "",
            original_filename=original_filename,
            media_type=media_type,
        ))

    return TrustReportResponse(
        total_audits=total,
        trust_count=trust_count,
        review_count=review_count,
        unsafe_count=unsafe_count,
        avg_trust_score=round(avg_score, 4),
        risk_distribution=risk_dist,
        recent_audits=recent,
        safety_modes_used=safety_modes,
    )
