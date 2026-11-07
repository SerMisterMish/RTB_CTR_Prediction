"""
FastAPI route handlers for health checks, model info, and CTR predictions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    BatchImpressionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ImpressionRequest,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import settings
from src.services.model_runner import ModelRunner, get_model_runner

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    tags=["Monitoring"],
)
def health_check(runner: ModelRunner = Depends(get_model_runner)) -> HealthResponse:
    """
    Returns readiness status, loaded model version, and current server time.
    Used by Docker HEALTHCHECK and Kubernetes readiness probes.
    """
    ready = runner.is_ready
    version = runner.metadata.get("version", settings.APP_VERSION)
    return HealthResponse(
        status="ok" if ready else "initializing",
        ready=ready,
        model_loaded=ready,
        model_version=version,
        timestamp=datetime.now().isoformat(),
    )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model metadata and performance metrics",
    tags=["Monitoring"],
)
def model_info(runner: ModelRunner = Depends(get_model_runner)) -> ModelInfoResponse:
    """
    Returns details on the loaded model: algorithm, hyperparameters,
    offline test metrics (ROC-AUC, PR-AUC, F1), latency benchmarks, and target CTR KPI table.
    """
    if not runner.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or artifacts are loading.",
        )

    meta = runner.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "CTR_XGBoost_Classifier"),
        version=meta.get("version", settings.APP_VERSION),
        trained_at=meta.get("trained_at", ""),
        algorithm=meta.get("algorithm", "xgboost.XGBClassifier"),
        feature_count=meta.get("dataset_stats", {}).get(
            "feature_count", len(runner.preprocessor.feature_names_)
        ),
        default_bid_threshold=runner.default_threshold,
        metrics=meta.get("metrics", {}),
        dataset_stats=meta.get("dataset_stats", {}),
        latency_benchmarks_ms=meta.get("latency_benchmarks_ms", {}),
        kpi_simulation=meta.get("kpi_simulation", []),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict CTR for a single impression",
    tags=["Inference"],
)
def predict_single(
    request: ImpressionRequest,
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom bid threshold. If omitted, uses trained optimal threshold.",
    ),
    runner: ModelRunner = Depends(get_model_runner),
) -> PredictionResponse:
    """
    Real-time single impression scoring.
    Returns pCTR and boolean bid recommendation.
    """
    if not runner.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized.",
        )
    try:
        return runner.predict_single(request, custom_threshold=threshold)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict clicks for a batch of impressions",
    tags=["Inference"],
)
def predict_batch(
    request: BatchImpressionRequest,
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom bid threshold applied across batch.",
    ),
    runner: ModelRunner = Depends(get_model_runner),
) -> BatchPredictionResponse:
    """
    Vectorized batch scoring for up to 1000 impression opportunities simultaneously.
    """
    if not runner.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized.",
        )
    try:
        return runner.predict_batch(request, custom_threshold=threshold)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        ) from exc
