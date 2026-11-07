"""
ModelRunner service managing in-memory preprocessor and XGBoost model inference.
Optimized for real-time single-request bidding and high-throughput batching.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import xgboost as xgb

from src.api.schemas import (
    BatchImpressionRequest,
    BatchPredictionResponse,
    ImpressionRequest,
    PredictionResponse,
)
from src.config import settings
from src.preprocessor import CTRDataPreprocessor

logger = logging.getLogger("model_runner")


class ModelRunner:
    """
    In-memory model runner singleton loaded once on application startup.
    """

    def __init__(self):
        self.preprocessor: Optional[CTRDataPreprocessor] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.metadata: Dict[str, Any] = {}
        self.default_threshold: float = settings.BID_THRESHOLD
        self._is_ready: bool = False

    @property
    def is_ready(self) -> bool:
        return (
            self._is_ready and self.model is not None and self.preprocessor is not None
        )

    def load_artifacts(
        self,
        preprocessor_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        """Loads serialized pipeline and model into RAM."""
        p_path = preprocessor_path or settings.PREPROCESSOR_PATH
        m_path = model_path or settings.MODEL_PATH
        meta_path = metadata_path or settings.METADATA_PATH

        logger.info(f"Loading preprocessor from: {p_path}")
        if not p_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {p_path}")
        self.preprocessor = joblib.load(p_path)

        logger.info(f"Loading model from: {m_path}")
        if not m_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {m_path}")
        self.model = joblib.load(m_path)

        if meta_path.exists():
            logger.info(f"Loading metadata from: {meta_path}")
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            # Use default_bid_threshold from metadata if available
            meta_threshold = self.metadata.get("metrics", {}).get(
                "default_bid_threshold"
            )
            if meta_threshold is not None:
                self.default_threshold = float(meta_threshold)

        self._is_ready = True
        logger.info(
            f"Artifacts loaded successfully. Features: {len(self.preprocessor.feature_names_)}, "
            f"Bid Threshold: {self.default_threshold:.4f}"
        )

    def predict_single(
        self,
        impression: ImpressionRequest,
        custom_threshold: Optional[float] = None,
    ) -> PredictionResponse:
        """
        Inference for a single impression bid opportunity.
        """
        if not self.is_ready:
            raise RuntimeError(
                "ModelRunner is not initialized or artifacts are missing."
            )

        threshold = (
            custom_threshold if custom_threshold is not None else self.default_threshold
        )

        t0 = time.perf_counter()
        record_dict = impression.model_dump()
        feature_vec = self.preprocessor.transform_single(record_dict)
        prob = float(self.model.predict_proba(feature_vec)[0, 1])
        t1 = time.perf_counter()

        bid_decision = bool(prob >= threshold)
        latency_ms = (t1 - t0) * 1000.0

        return PredictionResponse(
            auction_id=impression.auction_id,
            p_ctr=round(prob, 5),
            bid=bid_decision,
            threshold=round(threshold, 5),
            latency_ms=round(latency_ms, 3),
        )

    def predict_batch(
        self,
        batch: BatchImpressionRequest,
        custom_threshold: Optional[float] = None,
    ) -> BatchPredictionResponse:
        """
        Vectorized high-throughput batch prediction.
        """
        if not self.is_ready:
            raise RuntimeError(
                "ModelRunner is not initialized or artifacts are missing."
            )

        threshold = (
            custom_threshold if custom_threshold is not None else self.default_threshold
        )

        t0 = time.perf_counter()
        records = [item.model_dump() for item in batch.items]
        feature_matrix = self.preprocessor.transform(records)
        probs = self.model.predict_proba(feature_matrix)[:, 1]
        t1 = time.perf_counter()

        total_latency_ms = (t1 - t0) * 1000.0
        avg_item_latency = total_latency_ms / max(1, len(batch.items))

        predictions = []
        for item, prob in zip(batch.items, probs):
            p_float = float(prob)
            predictions.append(
                PredictionResponse(
                    auction_id=item.auction_id,
                    p_ctr=round(p_float, 5),
                    bid=bool(p_float >= threshold),
                    threshold=round(threshold, 5),
                    latency_ms=round(avg_item_latency, 3),
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            total_latency_ms=round(total_latency_ms, 3),
        )


# Global singleton instance
_model_runner: Optional[ModelRunner] = None


def get_model_runner() -> ModelRunner:
    global _model_runner
    if _model_runner is None:
        _model_runner = ModelRunner()
    return _model_runner
