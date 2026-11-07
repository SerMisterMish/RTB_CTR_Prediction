"""
Tests for Pydantic v2 schemas and validation rules.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchImpressionRequest,
    ImpressionRequest,
    PredictionResponse,
)


def test_valid_impression_request():
    req = ImpressionRequest(
        auction_id="auc_test_01",
        ts="2024-04-13 09:27:46",
        exchange_id="14",
        dimension_id="250",
        rtb_url="https://example.com/ad",
        ab_fold="a",
        ias_viewability=65.0,
        google_ctr=0.087,
        google_viewability=77.0,
        bid_value=78.0,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        ip="hash_123",
        device_type="1",
        biz="home",
        connection_type="4g",
        country="jp",
        city="Tokyo",
        latitude=35.69,
        longitude=139.63,
        postal_code="100-0001",
    )
    assert req.auction_id == "auc_test_01"
    assert req.ias_viewability == 65.0
    assert req.google_ctr == 0.087


def test_invalid_range_validation():
    # google_ctr must be <= 1.0
    with pytest.raises(ValidationError):
        ImpressionRequest(google_ctr=1.5)

    # ias_viewability must be <= 100.0
    with pytest.raises(ValidationError):
        ImpressionRequest(ias_viewability=150.0)

    # latitude must be in [-90, 90]
    with pytest.raises(ValidationError):
        ImpressionRequest(latitude=95.0)

    # longitude must be in [-180, 180]
    with pytest.raises(ValidationError):
        ImpressionRequest(longitude=200.0)


def test_empty_batch_request_fails():
    with pytest.raises(ValidationError):
        BatchImpressionRequest(items=[])


def test_valid_batch_request():
    req = ImpressionRequest(auction_id="auc_1")
    batch = BatchImpressionRequest(items=[req, req])
    assert len(batch.items) == 2


def test_prediction_response_validation():
    resp = PredictionResponse(
        auction_id="auc_1",
        p_ctr=0.0452,
        bid=True,
        threshold=0.02,
        latency_ms=0.65,
    )
    assert resp.p_ctr == 0.0452
    assert resp.bid is True

    # Probability > 1.0 must fail
    with pytest.raises(ValidationError):
        PredictionResponse(
            auction_id="auc_1",
            p_ctr=1.5,
            bid=True,
            threshold=0.02,
            latency_ms=0.65,
        )
