"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["model_loaded"] is True
    assert "1.0.0" in data["model_version"]
    assert "timestamp" in data


def test_info_endpoint(client: TestClient):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "xgboost.XGBClassifier"
    assert "roc_auc" in data["metrics"]
    assert data["metrics"]["roc_auc"] > 0.90
    assert data["feature_count"] == 50
    assert "kpi_simulation" in data


def test_predict_single_endpoint(client: TestClient):
    payload = {
        "auction_id": "auc_live_test_01",
        "ts": "2024-04-13 14:30:00",
        "exchange_id": "14",
        "dimension_id": "250",
        "rtb_url": "https://news.yahoo.co.jp/articles/12345",
        "ab_fold": "a",
        "ias_viewability": 65.0,
        "google_ctr": 0.087,
        "google_viewability": 77.0,
        "bid_value": 78.0,
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) Mobile/15E148 Safari/604.1",
        "ip": "hashed_ip_live_01",
        "device_type": "1",
        "biz": "home",
        "connection_type": "4g",
        "country": "jp",
        "city": "Tokyo",
        "latitude": 35.69,
        "longitude": 139.63,
        "postal_code": "100-0001",
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    # Header check
    assert "X-Process-Time-Ms" in response.headers
    process_time = float(response.headers["X-Process-Time-Ms"])
    assert process_time >= 0.0

    data = response.json()
    assert data["auction_id"] == "auc_live_test_01"
    assert 0.0 <= data["p_ctr"] <= 1.0
    assert isinstance(data["bid"], bool)
    assert data["threshold"] > 0.0
    assert data["latency_ms"] >= 0.0


def test_predict_single_custom_threshold(client: TestClient):
    payload = {
        "auction_id": "auc_threshold_test",
        "bid_value": 50.0,
    }

    # Extremely high threshold -> bid must be False
    res_high = client.post("/api/v1/predict?threshold=0.999", json=payload)
    assert res_high.status_code == 200
    assert res_high.json()["bid"] is False
    assert res_high.json()["threshold"] == 0.999

    # Threshold 0.0 -> bid must be True
    res_low = client.post("/api/v1/predict?threshold=0.0", json=payload)
    assert res_low.status_code == 200
    assert res_low.json()["bid"] is True
    assert res_low.json()["threshold"] == 0.0


def test_predict_validation_error(client: TestClient):
    # Invalid google_ctr > 1.0
    invalid_payload = {
        "google_ctr": 99.0,
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_batch_endpoint(client: TestClient):
    single_item = {
        "auction_id": "auc_batch_1",
        "exchange_id": "14",
        "dimension_id": "250",
        "rtb_url": "https://example.com",
        "bid_value": 45.0,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/122.0.0.0",
        "country": "jp",
    }
    items = [{**single_item, "auction_id": f"auc_batch_{i}"} for i in range(10)]
    batch_payload = {"items": items}

    response = client.post("/api/v1/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 10
    assert len(data["predictions"]) == 10
    assert data["total_latency_ms"] > 0.0

    first_pred = data["predictions"][0]
    assert first_pred["auction_id"] == "auc_batch_0"
    assert 0.0 <= first_pred["p_ctr"] <= 1.0


def test_predict_batch_empty_fails(client: TestClient):
    response = client.post("/api/v1/predict/batch", json={"items": []})
    assert response.status_code == 422


def test_predict_batch_large_100_items(client: TestClient):
    single_item = {
        "exchange_id": "14",
        "dimension_id": "250",
        "rtb_url": "https://example.com/item",
        "bid_value": 35.0,
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X)",
        "country": "jp",
        "city": "Tokyo",
    }
    items = [{**single_item, "auction_id": f"large_batch_{i}"} for i in range(120)]
    response = client.post("/api/v1/predict/batch", json={"items": items})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 120
    assert len(data["predictions"]) == 120
    assert data["total_latency_ms"] > 0.0


def test_performance_smoke_latency(client: TestClient):
    """Smoke test: single request latency must be strictly under 15 ms on average."""
    import time

    payload = {
        "auction_id": "auc_perf_smoke",
        "bid_value": 60.0,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
        "rtb_url": "https://news.yahoo.co.jp",
        "country": "jp",
    }

    latencies = []
    # Warmup
    for _ in range(5):
        client.post("/api/v1/predict", json=payload)

    # 30 requests
    for _ in range(30):
        t0 = time.perf_counter()
        res = client.post("/api/v1/predict", json=payload)
        t1 = time.perf_counter()
        assert res.status_code == 200
        latencies.append((t1 - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    assert avg_ms < 15.0, f"Average latency {avg_ms:.2f} ms exceeded 15 ms limit"
