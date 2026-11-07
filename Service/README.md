# High-Performance CTR Inference Microservice in Real-Time Bidding

Production-ready ML microservice on **FastAPI**, **Pydantic v2**, and **XGBoost (CPU)** for sub-millisecond click-through rate (CTR) prediction and real-time bid recommendations in programmatic RTB ad auctions.

---

## Metrics and inference times

The model is trained on 314'183 banner impressions in RTB-auctions (class disbalance around ~1.71% clicks):

| Metric | Value|
| :--- | :---: |
| **ROC-AUC** | **0.9391** |
| **PR-AUC** | **0.7434** |
| **Best F1-Score** | **0.7730** |
| **Best Precision @ F1** | **86.46%** | 
| **Recall @ F1** | **69.89%** |

### Inference times:
* **Feature preprocessing (`transform_single`):** p50 = **0.43 ms**, p99 = **0.84 ms**
* **XGBoost inference (CPU):** p50 = **0.27 ms**, p99 = **0.60 ms**
* **Overall time:** p50 = **0.71 ms**, p95 = **1.16 ms**, p99 = **1.37 ms**

---

## API

### `GET /health`
Health check and readiness probe.
```json
{
  "status": "ok",
  "ready": true,
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2026-11-06T15:25:00.123456"
}
```

### `GET /info`
Returns model metadata, training parameters, metrics and target CTR simulation table.

### `POST /api/v1/predict`
Single sample bid scoring in real time.

**Example request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "auction_id": "auc_1001",
       "ts": "2024-04-13 09:27:46",
       "exchange_id": "14",
       "dimension_id": "250",
       "rtb_url": "https://news.yahoo.co.jp/articles/sample",
       "ab_fold": "a",
       "ias_viewability": 65.0,
       "google_ctr": 0.087,
       "google_viewability": 77.0,
       "bid_value": 78.0,
       "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) Mobile/15E148 Safari/604.1",
       "ip": "hashed_ip_1001",
       "device_type": "1",
       "biz": "home",
       "connection_type": "4g",
       "country": "jp",
       "city": "Tokyo"
     }'
```

**Example responce (200 OK):**
```json
{
  "auction_id": "auc_1001",
  "p_ctr": 0.04182,
  "bid": false,
  "threshold": 0.38208,
  "latency_ms": 0.68
}
```
*Responce header:* `X-Process-Time-Ms: 0.95`

### `POST /api/v1/predict/batch`
Batch scoring (at most 1000 impressions in single HTTP-request) with vectorization.

---

## Quick Start

### Option 1. Locally via Python / Uvicorn

```bash
pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
Documentation will be at [http://localhost:8000/docs](http://localhost:8000/docs).

### Option 2. Docker

```bash
docker build -t ctr-rtb-service:latest .
docker run -d -p 8000:8000 --name ctr_service ctr-rtb-service:latest
docker inspect --format='{{json .State.Health.Status}}' ctr_service
```

Or through Docker Compose:
```bash
docker compose up --build -d
```
