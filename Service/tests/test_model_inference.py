"""
Integration test for loading artifacts and performing model inference.
"""

from pathlib import Path
import joblib
import numpy as np
import pytest
import xgboost as xgb

from src.preprocessor import CTRDataPreprocessor


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def test_artifacts_exist():
    assert (ARTIFACTS_DIR / "preprocessor.joblib").exists()
    assert (ARTIFACTS_DIR / "model.joblib").exists()
    assert (ARTIFACTS_DIR / "metadata.json").exists()


def test_artifact_inference_single_sample():
    preprocessor: CTRDataPreprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    model: xgb.XGBClassifier = joblib.load(ARTIFACTS_DIR / "model.joblib")

    # Sample typical request
    request_data = {
        "ts": "2024-04-13 14:30:00",
        "exchange_id": "14",
        "dimension_id": "250",
        "rtb_url": "https://news.yahoo.co.jp/articles/sample",
        "ab_fold": "a",
        "ias_viewability": 65.0,
        "google_ctr": 0.08,
        "google_viewability": 77.0,
        "bid_value": 78.0,
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 CriOS/122.0.6261.89 Mobile/15E148 Safari/604.1",
        "ip": "hashed_ip_12345",
        "device_type": "1",
        "biz": "home",
        "connection_type": "4g",
        "country": "jp",
        "city": "Tokyo",
        "latitude": 35.69,
        "longitude": 139.63,
        "postal_code": "100-0001",
    }

    vec = preprocessor.transform_single(request_data)
    assert vec.shape == (1, len(preprocessor.feature_names_))
    assert not np.isnan(vec).any()

    # Predict proba
    prob = float(model.predict_proba(vec)[0, 1])
    assert 0.0 <= prob <= 1.0


def test_artifact_inference_with_unseen_and_missing_values():
    preprocessor: CTRDataPreprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    model: xgb.XGBClassifier = joblib.load(ARTIFACTS_DIR / "model.joblib")

    # Request with completely unseen IP, empty URL, unknown device, missing numbers
    unseen_request = {
        "ts": None,
        "exchange_id": "9999",  # unseen
        "dimension_id": "unknown_dim",  # unseen
        "rtb_url": "",
        "ab_fold": "x",
        "ias_viewability": None,
        "google_ctr": None,
        "google_viewability": None,
        "bid_value": None,
        "user_agent": "CustomBot/3.0 (MysteryOS; AlienDevice)",
        "ip": "brand_new_unseen_ip",
        "device_type": "99",
        "biz": "alien",
        "connection_type": "quantum_wifi",
        "country": "mars",
        "city": "Olympus Mons",
        "latitude": None,
        "longitude": None,
        "postal_code": "999-9999",
    }

    vec = preprocessor.transform_single(unseen_request)
    assert vec.shape == (1, len(preprocessor.feature_names_))
    assert not np.isnan(vec).any()

    prob = float(model.predict_proba(vec)[0, 1])
    assert 0.0 <= prob <= 1.0
