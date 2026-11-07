"""
Unit tests for CTRDataPreprocessor and parsing utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import (
    CTRDataPreprocessor,
    TargetEncoder,
    parse_timestamp,
    parse_url,
    parse_user_agent,
)


def test_parse_user_agent():
    # Chrome on Android Mobile
    ua_android = "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36"
    br, dev = parse_user_agent(ua_android)
    assert br == "chrome"
    assert dev == "mobile"

    # Edge on Windows Desktop
    ua_edge = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
    br, dev = parse_user_agent(ua_edge)
    assert br == "edge"
    assert dev == "desktop"

    # Safari on iPhone
    ua_iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1"
    br, dev = parse_user_agent(ua_iphone)
    assert br == "safari"
    assert dev == "mobile"

    # Safari on iPad (Tablet)
    ua_ipad = "Mozilla/5.0 (iPad; CPU OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
    br, dev = parse_user_agent(ua_ipad)
    assert br == "safari"
    assert dev == "tablet"

    # Firefox on Linux
    ua_firefox = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
    )
    br, dev = parse_user_agent(ua_firefox)
    assert br == "firefox"
    assert dev == "desktop"

    # Edge cases
    assert parse_user_agent("") == ("other", "other")
    assert parse_user_agent(None) == ("other", "other")
    assert parse_user_agent("UnknownBot/1.0") == ("other", "other")


def test_parse_url():
    url = "https://news.yahoo.co.jp/articles/12345"
    main, full = parse_url(url)
    assert main == "news.yahoo.co.jp"
    assert full == "news.yahoo.co.jp/articles/12345"

    url_plain = "google.com"
    main, full = parse_url(url_plain)
    assert main == "google.com"
    assert full == "google.com"

    assert parse_url("") == ("", "")
    assert parse_url(None) == ("", "")


def test_parse_timestamp():
    h, w, m = parse_timestamp("2024-04-13 09:27:46")
    assert h == 9
    assert w == 5
    assert m == 4

    h, w, m = parse_timestamp(None)
    assert 0 <= h <= 23
    assert 0 <= w <= 6
    assert 1 <= m <= 12


def test_target_encoder_smoothing():
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "A", "A", "B", "B", "C"],
        }
    )
    y = np.array([1, 1, 1, 0, 0, 0, 1])  # global mean = 4/7 = 0.5714
    global_mean = 4.0 / 7.0

    te = TargetEncoder(cols=["cat"], smoothing=5.0)
    te.fit(df, y)

    # For A: n=4, cat_mean=0.75. smoothed = (4/9)*0.75 + (5/9)*0.5714 = 0.3333 + 0.3175 = 0.6508
    trans = te.transform(pd.DataFrame({"cat": ["A", "B", "UNKNOWN"]}))
    assert len(trans) == 3
    assert trans.iloc[0]["cat"] == pytest.approx(
        (4 / 9) * 0.75 + (5 / 9) * global_mean, rel=1e-3
    )
    # UNKNOWN category should get global mean
    assert trans.iloc[2]["cat"] == pytest.approx(global_mean, rel=1e-3)


def test_preprocessor_end_to_end_and_single_consistency():
    # Sample mock dataset
    records = [
        {
            "ts": "2024-04-13 09:27:46",
            "exchange_id": "14",
            "dimension_id": "250",
            "rtb_url": "https://example.com/ad",
            "ab_fold": "a",
            "ias_viewability": 65.0,
            "google_ctr": 0.08,
            "google_viewability": 77.0,
            "bid_value": 78.0,
            "user_agent": "Mozilla/5.0 (Linux; Android 10; K) Chrome/122.0.0.0 Mobile Safari/537.36",
            "ip": "ip_hash_1",
            "device_type": "1",
            "biz": "home",
            "connection_type": "4g",
            "country": "jp",
            "city": "Tokyo",
            "latitude": 35.68,
            "longitude": 139.69,
            "postal_code": "100-0001",
        },
        {
            "ts": "2024-04-13 10:15:00",
            "exchange_id": "2",
            "dimension_id": "251",
            "rtb_url": "https://news.site.org/world",
            "ab_fold": "b",
            "ias_viewability": 85.0,
            "google_ctr": None,
            "google_viewability": None,
            "bid_value": 27.0,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/122.0.0.0",
            "ip": "ip_hash_2",
            "device_type": "0",
            "biz": "biz",
            "connection_type": "wifi",
            "country": "jp",
            "city": "Osaka",
            "latitude": 34.69,
            "longitude": 135.50,
            "postal_code": "530-0001",
        },
    ]
    y = np.array([1, 0])
    df = pd.DataFrame(records)

    prep = CTRDataPreprocessor(smoothing=5.0, use_oof=False)
    X_mat = prep.fit_transform(df, y)

    assert X_mat.shape[0] == 2
    assert not np.isnan(X_mat).any()

    # Verify single-instance fast transform matches batch transform
    rec0 = records[0]
    single_vec = prep.transform_single(rec0)
    batch_vec = prep.transform(pd.DataFrame([rec0]))

    assert single_vec.shape == (1, X_mat.shape[1])
    assert batch_vec.shape == (1, X_mat.shape[1])
    np.testing.assert_allclose(single_vec, batch_vec, rtol=1e-5, atol=1e-5)
