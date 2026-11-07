"""
Feature engineering and preprocessing pipeline for CTR prediction in RTB auctions.
Compatible with scikit-learn Transformers and optimized for sub-millisecond inference.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Precompiled Regexes for Fast Feature Extraction
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"(?:https?://)?([A-Za-z0-9.-]+)/?(.*)")

DEVICE_PATTERN = re.compile(
    r"(?P<tablet>iPad|Android(?!.*Mobile)|Tablet)|"
    r"(?P<mobile>Android.*Mobile|iPhone|iPod|Windows Phone|Mobile.*Safari)|"
    r"(?P<desktop>Windows NT|Macintosh|X11; Linux|CrOS)",
    re.IGNORECASE,
)

# Browser regexes in priority order (Edge and Opera check required before Chrome/Safari)
BROWSER_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"Edg(?:e|A|iOS)?/\d+", re.IGNORECASE), "edge"),
    (re.compile(r"OPR/\d+|Opera", re.IGNORECASE), "opera"),
    (re.compile(r"YJApp", re.IGNORECASE), "yjapp"),
    (re.compile(r"Firefox/\d+|FxiOS/\d+", re.IGNORECASE), "firefox"),
    (re.compile(r"Chrome/\d+|CriOS/\d+", re.IGNORECASE), "chrome"),
    (re.compile(r"Version/\d+.*Safari|Safari/\d+", re.IGNORECASE), "safari"),
]


def parse_user_agent(ua: Optional[str]) -> Tuple[str, str]:
    """
    Parsing of user_agent string into browser and device family.

    Returns:
        (browser, device): e.g. ('chrome', 'mobile') or ('safari', 'desktop')
    """
    if not ua or not isinstance(ua, str):
        return "other", "other"

    # Device detection
    m_dev = DEVICE_PATTERN.search(ua)
    device = "other"
    if m_dev:
        for k, v in m_dev.groupdict().items():
            if v is not None:
                device = k
                break

    # Browser detection
    browser = "other"
    for pat, name in BROWSER_PATTERNS:
        if pat.search(ua):
            browser = name
            break

    return browser, device


def parse_url(url: Optional[str]) -> Tuple[str, str]:
    """
    Extracts url_main (domain) and url_full from rtb_url.

    Returns:
        (url_main, url_full): e.g. ('example.com', 'example.com/path')
    """
    if not url or not isinstance(url, str):
        return "", ""
    m = URL_PATTERN.match(url.strip())
    if m:
        domain, path = m.groups()
        return domain, f"{domain}/{path}" if path else domain
    return "", ""


def parse_timestamp(ts: Any) -> Tuple[int, int, int]:
    """
    Extracts hour (0-23), weekday (0-6) and month (1-12) from timestamp input.
    """
    if ts is None or (not isinstance(ts, pd.Series) and pd.isna(ts)):
        now = datetime.now()
        return now.hour, now.weekday(), now.month

    if isinstance(ts, (datetime, pd.Timestamp)):
        return ts.hour, ts.weekday(), ts.month

    if isinstance(ts, pd.Series):
        ts_ = pd.to_datetime(ts, errors="coerce")
        return ts_.dt.hour, ts_.dt.day_of_week, ts_.dt.month

    if isinstance(ts, str):
        try:
            dt = pd.to_datetime(ts)
            return dt.hour, dt.weekday(), dt.month
        except Exception:
            now = datetime.now()
            return now.hour, now.weekday(), now.month

    now = datetime.now()
    return now.hour, now.weekday(), now.month


# ---------------------------------------------------------------------------
# Target Encoder with smoothing and optional Out-Of-Fold on train
# ---------------------------------------------------------------------------


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed Target Encoder for high-cardinality categorical features.

    Formula:
        encoded = (n / (n + m)) * category_mean + (m / (n + m)) * global_mean,
        where n is the category size.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        smoothing: float = 5.0,
        use_oof: bool = True,
        cv: int = 5,
        random_state: int = 42,
    ):
        self.cols = cols or []
        self.smoothing = smoothing
        self.use_oof = use_oof
        self.cv = cv
        self.random_state = random_state

        self.global_mean_: float = 0.0
        self.encodings_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        y_series = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(y_series.mean())
        m = self.smoothing

        self.encodings_ = {}
        for col in self.cols:
            if col not in X.columns:
                continue
            col_series = X[col].fillna("").astype(str).reset_index(drop=True)
            stats = (
                pd.DataFrame({"cat": col_series, "y": y_series})
                .groupby("cat")["y"]
                .agg(["count", "mean"])
            )
            n = stats["count"]
            cat_mean = stats["mean"]
            smoothed = (n / (n + m)) * cat_mean + (m / (n + m)) * self.global_mean_
            self.encodings_[col] = smoothed.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = pd.DataFrame(index=X.index)
        for col in self.cols:
            mapping = self.encodings_.get(col, {})
            if col in X.columns:
                col_series = X[col].fillna("").astype(str)
                X_out[col] = (
                    col_series.map(mapping).fillna(self.global_mean_).astype(float)
                )
            else:
                X_out[col] = self.global_mean_
        return X_out

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        self.fit(X, y)
        if not self.use_oof:
            return self.transform(X)

        # Out-Of-Fold calculation on training data to prevent target leakage
        y_series = pd.Series(y).reset_index(drop=True)
        m = self.smoothing
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        X_out = pd.DataFrame(index=X.index)
        for col in self.cols:
            if col not in X.columns:
                X_out[col] = self.global_mean_
                continue

            col_series = X[col].fillna("").astype(str).reset_index(drop=True)
            oof_col = pd.Series(index=col_series.index, dtype=float)

            for train_idx, val_idx in kf.split(col_series):
                tr_cats = col_series.iloc[train_idx]
                tr_y = y_series.iloc[train_idx]
                val_cats = col_series.iloc[val_idx]

                fold_global = float(tr_y.mean())
                stats = (
                    pd.DataFrame({"cat": tr_cats, "y": tr_y})
                    .groupby("cat")["y"]
                    .agg(["count", "mean"])
                )
                n = stats["count"]
                cat_mean = stats["mean"]
                fold_smoothed = (
                    (n / (n + m)) * cat_mean + (m / (n + m)) * fold_global
                ).to_dict()

                oof_col.iloc[val_idx] = val_cats.map(fold_smoothed).fillna(fold_global)

            X_out[col] = oof_col.values

        return X_out


# ---------------------------------------------------------------------------
# Complete CTR Data Preprocessor Pipeline
# ---------------------------------------------------------------------------


class CTRDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    scikit-learn-compatible transformer for CTR prediction features.

    Transforms raw RTB auction events (dict or DataFrame) into model-ready numeric vectors.
    """

    NUMERICAL_COLS: List[str] = [
        "ias_viewability",
        "google_ctr",
        "google_viewability",
        "bid_value_log",
        "latitude",
        "longitude",
        "hour",
        "month",
    ]

    TARGET_ENCODE_COLS: List[str] = [
        "ip",
        "user_agent",
        "url_main",
        "url_full",
        "city",
        "postal_code",
        "country",
        "connection_type",
    ]

    ONE_HOT_COLS: List[str] = [
        "exchange_id",
        "dimension_id",
        "ab_fold",
        "device_type",
        "biz",
        "ua_browser",
        "ua_device",
    ]

    def __init__(
        self,
        smoothing: float = 5.0,
        use_oof: bool = True,
        cv: int = 5,
        random_state: int = 42,
    ):
        self.smoothing = smoothing
        self.use_oof = use_oof
        self.cv = cv
        self.random_state = random_state

        self.medians_: Dict[str, float] = {}
        self.target_encoder_: Optional[TargetEncoder] = None
        self.ohe_: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []

        # Fast lookup mapping for single-request inference
        self._ohe_lookup_: Dict[str, Dict[str, int]] = {}
        self._n_features_: int = 0

    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses URL, User-Agent, timestamps and computes log-transformed features."""
        df_out = pd.DataFrame(index=df.index)

        # Parse timestamps
        if "hour" in df.columns and "weekday" in df.columns and "month" in df.columns:
            df_out["hour"] = pd.to_numeric(df["hour"], errors="coerce")
            df_out["weekday"] = pd.to_numeric(df["weekday"], errors="coerce")
            df_out["month"] = pd.to_numeric(df["month"], errors="coerce")
        elif "ts" in df.columns:
            df_out["hour"], df_out["weekday"], df_out["month"] = parse_timestamp(
                df["ts"]
            )
        else:
            now = datetime.now()
            df_out["hour"] = now.hour
            df_out["weekday"] = now.weekday()
            df_out["month"] = now.month

        # Parse rtb_url
        if "url_main" in df.columns and "url_full" in df.columns:
            df_out["url_main"] = df["url_main"].fillna("").astype(str)
            df_out["url_full"] = df["url_full"].fillna("").astype(str)
        elif "rtb_url" in df.columns:
            urls = df["rtb_url"].fillna("").astype(str).tolist()
            mains, fulls = [], []
            for u in urls:
                domain, full = parse_url(u)
                mains.append(domain)
                fulls.append(full)
            df_out["url_main"] = mains
            df_out["url_full"] = fulls
        else:
            df_out["url_main"] = ""
            df_out["url_full"] = ""

        # Parse user_agent
        if "ua_browser" in df.columns and "ua_device" in df.columns:
            df_out["ua_browser"] = df["ua_browser"].fillna("other").astype(str)
            df_out["ua_device"] = df["ua_device"].fillna("other").astype(str)
        elif "user_agent" in df.columns:
            uas = df["user_agent"].fillna("").astype(str).tolist()
            browsers, devices = [], []
            for u in uas:
                br, dev = parse_user_agent(u)
                browsers.append(br)
                devices.append(dev)
            df_out["ua_browser"] = browsers
            df_out["ua_device"] = devices
        else:
            df_out["ua_browser"] = "other"
            df_out["ua_device"] = "other"

        # Numerical fields
        for col in [
            "ias_viewability",
            "google_ctr",
            "google_viewability",
            "bid_value",
            "latitude",
            "longitude",
        ]:
            if col in df.columns:
                df_out[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df_out[col] = np.nan

        # Bid value log
        if "bid_value" in df_out.columns:
            df_out["bid_value_log"] = np.log(
                np.maximum(1e-5, df_out["bid_value"].fillna(1.0))
            )
            df_out = df_out.drop("bid_value", axis=1)
        else:
            df_out["bid_value_log"] = 0.0

        # Categorical fields (including user_agent string if present for TE)
        cat_cols = [
            "exchange_id",
            "dimension_id",
            "ab_fold",
            "device_type",
            "biz",
            "connection_type",
            "country",
            "city",
            "postal_code",
            "ip",
            "user_agent",
        ]
        for col in cat_cols:
            if col in df.columns:
                df_out[col] = df[col].fillna("").astype(str)
            else:
                df_out[col] = ""

        return df_out

    def fit(
        self,
        X: Union[pd.DataFrame, List[Dict[str, Any]]],
        y: Union[pd.Series, np.ndarray],
    ):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_feat = self._extract_raw_features(X)

        # 1. Learn medians for numerical columns
        self.medians_ = df_feat[self.NUMERICAL_COLS].median().to_dict()

        # 2. Fit OneHotEncoder for low-cardinality columns
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe_.fit(df_feat[self.ONE_HOT_COLS])
        ohe_names = list(self.ohe_.get_feature_names_out(self.ONE_HOT_COLS))

        # 3. Fit TargetEncoder for high-cardinality columns
        self.target_encoder_ = TargetEncoder(
            cols=self.TARGET_ENCODE_COLS,
            smoothing=self.smoothing,
            use_oof=self.use_oof,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.target_encoder_.fit(df_feat[self.TARGET_ENCODE_COLS], y)

        # 4. Save feature names and ordering
        self.feature_names_ = self.NUMERICAL_COLS + self.TARGET_ENCODE_COLS + ohe_names
        self._n_features_ = len(self.feature_names_)

        # Build fast index lookup for single inference
        self._build_fast_lookup(ohe_names)

        return self

    def _build_fast_lookup(self, ohe_names: List[str]):
        """Creates lookup maps for faster single request inference without pandas."""
        self._ohe_lookup_ = {}
        ohe_offset = len(self.NUMERICAL_COLS) + len(self.TARGET_ENCODE_COLS)

        for col_idx, feat_name in enumerate(ohe_names):
            # feat_name format: '<col_name>_<category>'
            for orig_col in self.ONE_HOT_COLS:
                prefix = f"{orig_col}_"
                if feat_name.startswith(prefix):
                    val = feat_name[len(prefix) :]
                    if orig_col not in self._ohe_lookup_:
                        self._ohe_lookup_[orig_col] = {}
                    self._ohe_lookup_[orig_col][val] = ohe_offset + col_idx
                    break

    def transform(self, X: Union[pd.DataFrame, List[Dict[str, Any]]]) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_feat = self._extract_raw_features(X)

        # Impute numericals
        num_vals = (
            df_feat[self.NUMERICAL_COLS]
            .fillna(self.medians_)
            .to_numpy(dtype=np.float32)
        )

        # Target encode
        te_df = self.target_encoder_.transform(df_feat[self.TARGET_ENCODE_COLS])
        te_vals = te_df.to_numpy(dtype=np.float32)

        # One-hot encode
        ohe_vals = self.ohe_.transform(df_feat[self.ONE_HOT_COLS]).astype(np.float32)

        return np.hstack([num_vals, te_vals, ohe_vals])

    def fit_transform(
        self,
        X: Union[pd.DataFrame, List[Dict[str, Any]]],
        y: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.fit(X, y)
        df_feat = self._extract_raw_features(X)

        # Impute numericals
        num_vals = (
            df_feat[self.NUMERICAL_COLS]
            .fillna(self.medians_)
            .to_numpy(dtype=np.float32)
        )

        # Target encode (uses OOF if use_oof is True)
        te_df = self.target_encoder_.fit_transform(df_feat[self.TARGET_ENCODE_COLS], y)
        te_vals = te_df.to_numpy(dtype=np.float32)

        # One-hot encode
        ohe_vals = self.ohe_.transform(df_feat[self.ONE_HOT_COLS]).astype(np.float32)

        return np.hstack([num_vals, te_vals, ohe_vals])

    def transform_single(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Faster transformation for single instance.
        Constructs the feature vector directly into a numpy array bypassing pandas DataFrame creation.
        """
        vec = np.zeros(self._n_features_, dtype=np.float32)

        # 1. Parse time
        hour, weekday, month = parse_timestamp(record.get("ts"))
        if "hour" in record and record["hour"] is not None:
            hour = int(record["hour"])
        if "weekday" in record and record["weekday"] is not None:
            weekday = int(record["weekday"])
        if "month" in record and record["month"] is not None:
            month = int(record["month"])

        # 2. Parse URL
        rtb_url = record.get("rtb_url") or ""
        url_main, url_full = parse_url(rtb_url)
        if "url_main" in record and record["url_main"]:
            url_main = str(record["url_main"])
        if "url_full" in record and record["url_full"]:
            url_full = str(record["url_full"])

        # 3. Parse User-Agent
        ua_str = record.get("user_agent") or ""
        ua_browser, ua_device = parse_user_agent(ua_str)
        if "ua_browser" in record and record["ua_browser"]:
            ua_browser = str(record["ua_browser"])
        if "ua_device" in record and record["ua_device"]:
            ua_device = str(record["ua_device"])

        # 4. Fill numericals
        def _get_num(col: str) -> float:
            v = record.get(col)
            if v is not None and not np.isnan(float(v)):
                return float(v)
            return self.medians_[col]

        bid_value = record.get("bid_value")
        if bid_value is not None and not np.isnan(float(bid_value)):
            bid_value_log = np.log(np.maximum(1e-5, float(bid_value)))
        else:
            bid_value_log = self.medians_["bid_value_log"]

        num_values = [
            _get_num("ias_viewability"),
            _get_num("google_ctr"),
            _get_num("google_viewability"),
            bid_value_log,
            _get_num("latitude"),
            _get_num("longitude"),
            float(hour),
            float(month),
        ]
        vec[: len(self.NUMERICAL_COLS)] = num_values

        # 5. Fill target-encoded features
        te_offset = len(self.NUMERICAL_COLS)
        te_vals_map = {
            "ip": str(record.get("ip") or ""),
            "user_agent": str(ua_str),
            "url_main": url_main,
            "url_full": url_full,
            "city": str(record.get("city") or ""),
            "postal_code": str(record.get("postal_code") or ""),
            "country": str(record.get("country") or ""),
            "connection_type": str(record.get("connection_type") or ""),
        }
        global_mean = self.target_encoder_.global_mean_
        for idx, col in enumerate(self.TARGET_ENCODE_COLS):
            val_str = te_vals_map.get(col, "")
            encoded_val = self.target_encoder_.encodings_.get(col, {}).get(
                val_str, global_mean
            )
            vec[te_offset + idx] = encoded_val

        # 6. Fill one-hot encoded features
        ohe_vals_map = {
            "exchange_id": str(record.get("exchange_id") or ""),
            "dimension_id": str(record.get("dimension_id") or ""),
            "ab_fold": str(record.get("ab_fold") or ""),
            "device_type": str(record.get("device_type") or ""),
            "biz": str(record.get("biz") or ""),
            "ua_browser": ua_browser,
            "ua_device": ua_device,
        }
        for orig_col, cat_val in ohe_vals_map.items():
            col_map = self._ohe_lookup_.get(orig_col, {})
            out_idx = col_map.get(cat_val)
            if out_idx is not None:
                vec[out_idx] = 1.0

        return vec.reshape(1, -1)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names_)
