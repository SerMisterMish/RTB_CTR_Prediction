"""
Pydantic v2 schemas for inference API requests, responses, and health checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ImpressionRequest(BaseModel):
    """
    Single impression bid opportunity from RTB exchange.
    """

    auction_id: Optional[str] = Field(
        default=None,
        description="Unique auction/impression ID for tracing",
        examples=["auc_987654321"],
    )
    ts: Optional[str] = Field(
        default=None,
        description="Timestamp of the impression event (ISO format or YYYY-MM-DD HH:MM:SS)",
        examples=["2024-04-13 09:27:46"],
    )
    exchange_id: Optional[str] = Field(
        default="14",
        description="ID of the ad exchange delivering the bid opportunity",
        examples=["14"],
    )
    dimension_id: Optional[str] = Field(
        default="250",
        description="Dimension/format ID of the ad slot",
        examples=["250"],
    )
    rtb_url: Optional[str] = Field(
        default="https://news.yahoo.co.jp/articles/sample",
        description="URL of the page where the banner is to be displayed",
        examples=["https://news.yahoo.co.jp/articles/sample"],
    )
    ab_fold: Optional[str] = Field(
        default="a",
        description="Banner position relative to the fold ('a' = above fold, 'b' = below fold, 'u' = unknown)",
        examples=["a"],
    )
    ias_viewability: Optional[float] = Field(
        default=65.0,
        ge=0.0,
        le=100.0,
        description="Historical IAS viewability percentage (0-100)",
        examples=[65.0],
    )
    google_ctr: Optional[float] = Field(
        default=0.087,
        ge=0.0,
        le=1.0,
        description="Historical Google CTR on the URL and slot (0-1)",
        examples=[0.087],
    )
    google_viewability: Optional[float] = Field(
        default=77.0,
        ge=0.0,
        le=100.0,
        description="Historical Google viewability percentage (0-100)",
        examples=[77.0],
    )
    bid_value: Optional[float] = Field(
        default=78.0,
        ge=0.0,
        description="Bid floor / value in USD cents per 1000 impressions (CPM)",
        examples=[78.0],
    )
    user_agent: Optional[str] = Field(
        default="Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
        description="User-Agent string of the visitor's browser",
        examples=[
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
        ],
    )
    ip: Optional[str] = Field(
        default="ip_hash_sample_123",
        description="Hashed IP address of the user",
        examples=["ip_hash_sample_123"],
    )
    device_type: Optional[str] = Field(
        default="1",
        description="Device type code (e.g. '0'=Desktop, '1'=Mobile, '2'=Tablet)",
        examples=["1"],
    )
    biz: Optional[str] = Field(
        default="home",
        description="Business or home connection type ('home', 'business')",
        examples=["home"],
    )
    connection_type: Optional[str] = Field(
        default="4g",
        description="Network connection type (e.g. 'wifi', '4g', '3g', 'cellular')",
        examples=["4g"],
    )
    country: Optional[str] = Field(
        default="jp",
        description="Two-letter ISO country code",
        examples=["jp"],
    )
    city: Optional[str] = Field(
        default="Tokyo",
        description="City of the impression",
        examples=["Tokyo"],
    )
    latitude: Optional[float] = Field(
        default=35.69,
        ge=-90.0,
        le=90.0,
        description="Geographical latitude",
        examples=[35.69],
    )
    longitude: Optional[float] = Field(
        default=139.63,
        ge=-180.0,
        le=180.0,
        description="Geographical longitude",
        examples=[139.63],
    )
    postal_code: Optional[str] = Field(
        default="100-0001",
        description="Postal code of the user location",
        examples=["100-0001"],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
                "ip": "hashed_ip_1001",
                "device_type": "1",
                "biz": "home",
                "connection_type": "4g",
                "country": "jp",
                "city": "Tokyo",
                "latitude": 35.692,
                "longitude": 139.637,
                "postal_code": "100-0001",
            }
        },
    )


class BatchImpressionRequest(BaseModel):
    """
    Batch of impressions for high-throughput vectorized scoring.
    """

    items: List[ImpressionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of impression requests to score (max 1000)",
    )


class PredictionResponse(BaseModel):
    """
    Click prediction and bid recommendation for a single impression.
    """

    auction_id: Optional[str] = Field(
        default=None,
        description="Echoed auction ID for request correlation",
    )
    p_ctr: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted Click-Through Rate probability (pCTR in [0, 1])",
        examples=[0.0245],
    )
    bid: bool = Field(
        ...,
        description="Bidding recommendation: True if pCTR >= threshold, False otherwise",
        examples=[True],
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used for the bid recommendation",
        examples=[0.3821],
    )
    latency_ms: float = Field(
        ...,
        description="Inference execution latency in milliseconds for this item",
        examples=[0.72],
    )


class BatchPredictionResponse(BaseModel):
    """
    Results of batch impression scoring.
    """

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions matching input items"
    )
    count: int = Field(..., description="Number of scored impressions")
    total_latency_ms: float = Field(
        ..., description="Total batch inference latency in milliseconds"
    )


class HealthResponse(BaseModel):
    """
    Service health and readiness status.
    """

    status: str = Field(default="ok", examples=["ok"])
    ready: bool = Field(default=True, examples=[True])
    model_loaded: bool = Field(default=True, examples=[True])
    model_version: str = Field(default="1.0.0", examples=["1.0.0"])
    timestamp: str = Field(..., description="Server ISO timestamp")


class ModelInfoResponse(BaseModel):
    """
    Metadata, metrics, and parameters of the loaded model.
    """

    model_name: str
    version: str
    trained_at: str
    algorithm: str
    feature_count: int
    default_bid_threshold: float
    metrics: Dict[str, float]
    dataset_stats: Dict[str, Any]
    latency_benchmarks_ms: Dict[str, float]
    kpi_simulation: List[Dict[str, Any]]
