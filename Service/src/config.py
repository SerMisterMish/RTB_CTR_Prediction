"""
Application configuration for CTR RTB Inference Service.
Loads configuration from environment variables or .env file with defaults.
"""

from _io import TextIOWrapper
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from json import load

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "CTR RTB Inference Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Artifact paths
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    MODEL_PATH: Path = BASE_DIR / "artifacts" / "model.joblib"
    PREPROCESSOR_PATH: Path = BASE_DIR / "artifacts" / "preprocessor.joblib"
    METADATA_PATH: Path = BASE_DIR / "artifacts" / "metadata.json"

    # Inference parameters
    # Default threshold from training metadata (best F1-score threshold)
    BID_THRESHOLD: float = 0.47208
    if Path.exists(METADATA_PATH):
        f: TextIOWrapper = Path.open(METADATA_PATH, "r")
        BID_THRESHOLD: float = float(
            load(f).get("metrics", {}).get("best_f1_threshold", BID_THRESHOLD)
        )
        f.close()

    TARGET_CTR_KPI: Optional[float] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
