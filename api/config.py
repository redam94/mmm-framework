"""
Configuration settings for the MMM API.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App settings
    app_name: str = "MMM Framework API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: str | None = None

    # Storage settings
    storage_backend: Literal["local", "s3"] = "local"
    storage_path: Path = Path("./storage")

    # S3 settings (if using S3 backend)
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # Job settings
    job_timeout: int = 3600  # 1 hour max for model fitting
    job_max_retries: int = 1

    # Data settings
    max_upload_size_mb: int = 100
    data_retention_days: int = 30

    # Model settings
    default_n_chains: int = 4
    default_n_draws: int = 1000
    default_n_tune: int = 1000

    # Authentication settings
    api_keys_enabled: bool = False
    valid_api_keys: list[str] = Field(default_factory=list)

    # CORS settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://localhost:3000"]
    )
    cors_allow_credentials: bool = True

    # Rate limiting settings
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_period: str = "minute"

    @property
    def redis_settings(self) -> dict:
        """Get Redis connection settings for ARQ."""
        from arq.connections import RedisSettings

        return RedisSettings(
            host=self.redis_url.replace("redis://", "").split(":")[0],
            port=int(self.redis_url.split(":")[-1]) if ":" in self.redis_url else 6379,
            database=self.redis_db,
            password=self.redis_password,
        )

    def ensure_storage_dirs(self):
        """Ensure storage directories exist."""
        if self.storage_backend == "local":
            (self.storage_path / "data").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "configs").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "models").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "results").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings
