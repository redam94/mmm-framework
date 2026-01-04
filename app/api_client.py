"""
API Client for MMM Framework Backend.

Provides cached HTTP client with async support for the FastAPI backend.
Uses st.cache_resource for connection pooling and st.cache_data for response caching.
"""

import os
import streamlit as st
import httpx
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.getenv("MMM_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("MMM_API_TIMEOUT", "30.0"))


# =============================================================================
# Enums
# =============================================================================

class JobStatus(str, Enum):
    """Job status enum."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatasetInfo:
    """Dataset information matching API DataInfo schema."""
    data_id: str
    filename: str
    rows: int
    columns: int
    variables: list
    dimensions: dict
    created_at: datetime
    size_bytes: int = 0
    preview: list = None
    
    # Computed on access, not as properties
    def get_n_rows(self) -> int:
        """Alias for rows."""
        return self.rows
    
    def get_n_cols(self) -> int:
        """Alias for columns."""
        return self.columns
    
    def get_geographies(self) -> list:
        """Get geographies from dimensions."""
        if self.dimensions and isinstance(self.dimensions, dict):
            return self.dimensions.get('Geography', [])
        return []
    
    def get_products(self) -> list:
        """Get products from dimensions."""
        if self.dimensions and isinstance(self.dimensions, dict):
            return self.dimensions.get('Product', [])
        return []


@dataclass
class ConfigInfo:
    """Configuration information."""
    config_id: str
    name: str
    created_at: datetime
    updated_at: datetime = None
    description: str = None
    mff_config: dict = None
    model_settings: dict = None
    
    def get_kpi_name(self) -> str:
        """Get KPI name from mff_config."""
        if self.mff_config and isinstance(self.mff_config, dict):
            kpi = self.mff_config.get('kpi', {})
            if isinstance(kpi, dict):
                return kpi.get('name', 'N/A')
        return 'N/A'
    
    def get_media_channels(self) -> list:
        """Get media channels from mff_config."""
        if self.mff_config and isinstance(self.mff_config, dict):
            return self.mff_config.get('media_channels', [])
        return []
    
    def get_controls(self) -> list:
        """Get control variables from mff_config."""
        if self.mff_config and isinstance(self.mff_config, dict):
            return self.mff_config.get('controls', [])
        return []
    
    def get_inference_method(self) -> str:
        """Get inference method from model_settings."""
        if self.model_settings and isinstance(self.model_settings, dict):
            return self.model_settings.get('inference_method', 'N/A')
        return 'N/A'


@dataclass
class ModelInfo:
    """Model information."""
    model_id: str
    config_id: str
    status: str
    created_at: datetime
    completed_at: datetime = None
    metrics: dict = None


@dataclass
class JobInfo:
    """Job information."""
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    result: dict = None


# =============================================================================
# API Error
# =============================================================================

class APIError(Exception):
    """API error with status code and message."""
    def __init__(self, status_code: int, message: str, details: dict = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(f"API Error {status_code}: {message}")


# =============================================================================
# API Client
# =============================================================================

class MMMAPIClient:
    """HTTP client for MMM Framework API."""
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
    
    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response and raise errors if needed."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("detail", response.text)
            except Exception:
                message = response.text
            raise APIError(response.status_code, message)
        
        if response.status_code == 204:
            return {}
        
        return response.json()
    
    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------
    
    def health_check(self) -> dict:
        """Check API health."""
        response = self._client.get("/health")
        return self._handle_response(response)
    
    # Alias for compatibility
    def health(self) -> dict:
        """Check API health (alias for health_check)."""
        return self.health_check()
    
    def health_detailed(self) -> dict:
        """Get detailed health status including component status."""
        try:
            response = self._client.get("/health/detailed")
            return self._handle_response(response)
        except Exception:
            # Fall back to basic health check if detailed endpoint doesn't exist
            basic = self.health_check()
            return {
                "status": basic.get("status", "healthy"),
                "api": True,
                "storage": True,
                "worker": basic.get("worker_active", False),
                **basic
            }
    
    # -------------------------------------------------------------------------
    # Datasets (uses /data endpoint)
    # -------------------------------------------------------------------------
    
    def list_datasets(self, skip: int = 0, limit: int = 50) -> list[DatasetInfo]:
        """List all datasets."""
        params = {"skip": skip, "limit": limit}
        response = self._client.get("/data", params=params)
        data = self._handle_response(response)
        return [
            DatasetInfo(
                data_id=d["data_id"],
                filename=d["filename"],
                rows=d["rows"],
                columns=d["columns"],
                variables=d.get("variables", []),
                dimensions=d.get("dimensions", {}),
                created_at=datetime.fromisoformat(d["created_at"]),
                size_bytes=d.get("size_bytes", 0),
            )
            for d in data.get("datasets", [])
        ]
    
    def get_dataset(self, data_id: str, include_preview: bool = False, preview_rows: int = 10) -> DatasetInfo:
        """Get dataset by ID."""
        params = {"include_preview": include_preview, "preview_rows": preview_rows}
        response = self._client.get(f"/data/{data_id}", params=params)
        d = self._handle_response(response)
        return DatasetInfo(
            data_id=d["data_id"],
            filename=d["filename"],
            rows=d["rows"],
            columns=d["columns"],
            variables=d.get("variables", []),
            dimensions=d.get("dimensions", {}),
            created_at=datetime.fromisoformat(d["created_at"]),
            size_bytes=d.get("size_bytes", 0),
            preview=d.get("preview"),
        )
    
    def upload_dataset(self, file_content: bytes, filename: str) -> dict:
        """Upload a dataset."""
        files = {"file": (filename, file_content)}
        response = self._client.post(
            "/data/upload",
            files=files,
            headers={}  # Let httpx set content-type for multipart
        )
        return self._handle_response(response)
    
    def delete_dataset(self, data_id: str) -> dict:
        """Delete a dataset."""
        response = self._client.delete(f"/data/{data_id}")
        return self._handle_response(response)
    
    def get_dataset_variables(self, data_id: str) -> dict:
        """Get variable names and summary statistics from a dataset."""
        response = self._client.get(f"/data/{data_id}/variables")
        return self._handle_response(response)
    
    # -------------------------------------------------------------------------
    # Configurations
    # -------------------------------------------------------------------------
    
    def list_configs(self, skip: int = 0, limit: int = 50) -> list[ConfigInfo]:
        """List all configurations."""
        params = {"skip": skip, "limit": limit}
        response = self._client.get("/configs", params=params)
        data = self._handle_response(response)
        return [
            ConfigInfo(
                config_id=c.get("config_id", ""),
                name=c.get("name", "Unnamed"),
                description=c.get("description"),
                mff_config=c.get("mff_config"),
                model_settings=c.get("model_settings"),
                created_at=datetime.fromisoformat(c["created_at"]) if c.get("created_at") else datetime.now(),
                updated_at=datetime.fromisoformat(c["updated_at"]) if c.get("updated_at") else None,
            )
            for c in data.get("configs", [])
        ]
    
    def get_config(self, config_id: str) -> ConfigInfo:
        """Get configuration by ID."""
        response = self._client.get(f"/configs/{config_id}")
        c = self._handle_response(response)
        return ConfigInfo(
            config_id=c.get("config_id", ""),
            name=c.get("name", "Unnamed"),
            description=c.get("description"),
            mff_config=c.get("mff_config"),
            model_settings=c.get("model_settings"),
            created_at=datetime.fromisoformat(c["created_at"]) if c.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(c["updated_at"]) if c.get("updated_at") else None,
        )
    
    def create_config(self, config_data: dict) -> dict:
        """Create a new configuration."""
        response = self._client.post("/configs", json=config_data)
        return self._handle_response(response)
    
    def delete_config(self, config_id: str) -> dict:
        """Delete a configuration."""
        response = self._client.delete(f"/configs/{config_id}")
        return self._handle_response(response)
    
    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    
    def list_models(self, skip: int = 0, limit: int = 50) -> list[ModelInfo]:
        """List all models."""
        params = {"skip": skip, "limit": limit}
        response = self._client.get("/models", params=params)
        data = self._handle_response(response)
        return [
            ModelInfo(
                model_id=m["model_id"],
                config_id=m["config_id"],
                status=m["status"],
                created_at=datetime.fromisoformat(m["created_at"]),
                completed_at=datetime.fromisoformat(m["completed_at"]) if m.get("completed_at") else None,
                metrics=m.get("metrics"),
            )
            for m in data.get("models", [])
        ]
    
    def get_model(self, model_id: str) -> ModelInfo:
        """Get model by ID."""
        response = self._client.get(f"/models/{model_id}")
        m = self._handle_response(response)
        return ModelInfo(
            model_id=m["model_id"],
            config_id=m["config_id"],
            status=m["status"],
            created_at=datetime.fromisoformat(m["created_at"]),
            completed_at=datetime.fromisoformat(m["completed_at"]) if m.get("completed_at") else None,
            metrics=m.get("metrics"),
        )
    
    def get_model_results(self, model_id: str) -> dict:
        """Get model results."""
        response = self._client.get(f"/models/{model_id}/results")
        return self._handle_response(response)
    
    # -------------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------------
    
    def submit_fit_job(self, config_id: str) -> dict:
        """Submit a model fitting job."""
        response = self._client.post("/jobs/fit", json={"config_id": config_id})
        return self._handle_response(response)
    
    def get_job_status(self, job_id: str) -> JobInfo:
        """Get job status."""
        response = self._client.get(f"/jobs/{job_id}")
        j = self._handle_response(response)
        return JobInfo(
            job_id=j["job_id"],
            status=j["status"],
            progress=j.get("progress", 0.0),
            message=j.get("message", ""),
            result=j.get("result"),
        )


# =============================================================================
# Cached Client & Helpers
# =============================================================================

@st.cache_resource
def get_api_client() -> MMMAPIClient:
    """Get cached API client instance."""
    return MMMAPIClient()


@st.cache_resource(ttl=60)
def fetch_datasets(_client: MMMAPIClient) -> list[DatasetInfo]:
    """Fetch datasets with caching."""
    return _client.list_datasets()


@st.cache_resource(ttl=60)
def fetch_datasets_with_total(_client: MMMAPIClient) -> tuple[list[DatasetInfo], int]:
    """Fetch datasets with caching, returns (datasets, total)."""
    datasets = _client.list_datasets()
    return datasets, len(datasets)


@st.cache_resource(ttl=60)
def fetch_configs(_client: MMMAPIClient) -> list[ConfigInfo]:
    """Fetch configs with caching."""
    return _client.list_configs()


@st.cache_resource(ttl=60)
def fetch_configs_with_total(_client: MMMAPIClient) -> tuple[list[ConfigInfo], int]:
    """Fetch configs with caching, returns (configs, total)."""
    configs = _client.list_configs()
    return configs, len(configs)


@st.cache_resource(ttl=60)
def fetch_models(_client: MMMAPIClient) -> list[ModelInfo]:
    """Fetch models with caching."""
    return _client.list_models()


@st.cache_resource(ttl=60)
def fetch_models_with_total(_client: MMMAPIClient) -> tuple[list[ModelInfo], int]:
    """Fetch models with caching, returns (models, total)."""
    models = _client.list_models()
    return models, len(models)


@st.cache_resource(ttl=300)
def fetch_model_results(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch model results with caching."""
    return _client.get_model_results(model_id)


def clear_dataset_cache():
    """Clear dataset cache."""
    fetch_datasets.clear()
    fetch_datasets_with_total.clear()


# Alias for compatibility
clear_data_cache = clear_dataset_cache


def clear_config_cache():
    """Clear config cache."""
    fetch_configs.clear()
    fetch_configs_with_total.clear()


def clear_model_cache():
    """Clear model cache."""
    fetch_models.clear()
    fetch_models_with_total.clear()
    fetch_model_results.clear()


def clear_all_caches():
    """Clear all API caches."""
    fetch_datasets.clear()
    fetch_datasets_with_total.clear()
    fetch_configs.clear()
    fetch_configs_with_total.clear()
    fetch_models.clear()
    fetch_models_with_total.clear()
    fetch_model_results.clear()