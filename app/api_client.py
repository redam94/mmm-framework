"""
API Client for MMM Framework Backend.

Provides cached HTTP client with async support for the FastAPI backend.
Uses st.cache_resource for connection pooling and st.cache_data for response caching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, BinaryIO

import httpx
import streamlit as st


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.getenv("MMM_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("MMM_API_TIMEOUT", "30.0"))


class JobStatus(str, Enum):
    """Status of a background job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Classes for Type Safety
# =============================================================================

@dataclass
class DatasetInfo:
    """Dataset information."""
    data_id: str
    filename: str
    rows: int
    columns: int
    variables: list[str]
    dimensions: dict[str, list[str]]
    created_at: datetime
    size_bytes: int
    preview: list[dict[str, Any]] | None = None
    
    @classmethod
    def from_dict(cls, data: dict) -> DatasetInfo:
        return cls(
            data_id=data["data_id"],
            filename=data["filename"],
            rows=data["rows"],
            columns=data["columns"],
            variables=data["variables"],
            dimensions=data["dimensions"],
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            size_bytes=data["size_bytes"],
            preview=data.get("preview"),
        )


@dataclass
class ConfigInfo:
    """Configuration information."""
    config_id: str
    name: str
    description: str | None
    mff_config: dict[str, Any]
    model_settings: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def from_dict(cls, data: dict) -> ConfigInfo:
        return cls(
            config_id=data["config_id"],
            name=data["name"],
            description=data.get("description"),
            mff_config=data["mff_config"],
            model_settings=data["model_settings"],
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
        )


@dataclass
class ModelInfo:
    """Model information."""
    model_id: str
    name: str | None
    description: str | None
    data_id: str
    config_id: str
    status: JobStatus
    progress: float
    progress_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    diagnostics: dict[str, Any] | None
    
    @classmethod
    def from_dict(cls, data: dict) -> ModelInfo:
        return cls(
            model_id=data["model_id"],
            name=data.get("name"),
            description=data.get("description"),
            data_id=data["data_id"],
            config_id=data["config_id"],
            status=JobStatus(data["status"]),
            progress=float(data.get("progress", 0)),
            progress_message=data.get("progress_message"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            started_at=datetime.fromisoformat(data["started_at"].replace("Z", "+00:00")) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00")) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            diagnostics=data.get("diagnostics"),
        )
    
    @property
    def is_active(self) -> bool:
        return self.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]
    
    @property
    def is_complete(self) -> bool:
        return self.status == JobStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        return self.status == JobStatus.FAILED


@dataclass
class HealthInfo:
    """API health information."""
    status: str
    version: str
    redis_connected: bool
    worker_active: bool


# =============================================================================
# API Client
# =============================================================================

class APIError(Exception):
    """API error with status code and detail."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class MMMClient:
    """
    Client for the MMM Framework API.
    
    Provides synchronous methods for all API endpoints with proper error handling.
    """
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
    
    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                detail = error_data.get("detail", error_data.get("error", response.text))
            except Exception:
                detail = response.text
            raise APIError(response.status_code, detail)
        return response.json()
    
    # =========================================================================
    # Health Endpoints
    # =========================================================================
    
    def health(self) -> HealthInfo:
        """Check API health."""
        response = self._client.get("/health")
        data = self._handle_response(response)
        return HealthInfo(
            status=data["status"],
            version=data["version"],
            redis_connected=data["redis_connected"],
            worker_active=data["worker_active"],
        )
    
    def health_detailed(self) -> dict[str, Any]:
        """Get detailed health information."""
        response = self._client.get("/health/detailed")
        return self._handle_response(response)
    
    # =========================================================================
    # Data Endpoints
    # =========================================================================
    
    def upload_data(self, file_content: bytes, filename: str) -> DatasetInfo:
        """Upload a dataset."""
        files = {"file": (filename, file_content)}
        response = self._client.post("/data/upload", files=files)
        data = self._handle_response(response)
        return DatasetInfo.from_dict(data)
    
    def list_datasets(self, skip: int = 0, limit: int = 50) -> tuple[list[DatasetInfo], int]:
        """List all datasets."""
        response = self._client.get("/data", params={"skip": skip, "limit": limit})
        data = self._handle_response(response)
        datasets = [DatasetInfo.from_dict(d) for d in data["datasets"]]
        return datasets, data["total"]
    
    def get_dataset(self, data_id: str, include_preview: bool = False, preview_rows: int = 10) -> DatasetInfo:
        """Get dataset information."""
        params = {"include_preview": include_preview, "preview_rows": preview_rows}
        response = self._client.get(f"/data/{data_id}", params=params)
        data = self._handle_response(response)
        return DatasetInfo.from_dict(data)
    
    def get_dataset_variables(self, data_id: str) -> dict[str, Any]:
        """Get dataset variable summary."""
        response = self._client.get(f"/data/{data_id}/variables")
        return self._handle_response(response)
    
    def delete_dataset(self, data_id: str) -> bool:
        """Delete a dataset."""
        response = self._client.delete(f"/data/{data_id}")
        data = self._handle_response(response)
        return data.get("success", False)
    
    # =========================================================================
    # Config Endpoints
    # =========================================================================
    
    def create_config(
        self,
        name: str,
        mff_config: dict[str, Any],
        model_settings: dict[str, Any],
        description: str | None = None,
    ) -> ConfigInfo:
        """Create a new configuration."""
        payload = {
            "name": name,
            "description": description,
            "mff_config": mff_config,
            "model_settings": model_settings,
        }
        response = self._client.post("/configs", json=payload)
        data = self._handle_response(response)
        return ConfigInfo.from_dict(data)
    
    def list_configs(self, skip: int = 0, limit: int = 50) -> tuple[list[ConfigInfo], int]:
        """List all configurations."""
        response = self._client.get("/configs", params={"skip": skip, "limit": limit})
        data = self._handle_response(response)
        configs = [ConfigInfo.from_dict(c) for c in data["configs"]]
        return configs, data["total"]
    
    def get_config(self, config_id: str) -> ConfigInfo:
        """Get configuration details."""
        response = self._client.get(f"/configs/{config_id}")
        data = self._handle_response(response)
        return ConfigInfo.from_dict(data)
    
    def update_config(
        self,
        config_id: str,
        name: str | None = None,
        description: str | None = None,
        mff_config: dict[str, Any] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> ConfigInfo:
        """Update a configuration."""
        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if mff_config is not None:
            payload["mff_config"] = mff_config
        if model_settings is not None:
            payload["model_settings"] = model_settings
        
        response = self._client.patch(f"/configs/{config_id}", json=payload)
        data = self._handle_response(response)
        return ConfigInfo.from_dict(data)
    
    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration."""
        response = self._client.delete(f"/configs/{config_id}")
        data = self._handle_response(response)
        return data.get("success", False)
    
    # =========================================================================
    # Model Endpoints
    # =========================================================================
    
    def start_model_fit(
        self,
        data_id: str,
        config_id: str,
        name: str | None = None,
        description: str | None = None,
        n_chains: int | None = None,
        n_draws: int | None = None,
        n_tune: int | None = None,
        random_seed: int | None = None,
    ) -> ModelInfo:
        """Start model fitting job."""
        payload = {
            "data_id": data_id,
            "config_id": config_id,
        }
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if n_chains is not None:
            payload["n_chains"] = n_chains
        if n_draws is not None:
            payload["n_draws"] = n_draws
        if n_tune is not None:
            payload["n_tune"] = n_tune
        if random_seed is not None:
            payload["random_seed"] = random_seed
        
        response = self._client.post("/models/fit", json=payload)
        data = self._handle_response(response)
        return ModelInfo.from_dict(data)
    
    def list_models(
        self,
        status_filter: JobStatus | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[ModelInfo], int]:
        """List all models."""
        params = {"skip": skip, "limit": limit}
        if status_filter:
            params["status_filter"] = status_filter.value
        
        response = self._client.get("/models", params=params)
        data = self._handle_response(response)
        models = [ModelInfo.from_dict(m) for m in data["models"]]
        return models, data["total"]
    
    def get_model(self, model_id: str) -> ModelInfo:
        """Get model information and status."""
        response = self._client.get(f"/models/{model_id}")
        data = self._handle_response(response)
        return ModelInfo.from_dict(data)
    
    def get_model_status(self, model_id: str) -> dict[str, Any]:
        """Get model fitting status."""
        response = self._client.get(f"/models/{model_id}/status")
        return self._handle_response(response)
    
    def get_model_results(self, model_id: str) -> dict[str, Any]:
        """Get model results (posterior summary, diagnostics)."""
        response = self._client.get(f"/models/{model_id}/results")
        return self._handle_response(response)
    
    def get_model_summary(self, model_id: str) -> dict[str, Any]:
        """Get model summary."""
        response = self._client.get(f"/models/{model_id}/summary")
        return self._handle_response(response)
    
    def compute_contributions(
        self,
        model_id: str,
        time_period: tuple[int, int] | None = None,
        channels: list[str] | None = None,
        compute_uncertainty: bool = True,
        hdi_prob: float = 0.94,
    ) -> dict[str, Any]:
        """Compute counterfactual contributions."""
        payload = {
            "compute_uncertainty": compute_uncertainty,
            "hdi_prob": hdi_prob,
        }
        if time_period:
            payload["time_period"] = list(time_period)
        if channels:
            payload["channels"] = channels
        
        response = self._client.post(f"/models/{model_id}/contributions", json=payload)
        return self._handle_response(response)
    
    def run_scenario(
        self,
        model_id: str,
        spend_changes: dict[str, float],
        time_period: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Run what-if scenario."""
        payload = {
            "spend_changes": spend_changes,
        }
        if time_period:
            payload["time_period"] = list(time_period)
        
        response = self._client.post(f"/models/{model_id}/scenario", json=payload)
        return self._handle_response(response)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        response = self._client.delete(f"/models/{model_id}")
        data = self._handle_response(response)
        return data.get("success", False)
    
    def download_model(self, model_id: str) -> bytes:
        """Download model artifact."""
        response = self._client.get(f"/models/{model_id}/download")
        if response.status_code >= 400:
            raise APIError(response.status_code, response.text)
        return response.content
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()


# =============================================================================
# Streamlit Cached Client
# =============================================================================

@st.cache_resource
def get_api_client() -> MMMClient:
    """
    Get a cached API client instance.
    
    Uses st.cache_resource to reuse the same HTTP client across reruns.
    """
    return MMMClient()


# =============================================================================
# Cached Data Operations (return raw dicts for pickle serialization)
# =============================================================================

@st.cache_data(ttl=60)
def _fetch_datasets_raw(_client: MMMClient) -> list[dict[str, Any]]:
    """Fetch raw dataset dicts with caching (1 minute TTL)."""
    response = _client._client.get("/data", params={"skip": 0, "limit": 100})
    data = _client._handle_response(response)
    return data["datasets"]


@st.cache_data(ttl=60)
def _fetch_configs_raw(_client: MMMClient) -> list[dict[str, Any]]:
    """Fetch raw config dicts with caching (1 minute TTL)."""
    response = _client._client.get("/configs", params={"skip": 0, "limit": 100})
    data = _client._handle_response(response)
    return data["configs"]


@st.cache_data(ttl=30)
def _fetch_models_raw(_client: MMMClient, status_filter: str | None = None) -> list[dict[str, Any]]:
    """Fetch raw model dicts with caching (30 second TTL)."""
    params = {"skip": 0, "limit": 100}
    if status_filter:
        params["status"] = status_filter
    response = _client._client.get("/models", params=params)
    data = _client._handle_response(response)
    return data["models"]


@st.cache_data(ttl=300)
def fetch_model_results(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch model results with caching (5 minute TTL)."""
    return _client.get_model_results(model_id)


@st.cache_data(ttl=300)
def fetch_contributions(_client: MMMClient, model_id: str, hdi_prob: float = 0.94) -> dict[str, Any]:
    """Fetch contributions with caching (5 minute TTL)."""
    return _client.compute_contributions(model_id, hdi_prob=hdi_prob)


# =============================================================================
# Public Cached Fetchers (convert raw dicts to dataclasses)
# =============================================================================

def fetch_datasets(client: MMMClient) -> list[DatasetInfo]:
    """Fetch datasets with caching, returns DatasetInfo objects."""
    raw_data = _fetch_datasets_raw(client)
    return [DatasetInfo.from_dict(d) for d in raw_data]


def fetch_configs(client: MMMClient) -> list[ConfigInfo]:
    """Fetch configurations with caching, returns ConfigInfo objects."""
    raw_data = _fetch_configs_raw(client)
    return [ConfigInfo.from_dict(c) for c in raw_data]


def fetch_models(client: MMMClient, status_filter: str | None = None) -> list[ModelInfo]:
    """Fetch models with caching, returns ModelInfo objects."""
    raw_data = _fetch_models_raw(client, status_filter)
    return [ModelInfo.from_dict(m) for m in raw_data]


def clear_data_cache():
    """Clear dataset cache."""
    _fetch_datasets_raw.clear()


def clear_config_cache():
    """Clear configuration cache."""
    _fetch_configs_raw.clear()


def clear_model_cache():
    """Clear model cache."""
    _fetch_models_raw.clear()
    fetch_model_results.clear()
    fetch_contributions.clear()


def clear_all_cache():
    """Clear all caches."""
    clear_data_cache()
    clear_config_cache()
    clear_model_cache()