"""
API Client for MMM Framework Backend.

Provides cached HTTP client with async support for the FastAPI backend.
Uses st.cache_resource for connection pooling and st.cache_data for response caching.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import httpx
import streamlit as st


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.getenv("MMM_API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("MMM_API_TIMEOUT", "30.0"))


# =============================================================================
# Enums and Data Classes
# =============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    def from_dict(cls, data: dict) -> "DatasetInfo":
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return cls(
            data_id=data["data_id"],
            filename=data["filename"],
            rows=data["rows"],
            columns=data["columns"],
            variables=data["variables"],
            dimensions=data["dimensions"],
            created_at=created_at,
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
    def from_dict(cls, data: dict) -> "ConfigInfo":
        created_at = data["created_at"]
        updated_at = data["updated_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        return cls(
            config_id=data["config_id"],
            name=data["name"],
            description=data.get("description"),
            mff_config=data["mff_config"],
            model_settings=data["model_settings"],
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class ModelInfo:
    """Model information."""
    model_id: str
    data_id: str
    config_id: str
    status: JobStatus
    created_at: datetime
    name: str | None = None
    description: str | None = None
    progress: float = 0.0
    progress_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    diagnostics: dict | None = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        
        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        
        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        
        status = data["status"]
        if isinstance(status, str):
            status = JobStatus(status)
        
        return cls(
            model_id=data["model_id"],
            data_id=data["data_id"],
            config_id=data["config_id"],
            status=status,
            created_at=created_at,
            name=data.get("name"),
            description=data.get("description"),
            progress=data.get("progress", 0.0),
            progress_message=data.get("progress_message"),
            started_at=started_at,
            completed_at=completed_at,
            error_message=data.get("error_message"),
            diagnostics=data.get("diagnostics"),
        )


# =============================================================================
# Exceptions
# =============================================================================

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ConnectionError(APIError):
    """Connection to API failed."""
    pass


class NotFoundError(APIError):
    """Resource not found."""
    pass


class ValidationError(APIError):
    """Request validation failed."""
    pass


# =============================================================================
# API Client
# =============================================================================

class MMMClient:
    """HTTP client for MMM Framework API."""
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
    
    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response and raise appropriate exceptions."""
        try:
            data = response.json()
        except Exception:
            data = {"detail": response.text}
        
        if response.status_code >= 400:
            detail = data.get("detail", str(data))
            
            if response.status_code == 404:
                raise NotFoundError(f"Not found: {detail}", response.status_code, detail)
            elif response.status_code == 422:
                raise ValidationError(f"Validation error: {detail}", response.status_code, detail)
            else:
                raise APIError(f"API error: {detail}", response.status_code, detail)
        
        return data
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    # =========================================================================
    # Health Endpoints
    # =========================================================================
    
    def health(self) -> dict:
        """Check API health."""
        try:
            response = self._client.get("/health")
            return self._handle_response(response)
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to API at {self.base_url}") from e
    
    def health_detailed(self) -> dict:
        """Get detailed health information."""
        response = self._client.get("/health/detailed")
        return self._handle_response(response)
    
    # =========================================================================
    # Data Endpoints
    # =========================================================================
    
    def upload_data(self, file_content: bytes, filename: str) -> DatasetInfo:
        """Upload a dataset file."""
        import io
        
        # Validate input
        if file_content is None:
            raise ValueError("File content is None")
        if not isinstance(file_content, bytes):
            raise ValueError(f"File content must be bytes, got {type(file_content)}")
        if len(file_content) == 0:
            raise ValueError("File content is empty (0 bytes)")
        
        # Create a fresh client for uploads to avoid any connection state issues
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            file_obj = io.BytesIO(file_content)
            files = {"file": (filename, file_obj)}
            response = client.post("/data/upload", files=files)
        
        data = self._handle_response(response)
        return DatasetInfo.from_dict(data)
    
    def list_datasets(self, skip: int = 0, limit: int = 20) -> tuple[list[DatasetInfo], int]:
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
    
    def get_dataset_preview(self, data_id: str, n_rows: int = 10) -> dict:
        """Get a preview of the dataset."""
        response = self._client.get(f"/data/{data_id}/preview", params={"n_rows": n_rows})
        return self._handle_response(response)
    
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
        """Start fitting a model."""
        payload = {
            "data_id": data_id,
            "config_id": config_id,
        }
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if n_chains:
            payload["n_chains"] = n_chains
        if n_draws:
            payload["n_draws"] = n_draws
        if n_tune:
            payload["n_tune"] = n_tune
        if random_seed:
            payload["random_seed"] = random_seed
        
        response = self._client.post("/models/fit", json=payload)
        data = self._handle_response(response)
        return ModelInfo.from_dict(data)
    
    def list_models(
        self,
        skip: int = 0,
        limit: int = 20,
        status_filter: JobStatus | None = None,
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
        """Get model results (diagnostics, parameter summary)."""
        response = self._client.get(f"/models/{model_id}/results")
        return self._handle_response(response)
    
    def get_model_summary(self, model_id: str) -> dict[str, Any]:
        """Get model summary."""
        response = self._client.get(f"/models/{model_id}/summary")
        return self._handle_response(response)
    
    # =========================================================================
    # Detailed Results Endpoints
    # =========================================================================
    
    def get_model_fit(self, model_id: str) -> dict[str, Any]:
        """
        Get model fit data (observed vs predicted).
        
        Returns:
            dict with periods, observed, predicted_mean, predicted_std, r2, rmse, mape
        """
        response = self._client.get(f"/models/{model_id}/fit")
        return self._handle_response(response)
    
    def get_posteriors(
        self,
        model_id: str,
        parameters: list[str] | None = None,
        n_samples: int = 500,
    ) -> dict[str, Any]:
        """
        Get posterior distribution samples.
        
        Args:
            model_id: Model identifier
            parameters: Optional list of specific parameters to return
            n_samples: Number of samples per parameter (default 500)
        
        Returns:
            dict mapping parameter names to their samples and stats
        """
        params = {"n_samples": n_samples}
        if parameters:
            params["parameters"] = parameters
        
        response = self._client.get(f"/models/{model_id}/posteriors", params=params)
        return self._handle_response(response)
    
    def get_prior_posterior(
        self,
        model_id: str,
        n_samples: int = 500,
    ) -> dict[str, Any]:
        """
        Get prior vs posterior comparison data.
        
        Args:
            model_id: Model identifier
            n_samples: Number of samples per parameter (default 500)
        
        Returns:
            dict with parameters mapping to prior/posterior samples and shrinkage metrics
        """
        response = self._client.get(
            f"/models/{model_id}/prior-posterior",
            params={"n_samples": n_samples},
        )
        return self._handle_response(response)
    
    def get_response_curves(
        self,
        model_id: str,
        n_points: int = 100,
        n_samples: int = 200,
    ) -> dict[str, Any]:
        """
        Get response curves for each channel.
        
        Args:
            model_id: Model identifier
            n_points: Number of points per curve
            n_samples: Number of posterior samples for uncertainty
        
        Returns:
            dict with channels mapping to spend, response, and uncertainty bands
        """
        response = self._client.get(
            f"/models/{model_id}/response-curves",
            params={"n_points": n_points, "n_samples": n_samples},
        )
        return self._handle_response(response)
    
    def get_decomposition(self, model_id: str) -> dict[str, Any]:
        """
        Get component decomposition.
        
        Returns:
            dict with periods, observed, and components (trend, seasonality, media, etc.)
        """
        response = self._client.get(f"/models/{model_id}/decomposition")
        return self._handle_response(response)
    
    def get_marginal_roas(self, model_id: str) -> dict[str, Any]:
        """
        Get marginal ROAS for each channel.
        
        Returns:
            dict with channels mapping to mean, std, and HDI bounds
        """
        response = self._client.get(f"/models/{model_id}/roas")
        return self._handle_response(response)
    
    # =========================================================================
    # Analysis Endpoints
    # =========================================================================
    
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
    
    def generate_predictions(
        self,
        model_id: str,
        media_spend: dict[str, list[float]] | None = None,
        return_samples: bool = False,
    ) -> dict[str, Any]:
        """Generate predictions with optional modified spend."""
        payload = {"return_samples": return_samples}
        if media_spend:
            payload["media_spend"] = media_spend
        
        response = self._client.post(f"/models/{model_id}/predict", json=payload)
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
            self._handle_response(response)
        return response.content


# =============================================================================
# Streamlit Integration
# =============================================================================

@st.cache_resource
def get_api_client() -> MMMClient:
    """Get a cached API client instance."""
    return MMMClient()


def check_api_connection() -> bool:
    """Check if the API is accessible."""
    try:
        client = get_api_client()
        client.health()
        return True
    except Exception:
        return False


# =============================================================================
# Cached Data Fetchers
# =============================================================================

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


@st.cache_data(ttl=60)
def fetch_model_results(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch basic model results with 1-minute cache."""
    return _client.get_model_results(model_id)


@st.cache_data(ttl=300)
def fetch_model_fit(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch model fit data with 5-minute cache."""
    return _client.get_model_fit(model_id)


@st.cache_data(ttl=300)
def fetch_posteriors(_client: MMMClient, model_id: str, n_samples: int = 500) -> dict[str, Any]:
    """Fetch posterior samples with 5-minute cache."""
    return _client.get_posteriors(model_id, n_samples=n_samples)


@st.cache_data(ttl=300)
def fetch_prior_posterior(_client: MMMClient, model_id: str, n_samples: int = 500) -> dict[str, Any]:
    """Fetch prior vs posterior comparison with 5-minute cache."""
    return _client.get_prior_posterior(model_id, n_samples=n_samples)


@st.cache_data(ttl=300)
def fetch_response_curves(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch response curves with 5-minute cache."""
    return _client.get_response_curves(model_id)


@st.cache_data(ttl=300)
def fetch_decomposition(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch component decomposition with 5-minute cache."""
    return _client.get_decomposition(model_id)


@st.cache_data(ttl=300)
def fetch_marginal_roas(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch marginal ROAS with 5-minute cache."""
    return _client.get_marginal_roas(model_id)


@st.cache_data(ttl=60)
def _fetch_contributions_raw(_client: MMMClient, model_id: str, hdi_prob: float = 0.94) -> dict[str, Any]:
    """Fetch contributions with 1-minute cache."""
    try:
        return _client.compute_contributions(model_id, hdi_prob=hdi_prob)
    except APIError:
        return {}


@st.cache_data(ttl=300)
def fetch_model_summary(_client: MMMClient, model_id: str) -> dict[str, Any]:
    """Fetch model summary with 5-minute cache."""
    return _client.get_model_summary(model_id)


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


def fetch_contributions(client: MMMClient, model_id: str, hdi_prob: float = 0.94) -> dict[str, Any]:
    """Fetch contributions with caching."""
    return _fetch_contributions_raw(client, model_id, hdi_prob)


# =============================================================================
# Cache Clearing Functions
# =============================================================================

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
    _fetch_contributions_raw.clear()
    fetch_model_fit.clear()
    fetch_posteriors.clear()
    fetch_response_curves.clear()
    fetch_decomposition.clear()
    fetch_marginal_roas.clear()
    fetch_model_summary.clear()


def clear_all_cache():
    """Clear all caches."""
    clear_data_cache()
    clear_config_cache()
    clear_model_cache()