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
            return self.dimensions.get("Geography", [])
        return []

    def get_products(self) -> list:
        """Get products from dimensions."""
        if self.dimensions and isinstance(self.dimensions, dict):
            return self.dimensions.get("Product", [])
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
            kpi = self.mff_config.get("kpi", {})
            if isinstance(kpi, dict):
                return kpi.get("name", "N/A")
        return "N/A"

    def get_kpi_dimensions(self) -> list:
        """Get KPI dimensions from mff_config."""
        if self.mff_config and isinstance(self.mff_config, dict):
            kpi = self.mff_config.get("kpi", {})
            if isinstance(kpi, dict):
                return kpi.get("dimensions", ["Period"])
        return ["Period"]

    def get_media_channels(self) -> list:
        """Get media channels from mff_config."""
        if self.mff_config and isinstance(self.mff_config, dict):
            return self.mff_config.get("media_channels", [])
        return []

    def get_media_channel_names(self) -> list[str]:
        """Get list of media channel names."""
        return [ch.get("name", "") for ch in self.get_media_channels()]

    def get_controls(self) -> list:
        """Get control variables from mff_config."""
        if self.mff_config and isinstance(self.mff_config, dict):
            return self.mff_config.get("controls", [])
        return []

    def get_control_names(self) -> list[str]:
        """Get list of control variable names."""
        return [c.get("name", "") for c in self.get_controls()]

    def get_inference_method(self) -> str:
        """Get inference method from model_settings."""
        if self.model_settings and isinstance(self.model_settings, dict):
            return self.model_settings.get("inference_method", "N/A")
        return "N/A"

    def get_inference_method_display(self) -> str:
        """Get human-readable inference method name."""
        method = self.get_inference_method()
        method_map = {
            "bayesian_numpyro": "Bayesian (NumPyro/JAX)",
            "bayesian_pymc": "Bayesian (PyMC)",
            "frequentist_ridge": "Frequentist (Ridge)",
            "frequentist_cvxpy": "Frequentist (CVXPY)",
        }
        return method_map.get(method, method)

    def get_trend_type(self) -> str:
        """Get trend type from model_settings."""
        if self.model_settings and isinstance(self.model_settings, dict):
            trend = self.model_settings.get("trend", {})
            if isinstance(trend, dict):
                return trend.get("type", "linear")
        return "linear"

    def get_mcmc_settings(self) -> dict:
        """Get MCMC settings from model_settings."""
        if self.model_settings and isinstance(self.model_settings, dict):
            return {
                "n_chains": self.model_settings.get("n_chains", 4),
                "n_draws": self.model_settings.get("n_draws", 1000),
                "n_tune": self.model_settings.get("n_tune", 1000),
                "target_accept": self.model_settings.get("target_accept", 0.9),
            }
        return {"n_chains": 4, "n_draws": 1000, "n_tune": 1000, "target_accept": 0.9}

    def get_seasonality(self) -> dict:
        """Get seasonality settings from model_settings."""
        if self.model_settings and isinstance(self.model_settings, dict):
            return self.model_settings.get("seasonality", {})
        return {}

    def is_hierarchical(self) -> bool:
        """Check if hierarchical modeling is enabled."""
        if self.model_settings and isinstance(self.model_settings, dict):
            hier = self.model_settings.get("hierarchical", {})
            if isinstance(hier, dict):
                return hier.get("enabled", False)
        return False


@dataclass
class ModelInfo:
    """Model information."""

    model_id: str
    config_id: str
    status: str
    created_at: datetime
    completed_at: datetime = None
    metrics: dict = None
    name: str = None  # Add this
    progress: float = 0.0 
    progress_message: str | None = None
    started_at: datetime | None = None
    error_message: str | None = None
    # Diagnostics (available after completion)
    diagnostics: dict[str, Any] | None = None


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
            headers={"Content-Type": "application/json"},
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
                **basic,
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

    def get_dataset(
        self, data_id: str, include_preview: bool = False, preview_rows: int = 10
    ) -> DatasetInfo:
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
        response = httpx.post(
        f"{self.base_url}/data/upload",
        files=files,
        timeout=self.timeout,
    )
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
    )
        

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
                created_at=(
                    datetime.fromisoformat(c["created_at"])
                    if c.get("created_at")
                    else datetime.now()
                ),
                updated_at=(
                    datetime.fromisoformat(c["updated_at"])
                    if c.get("updated_at")
                    else None
                ),
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
            created_at=(
                datetime.fromisoformat(c["created_at"])
                if c.get("created_at")
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(c["updated_at"]) if c.get("updated_at") else None
            ),
        )

    def create_config(self, config_data: dict) -> dict:
        """Create a new configuration."""
        response = self._client.post("/configs", json=config_data)
        return self._handle_response(response)

    def delete_config(self, config_id: str) -> dict:
        """Delete a configuration."""
        response = self._client.delete(f"/configs/{config_id}")
        return self._handle_response(response)

    def update_config(self, config_id: str, config_data: dict) -> dict:
        """Update an existing configuration."""
        response = self._client.put(f"/configs/{config_id}", json=config_data)
        return self._handle_response(response)

    def validate_config(self, config_data: dict) -> dict:
        """Validate a configuration without saving it."""
        response = self._client.post("/configs/validate", json=config_data)
        return self._handle_response(response)

    def duplicate_config(self, config_id: str, new_name: str) -> dict:
        """Create a copy of an existing configuration."""
        response = self._client.post(
            f"/configs/{config_id}/duplicate",
            params={"new_name": new_name}
        )
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
                completed_at=(
                    datetime.fromisoformat(m["completed_at"])
                    if m.get("completed_at")
                    else None
                ),
                metrics=m.get("metrics"),
                name=m.get("name"),  # Add this
                progress=m.get("progress", 0.0),
                progress_message=m.get("progress_message"),
                started_at=(
                    datetime.fromisoformat(m["started_at"])
                    if m.get("started_at")
                    else None
                ),
                error_message=m.get("error_message"),
                diagnostics=m.get("diagnostics"),
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
            completed_at=(
                datetime.fromisoformat(m["completed_at"])
                if m.get("completed_at")
                else None
            ),
            metrics=m.get("metrics"),
            name=m.get("name"),
            progress=m.get("progress", 0.0),
            progress_message=m.get("progress_message"),
            started_at=(
                datetime.fromisoformat(m["started_at"])
                if m.get("started_at")
                else None
            ),
            error_message=m.get("error_message"),
            diagnostics=m.get("diagnostics"),
        )

    def get_model_results(self, model_id: str) -> dict:
        """Get model results."""
        response = self._client.get(f"/models/{model_id}/results")
        return self._handle_response(response)

    # -------------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------------

    def submit_fit_job(
        self, 
        data_id: str, 
        config_id: str, 
        name: str = None,
        description: str = None,
        n_chains: int = None,
        n_draws: int = None,
        n_tune: int = None,
        random_seed: int = None,
    ) -> dict:
        """Submit a model fitting job."""
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
        m = self._handle_response(response)
        return ModelInfo(
            model_id=m["model_id"],
            config_id=m["config_id"],
            status=m["status"],
            created_at=datetime.fromisoformat(m["created_at"]),
            completed_at=(
                datetime.fromisoformat(m["completed_at"])
                if m.get("completed_at")
                else None
            ),
            metrics=m.get("metrics"),
            name=m.get("name"),
            progress=m.get("progress", 0.0),
            progress_message=m.get("progress_message"),
            started_at=(
                datetime.fromisoformat(m["started_at"])
                if m.get("started_at")
                else None
            ),
            error_message=m.get("error_message"),
            diagnostics=m.get("diagnostics"),
        )

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

    def compute_contributions(self, model_id: str, hdi_prob: float = 0.94) -> dict:
        """Compute channel contributions."""
        params = {"hdi_prob": hdi_prob}
        response = self._client.get(f"/models/{model_id}/contributions", params=params)
        return self._handle_response(response)
    
    def generate_report(
        self,
        model_id: str,
        title: str = None,
        client: str = None,
        subtitle: str = None,
        analysis_period: str = None,
        include_executive_summary: bool = True,
        include_model_fit: bool = True,
        include_channel_roi: bool = True,
        include_decomposition: bool = True,
        include_saturation: bool = True,
        include_diagnostics: bool = True,
        include_methodology: bool = True,
        credible_interval: float = 0.8,
        currency_symbol: str = "$",
    ) -> dict:
        """Generate an HTML report for a model."""
        payload = {
            "title": title,
            "client": client,
            "subtitle": subtitle,
            "analysis_period": analysis_period,
            "include_executive_summary": include_executive_summary,
            "include_model_fit": include_model_fit,
            "include_channel_roi": include_channel_roi,
            "include_decomposition": include_decomposition,
            "include_saturation": include_saturation,
            "include_diagnostics": include_diagnostics,
            "include_methodology": include_methodology,
            "credible_interval": credible_interval,
            "currency_symbol": currency_symbol,
        }
        response = self._client.post(f"/models/{model_id}/report", json=payload)
        return self._handle_response(response)


    def get_report_status(self, model_id: str, report_id: str) -> dict:
        """Get report generation status."""
        response = self._client.get(f"/models/{model_id}/report/{report_id}/status")
        return self._handle_response(response)


    def download_report(self, model_id: str, report_id: str) -> bytes:
        """Download a generated report as HTML content."""
        response = self._client.get(f"/models/{model_id}/report/{report_id}/download")
        if response.status_code == 200:
            return response.content
        return self._handle_response(response)


    def list_reports(self, model_id: str) -> dict:
        """List all reports for a model."""
        response = self._client.get(f"/models/{model_id}/reports")
        return self._handle_response(response)

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
def fetch_models(_client: MMMAPIClient, status_filter: str = None) -> list[ModelInfo]:
    """Fetch models with caching."""
    models = _client.list_models()
    if status_filter:
        models = [m for m in models if m.status == status_filter]
    return models


@st.cache_resource(ttl=60)
def fetch_models_with_total(_client: MMMAPIClient) -> tuple[list[ModelInfo], int]:
    """Fetch models with caching, returns (models, total)."""
    models = _client.list_models()
    return models, len(models)


@st.cache_resource(ttl=300)
def fetch_model_results(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch model results with caching."""
    return _client.get_model_results(model_id)

@st.cache_resource(ttl=300)
def fetch_model_fit(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch model fit data with caching."""
    response = _client._client.get(f"/models/{model_id}/fit")
    return _client._handle_response(response)


@st.cache_resource(ttl=300)
def fetch_posteriors(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch posterior distributions with caching."""
    response = _client._client.get(f"/models/{model_id}/posteriors")
    return _client._handle_response(response)


@st.cache_resource(ttl=300)
def fetch_prior_posterior(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch prior vs posterior comparison with caching."""
    response = _client._client.get(f"/models/{model_id}/prior-posterior")
    return _client._handle_response(response)


@st.cache_resource(ttl=300)
def fetch_response_curves(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch response curves with caching."""
    response = _client._client.get(f"/models/{model_id}/response-curves")
    return _client._handle_response(response)


@st.cache_resource(ttl=300)
def fetch_decomposition(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch component decomposition with caching."""
    response = _client._client.get(f"/models/{model_id}/decomposition")
    return _client._handle_response(response)


@st.cache_resource(ttl=300)
def fetch_marginal_roas(_client: MMMAPIClient, model_id: str) -> dict | None:
    """Fetch marginal ROAS with caching."""
    try:
        response = _client._client.get(f"/models/{model_id}/roas")  # Changed from /marginal-roas
        return _client._handle_response(response)
    except Exception:
        return None


@st.cache_resource(ttl=300)
def fetch_contributions(_client: MMMAPIClient, model_id: str, hdi_prob: float = 0.94) -> dict:
    """Fetch channel contributions with caching."""
    payload = {
        "compute_uncertainty": True,
        "hdi_prob": hdi_prob,
    }
    response = _client._client.post(f"/models/{model_id}/contributions", json=payload)
    return _client._handle_response(response)


@st.cache_resource(ttl=300)
def fetch_model_summary(_client: MMMAPIClient, model_id: str) -> dict:
    """Fetch model summary with caching."""
    response = _client._client.get(f"/models/{model_id}/summary")
    return _client._handle_response(response)

@st.cache_resource(ttl=60)
def fetch_reports(_client: MMMAPIClient, model_id: str) -> list:
    """Fetch reports for a model with caching."""
    result = _client.list_reports(model_id)
    return result.get("reports", [])

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
