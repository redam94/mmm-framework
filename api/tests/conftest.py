"""
Test fixtures for MMM Framework API tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add API directory to path - must be done before importing api modules
api_dir = str(Path(__file__).parent.parent)
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

# Also add the src directory for mmm_framework imports
src_dir = str(Path(__file__).parent.parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)
        # Create required subdirectories
        (storage_path / "data").mkdir()
        (storage_path / "configs").mkdir()
        (storage_path / "models").mkdir()
        (storage_path / "results").mkdir()
        yield storage_path


@pytest.fixture
def mock_settings(temp_storage):
    """Mock settings with temp storage."""
    from config import Settings

    settings = Settings(
        storage_path=temp_storage,
        redis_url="redis://localhost:6379",
        api_keys_enabled=False,
        debug=True,
    )
    return settings


@pytest.fixture
def mock_redis():
    """Mock Redis service."""
    mock = AsyncMock()
    mock.ping.return_value = True
    mock.connect.return_value = mock
    mock.disconnect.return_value = None
    mock.check_worker_health.return_value = True
    mock.get_queue_stats.return_value = {"pending": 0, "active": 0}
    mock.set_job_status.return_value = None
    mock.get_job_status.return_value = None
    mock.delete_job_status.return_value = None
    return mock


@pytest.fixture
def test_client(mock_settings, mock_redis):
    """Create test client with mocked dependencies."""
    with patch("config.get_settings", return_value=mock_settings):
        with patch("redis_service.get_redis", return_value=mock_redis):
            with patch("main.get_redis", return_value=mock_redis):
                from main import create_app

                app = create_app(mock_settings)
                yield TestClient(app)


@pytest.fixture
def sample_mff_csv():
    """Sample MFF CSV data."""
    return """Period,Geography,Product,Campaign,Outlet,Creative,VariableName,VariableValue
2023-01-01,US,ProductA,Campaign1,Online,Banner,Sales,1000
2023-01-01,US,ProductA,Campaign1,Online,Banner,TV_Spend,100
2023-01-01,US,ProductA,Campaign1,Online,Banner,Digital_Spend,50
2023-01-01,US,ProductA,Campaign1,Online,Banner,Price,9.99
2023-01-08,US,ProductA,Campaign1,Online,Banner,Sales,1100
2023-01-08,US,ProductA,Campaign1,Online,Banner,TV_Spend,120
2023-01-08,US,ProductA,Campaign1,Online,Banner,Digital_Spend,60
2023-01-08,US,ProductA,Campaign1,Online,Banner,Price,9.99
2023-01-15,US,ProductA,Campaign1,Online,Banner,Sales,1050
2023-01-15,US,ProductA,Campaign1,Online,Banner,TV_Spend,110
2023-01-15,US,ProductA,Campaign1,Online,Banner,Digital_Spend,55
2023-01-15,US,ProductA,Campaign1,Online,Banner,Price,9.99"""


@pytest.fixture
def sample_config():
    """Sample model configuration."""
    return {
        "name": "Test Config",
        "description": "Test configuration for unit tests",
        "mff_config": {
            "kpi": {
                "name": "Sales",
                "display_name": "Sales Revenue",
                "dimensions": ["Period"],
                "log_transform": False,
                "floor_value": 1e-6,
            },
            "media_channels": [
                {
                    "name": "TV_Spend",
                    "display_name": "TV Advertising",
                    "dimensions": ["Period"],
                    "adstock": {
                        "type": "geometric",
                        "l_max": 8,
                        "normalize": True,
                    },
                    "saturation": {"type": "hill"},
                },
                {
                    "name": "Digital_Spend",
                    "display_name": "Digital Advertising",
                    "dimensions": ["Period"],
                    "adstock": {
                        "type": "geometric",
                        "l_max": 4,
                        "normalize": True,
                    },
                    "saturation": {"type": "hill"},
                },
            ],
            "controls": [
                {
                    "name": "Price",
                    "display_name": "Product Price",
                    "dimensions": ["Period"],
                    "allow_negative": True,
                }
            ],
            "alignment": {
                "geo_allocation": "equal",
                "product_allocation": "sales",
            },
            "date_format": "%Y-%m-%d",
            "frequency": "W",
        },
        "model_settings": {
            "inference_method": "bayesian_pymc",
            "n_chains": 2,
            "n_draws": 100,
            "n_tune": 100,
            "target_accept": 0.9,
            "trend": {"type": "linear"},
            "seasonality": {"yearly": 2},
            "hierarchical": {"enabled": False},
            "random_seed": 42,
        },
    }


@pytest.fixture
def storage_service(mock_settings):
    """Create storage service with mock settings."""
    from storage import StorageService

    return StorageService(mock_settings)


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata."""
    return {
        "model_id": "test123",
        "name": "Test Model",
        "description": "Test model description",
        "data_id": "data123",
        "config_id": "config123",
        "status": "completed",
        "progress": 100.0,
        "created_at": "2023-01-01T00:00:00",
        "started_at": "2023-01-01T00:01:00",
        "completed_at": "2023-01-01T00:10:00",
        "diagnostics": {
            "divergences": 0,
            "rhat_max": 1.01,
            "ess_bulk_min": 500,
        },
    }
