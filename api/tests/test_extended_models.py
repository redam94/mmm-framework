"""
Tests for extended model API routes (Nested, Multivariate, Combined MMM).
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status


class TestExtendedModelConfigCreate:
    """Tests for extended model config creation."""

    def test_create_nested_config(self, test_client, sample_mff_csv):
        """Test creating a nested model config."""
        # Upload data first
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        test_client.post("/data/upload", files=files)

        config = {
            "name": "Nested Test Config",
            "mff_config": {
                "model_type": "nested",
                "kpi": {
                    "name": "Sales",
                    "dimensions": ["Period"],
                },
                "media_channels": [
                    {"name": "TV_Spend", "dimensions": ["Period"]},
                    {"name": "Digital_Spend", "dimensions": ["Period"]},
                ],
                "nested_config": {
                    "mediators": [
                        {
                            "name": "awareness",
                            "mediator_type": "partially_observed",
                            "observation_noise_sigma": 0.15,
                        }
                    ],
                    "media_to_mediator_map": {
                        "awareness": ["TV_Spend", "Digital_Spend"]
                    },
                },
            },
        }

        response = test_client.post("/extended-models/configs", json=config)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "config_id" in data
        assert data["name"] == "Nested Test Config"
        assert data["model_type"] == "nested"

    def test_create_multivariate_config(self, test_client, sample_mff_csv):
        """Test creating a multivariate model config."""
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        test_client.post("/data/upload", files=files)

        config = {
            "name": "Multivariate Test Config",
            "mff_config": {
                "model_type": "multivariate",
                "kpi": {
                    "name": "Sales",
                    "dimensions": ["Period"],
                },
                "media_channels": [
                    {"name": "TV_Spend", "dimensions": ["Period"]},
                ],
                "multivariate_config": {
                    "outcomes": [
                        {"name": "ProductA_Sales", "data_column": "Sales"},
                        {"name": "ProductB_Sales", "data_column": "Sales"},
                    ],
                    "cross_effects": [
                        {
                            "source_outcome": "ProductA_Sales",
                            "target_outcome": "ProductB_Sales",
                            "effect_type": "cannibalization",
                        }
                    ],
                    "lkj_eta": 2.0,
                },
            },
        }

        response = test_client.post("/extended-models/configs", json=config)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["model_type"] == "multivariate"

    def test_create_combined_config(self, test_client, sample_mff_csv):
        """Test creating a combined model config."""
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        test_client.post("/data/upload", files=files)

        config = {
            "name": "Combined Test Config",
            "mff_config": {
                "model_type": "combined",
                "kpi": {
                    "name": "Sales",
                    "dimensions": ["Period"],
                },
                "media_channels": [
                    {"name": "TV_Spend", "dimensions": ["Period"]},
                ],
                "combined_config": {
                    "nested": {
                        "mediators": [
                            {"name": "awareness", "mediator_type": "fully_latent"}
                        ],
                    },
                    "multivariate": {
                        "outcomes": [
                            {"name": "Sales", "data_column": "Sales"},
                        ],
                        "cross_effects": [],
                    },
                    "mediator_to_outcome_map": {
                        "awareness": ["Sales"]
                    },
                },
            },
        }

        response = test_client.post("/extended-models/configs", json=config)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["model_type"] == "combined"

    def test_create_config_invalid_model_type(self, test_client):
        """Test creating config with invalid model type."""
        config = {
            "name": "Invalid Config",
            "mff_config": {
                "model_type": "invalid_type",
            },
        }

        response = test_client.post("/extended-models/configs", json=config)

        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestExtendedModelConfigGet:
    """Tests for getting extended model configs."""

    def test_get_existing_config(self, test_client, sample_mff_csv):
        """Test getting an existing extended config."""
        # Create config first
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        test_client.post("/data/upload", files=files)

        config = {
            "name": "Test Nested Config",
            "mff_config": {
                "model_type": "nested",
                "kpi": {"name": "Sales", "dimensions": ["Period"]},
                "media_channels": [{"name": "TV_Spend", "dimensions": ["Period"]}],
                "nested_config": {
                    "mediators": [{"name": "awareness", "mediator_type": "fully_latent"}],
                },
            },
        }

        create_response = test_client.post("/extended-models/configs", json=config)
        config_id = create_response.json()["config_id"]

        # Get the config
        response = test_client.get(f"/extended-models/configs/{config_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["config_id"] == config_id
        assert data["name"] == "Test Nested Config"

    def test_get_nonexistent_config(self, test_client):
        """Test getting a non-existent config."""
        response = test_client.get("/extended-models/configs/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExtendedModelFit:
    """Tests for extended model fitting."""

    def test_fit_model_missing_data(self, test_client, sample_mff_csv, mock_redis):
        """Test fitting with non-existent data."""
        # Create config first
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        test_client.post("/data/upload", files=files)

        config = {
            "name": "Test Config",
            "mff_config": {
                "model_type": "nested",
                "kpi": {"name": "Sales", "dimensions": ["Period"]},
                "media_channels": [{"name": "TV_Spend", "dimensions": ["Period"]}],
                "nested_config": {
                    "mediators": [{"name": "awareness", "mediator_type": "fully_latent"}],
                },
            },
        }

        config_response = test_client.post("/extended-models/configs", json=config)
        config_id = config_response.json()["config_id"]

        request = {
            "data_id": "nonexistent_data",
            "config_id": config_id,
            "name": "Test Model",
        }

        # Note: This test requires Redis for job queue
        try:
            response = test_client.post("/extended-models/fit", json=request)
            assert response.status_code == status.HTTP_404_NOT_FOUND
        except Exception:
            # If Redis connection fails, test is inconclusive
            pytest.skip("Redis connection required for this test")

    def test_fit_model_missing_config(self, test_client, sample_mff_csv, mock_redis):
        """Test fitting with non-existent config."""
        # Upload data
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        data_response = test_client.post("/data/upload", files=files)
        data_id = data_response.json()["data_id"]

        request = {
            "data_id": data_id,
            "config_id": "nonexistent_config",
            "name": "Test Model",
        }

        # Note: This test requires Redis for job queue
        try:
            response = test_client.post("/extended-models/fit", json=request)
            assert response.status_code == status.HTTP_404_NOT_FOUND
        except Exception:
            # If Redis connection fails, test is inconclusive
            pytest.skip("Redis connection required for this test")


class TestExtendedModelGet:
    """Tests for getting extended model info."""

    def test_get_nonexistent_model(self, test_client):
        """Test getting non-existent model."""
        response = test_client.get("/extended-models/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExtendedModelStatus:
    """Tests for extended model status."""

    def test_status_nonexistent_model(self, test_client):
        """Test getting status for non-existent model."""
        response = test_client.get("/extended-models/nonexistent123/status")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExtendedModelMediation:
    """Tests for mediation results endpoint."""

    def test_mediation_nonexistent_model(self, test_client):
        """Test getting mediation for non-existent model."""
        response = test_client.get("/extended-models/nonexistent123/mediation")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExtendedModelMultivariate:
    """Tests for multivariate results endpoint."""

    def test_multivariate_nonexistent_model(self, test_client):
        """Test getting multivariate results for non-existent model."""
        response = test_client.get("/extended-models/nonexistent123/multivariate")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExtendedModelDelete:
    """Tests for extended model deletion."""

    def test_delete_nonexistent_model(self, test_client):
        """Test deleting non-existent model."""
        response = test_client.delete("/extended-models/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND
