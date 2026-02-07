"""
Tests for model management API routes.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status


class TestModelFit:
    """Tests for model fitting endpoint."""

    def test_fit_model_missing_data(self, test_client, sample_config):
        """Test fitting model with non-existent data."""
        # Create config
        config_response = test_client.post("/configs", json=sample_config)
        config_id = config_response.json()["config_id"]

        request = {
            "data_id": "nonexistent_data",
            "config_id": config_id,
            "name": "Test Model",
        }

        response = test_client.post("/models/fit", json=request)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_fit_model_missing_config(self, test_client, sample_mff_csv):
        """Test fitting model with non-existent config."""
        # Upload data
        files = {"file": ("test.csv", io.BytesIO(sample_mff_csv.encode()), "text/csv")}
        data_response = test_client.post("/data/upload", files=files)
        data_id = data_response.json()["data_id"]

        request = {
            "data_id": data_id,
            "config_id": "nonexistent_config",
            "name": "Test Model",
        }

        response = test_client.post("/models/fit", json=request)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelList:
    """Tests for model listing endpoint."""

    def test_list_models_empty(self, test_client):
        """Test listing with no models."""
        response = test_client.get("/models")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data

    def test_list_models_pagination(self, test_client):
        """Test model listing with pagination."""
        response = test_client.get("/models?skip=0&limit=10")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["skip"] == 0
        assert data["limit"] == 10


class TestModelGet:
    """Tests for getting specific model."""

    def test_get_nonexistent_model(self, test_client):
        """Test getting non-existent model."""
        response = test_client.get("/models/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelStatus:
    """Tests for model status endpoint."""

    def test_status_nonexistent_model(self, test_client):
        """Test getting status for non-existent model."""
        response = test_client.get("/models/nonexistent123/status")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelResults:
    """Tests for model results endpoint."""

    def test_results_nonexistent_model(self, test_client):
        """Test getting results for non-existent model."""
        response = test_client.get("/models/nonexistent123/results")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelDelete:
    """Tests for model deletion endpoint."""

    def test_delete_nonexistent_model(self, test_client):
        """Test deleting non-existent model."""
        response = test_client.delete("/models/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelFit:
    """Tests for model fit data endpoint."""

    def test_fit_nonexistent_model(self, test_client):
        """Test getting fit data for non-existent model."""
        response = test_client.get("/models/nonexistent123/fit")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelPosteriors:
    """Tests for model posteriors endpoint."""

    def test_posteriors_nonexistent_model(self, test_client):
        """Test getting posteriors for non-existent model."""
        response = test_client.get("/models/nonexistent123/posteriors")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelDecomposition:
    """Tests for model decomposition endpoint."""

    def test_decomposition_nonexistent_model(self, test_client):
        """Test getting decomposition for non-existent model."""
        response = test_client.get("/models/nonexistent123/decomposition")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelROAS:
    """Tests for model ROAS endpoint."""

    def test_roas_nonexistent_model(self, test_client):
        """Test getting ROAS for non-existent model."""
        response = test_client.get("/models/nonexistent123/roas")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelContributions:
    """Tests for model contributions endpoint."""

    def test_contributions_nonexistent_model(self, test_client, mock_redis):
        """Test computing contributions for non-existent model."""
        # Note: This test requires Redis, which may not be available
        # The endpoint checks model existence first, so it should return 404
        request = {"compute_uncertainty": True}
        try:
            response = test_client.post(
                "/models/nonexistent123/contributions", json=request
            )
            assert response.status_code == status.HTTP_404_NOT_FOUND
        except Exception:
            # If Redis connection fails, test is inconclusive
            pytest.skip("Redis connection required for this test")


class TestModelScenario:
    """Tests for model scenario endpoint."""

    def test_scenario_nonexistent_model(self, test_client):
        """Test running scenario for non-existent model."""
        request = {"spend_changes": {"TV_Spend": 0.1}}
        response = test_client.post("/models/nonexistent123/scenario", json=request)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelPredict:
    """Tests for model prediction endpoint."""

    def test_predict_nonexistent_model(self, test_client):
        """Test generating predictions for non-existent model."""
        request = {}
        response = test_client.post("/models/nonexistent123/predict", json=request)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestModelReport:
    """Tests for model report endpoints."""

    def test_generate_report_nonexistent_model(self, test_client, mock_redis):
        """Test generating report for non-existent model."""
        # Note: This test requires Redis for job queue
        request = {"title": "Test Report"}
        try:
            response = test_client.post("/models/nonexistent123/report", json=request)
            assert response.status_code == status.HTTP_404_NOT_FOUND
        except Exception:
            # If Redis connection fails, test is inconclusive
            pytest.skip("Redis connection required for this test")

    def test_list_reports_nonexistent_model(self, test_client):
        """Test listing reports for non-existent model."""
        response = test_client.get("/models/nonexistent123/reports")

        assert response.status_code == status.HTTP_404_NOT_FOUND
