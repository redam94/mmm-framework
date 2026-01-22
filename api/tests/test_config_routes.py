"""
Tests for configuration management API routes.
"""

import pytest
from fastapi import status


class TestConfigCreate:
    """Tests for configuration creation endpoint."""

    def test_create_config_success(self, test_client, sample_config):
        """Test successful config creation."""
        response = test_client.post("/configs", json=sample_config)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "config_id" in data
        assert data["name"] == sample_config["name"]
        assert data["mff_config"]["kpi"]["name"] == "Sales"

    def test_create_config_missing_name(self, test_client, sample_config):
        """Test config creation without name."""
        config = sample_config.copy()
        del config["name"]

        response = test_client.post("/configs", json=config)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_config_without_kpi(self, test_client, sample_config):
        """Test config creation without KPI (allowed for extended models)."""
        config = sample_config.copy()
        config["mff_config"] = {
            "media_channels": config["mff_config"]["media_channels"],
            "controls": config["mff_config"]["controls"],
        }

        response = test_client.post("/configs", json=config)

        # KPI is now optional for extended model types
        assert response.status_code == status.HTTP_201_CREATED


class TestConfigList:
    """Tests for configuration listing endpoint."""

    def test_list_configs_empty(self, test_client):
        """Test listing with no configs."""
        response = test_client.get("/configs")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "configs" in data
        assert "total" in data

    def test_list_configs_with_data(self, test_client, sample_config):
        """Test listing with configs."""
        # Create a config
        test_client.post("/configs", json=sample_config)

        response = test_client.get("/configs")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 1


class TestConfigGet:
    """Tests for getting specific configuration."""

    def test_get_existing_config(self, test_client, sample_config):
        """Test getting existing config."""
        create_response = test_client.post("/configs", json=sample_config)
        config_id = create_response.json()["config_id"]

        response = test_client.get(f"/configs/{config_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["config_id"] == config_id
        assert data["name"] == sample_config["name"]

    def test_get_nonexistent_config(self, test_client):
        """Test getting non-existent config."""
        response = test_client.get("/configs/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConfigUpdate:
    """Tests for configuration update endpoint."""

    def test_update_config_name(self, test_client, sample_config):
        """Test updating config name."""
        create_response = test_client.post("/configs", json=sample_config)
        config_id = create_response.json()["config_id"]

        update_data = {"name": "Updated Config Name"}
        response = test_client.put(f"/configs/{config_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated Config Name"

    def test_update_nonexistent_config(self, test_client):
        """Test updating non-existent config."""
        update_data = {"name": "Updated Name"}
        response = test_client.put("/configs/nonexistent123", json=update_data)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConfigDelete:
    """Tests for configuration deletion endpoint."""

    def test_delete_existing_config(self, test_client, sample_config):
        """Test deleting existing config."""
        create_response = test_client.post("/configs", json=sample_config)
        config_id = create_response.json()["config_id"]

        response = test_client.delete(f"/configs/{config_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["success"] is True

        # Verify deletion
        get_response = test_client.get(f"/configs/{config_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_nonexistent_config(self, test_client):
        """Test deleting non-existent config."""
        response = test_client.delete("/configs/nonexistent123")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConfigDuplicate:
    """Tests for configuration duplication endpoint."""

    def test_duplicate_config(self, test_client, sample_config):
        """Test duplicating config."""
        create_response = test_client.post("/configs", json=sample_config)
        config_id = create_response.json()["config_id"]

        response = test_client.post(
            f"/configs/{config_id}/duplicate?new_name=Duplicated%20Config"
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Duplicated Config"
        assert data["config_id"] != config_id

    def test_duplicate_nonexistent_config(self, test_client):
        """Test duplicating non-existent config."""
        response = test_client.post(
            "/configs/nonexistent123/duplicate?new_name=New%20Name"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConfigValidate:
    """Tests for configuration validation endpoint."""

    def test_validate_valid_config(self, test_client, sample_config):
        """Test validating valid config."""
        response = test_client.post("/configs/validate", json=sample_config)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["valid"] is True

    def test_validate_config_no_channels(self, test_client, sample_config):
        """Test validating config without media channels."""
        config = sample_config.copy()
        config["mff_config"] = sample_config["mff_config"].copy()
        config["mff_config"]["media_channels"] = []

        response = test_client.post("/configs/validate", json=config)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_validate_config_duplicate_channels(self, test_client, sample_config):
        """Test validating config with duplicate channel names."""
        config = sample_config.copy()
        config["mff_config"] = sample_config["mff_config"].copy()
        config["mff_config"]["media_channels"] = [
            sample_config["mff_config"]["media_channels"][0],
            sample_config["mff_config"]["media_channels"][0],  # Duplicate
        ]

        response = test_client.post("/configs/validate", json=config)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
