"""
Tests for storage service.
"""

import json
import pytest


class TestDataStorage:
    """Tests for data storage functionality."""

    def test_save_and_load_data(self, storage_service, sample_mff_csv):
        """Test saving and loading data."""
        metadata = storage_service.save_data(sample_mff_csv.encode(), "test_data.csv")

        assert "data_id" in metadata
        assert metadata["filename"] == "test_data.csv"
        assert metadata["rows"] > 0

        # Load the data back
        df = storage_service.load_data(metadata["data_id"])
        assert len(df) > 0

    def test_data_exists(self, storage_service, sample_mff_csv):
        """Test data existence check."""
        metadata = storage_service.save_data(sample_mff_csv.encode(), "test.csv")

        assert storage_service.data_exists(metadata["data_id"]) is True
        assert storage_service.data_exists("nonexistent") is False

    def test_delete_data(self, storage_service, sample_mff_csv):
        """Test data deletion."""
        metadata = storage_service.save_data(sample_mff_csv.encode(), "test.csv")
        data_id = metadata["data_id"]

        assert storage_service.data_exists(data_id) is True

        result = storage_service.delete_data(data_id)
        assert result is True
        assert storage_service.data_exists(data_id) is False

    def test_list_data(self, storage_service, sample_mff_csv):
        """Test listing data."""
        # Initially empty
        assert len(storage_service.list_data()) == 0

        # Add some data
        storage_service.save_data(sample_mff_csv.encode(), "test1.csv")
        storage_service.save_data(sample_mff_csv.encode(), "test2.csv")

        datasets = storage_service.list_data()
        assert len(datasets) == 2


class TestConfigStorage:
    """Tests for config storage functionality."""

    def test_save_and_load_config(self, storage_service, sample_config):
        """Test saving and loading config."""
        saved = storage_service.save_config(sample_config)

        assert "config_id" in saved
        assert "created_at" in saved
        assert "updated_at" in saved

        # Load the config back
        loaded = storage_service.load_config(saved["config_id"])
        assert loaded["name"] == sample_config["name"]

    def test_config_exists(self, storage_service, sample_config):
        """Test config existence check."""
        saved = storage_service.save_config(sample_config)

        assert storage_service.config_exists(saved["config_id"]) is True
        assert storage_service.config_exists("nonexistent") is False

    def test_update_config(self, storage_service, sample_config):
        """Test config update."""
        saved = storage_service.save_config(sample_config)
        config_id = saved["config_id"]

        updated = storage_service.update_config(config_id, {"name": "Updated Name"})

        assert updated["name"] == "Updated Name"

    def test_delete_config(self, storage_service, sample_config):
        """Test config deletion."""
        saved = storage_service.save_config(sample_config)
        config_id = saved["config_id"]

        assert storage_service.config_exists(config_id) is True

        result = storage_service.delete_config(config_id)
        assert result is True
        assert storage_service.config_exists(config_id) is False

    def test_list_configs(self, storage_service, sample_config):
        """Test listing configs."""
        # Initially empty
        assert len(storage_service.list_configs()) == 0

        # Add some configs
        storage_service.save_config(sample_config)
        config2 = sample_config.copy()
        config2["name"] = "Second Config"
        storage_service.save_config(config2)

        configs = storage_service.list_configs()
        assert len(configs) == 2


class TestModelStorage:
    """Tests for model storage functionality."""

    def test_save_and_get_model_metadata(self, storage_service, sample_model_metadata):
        """Test saving and getting model metadata."""
        model_id = sample_model_metadata["model_id"]

        storage_service.save_model_metadata(model_id, sample_model_metadata)

        loaded = storage_service.get_model_metadata(model_id)
        assert loaded["model_id"] == model_id
        assert loaded["name"] == sample_model_metadata["name"]

    def test_model_exists(self, storage_service, sample_model_metadata):
        """Test model existence check."""
        model_id = sample_model_metadata["model_id"]

        storage_service.save_model_metadata(model_id, sample_model_metadata)

        assert storage_service.model_exists(model_id) is True
        assert storage_service.model_exists("nonexistent") is False

    def test_update_model_metadata(self, storage_service, sample_model_metadata):
        """Test model metadata update."""
        model_id = sample_model_metadata["model_id"]

        storage_service.save_model_metadata(model_id, sample_model_metadata)

        updated = storage_service.update_model_metadata(
            model_id, {"status": "failed", "error_message": "Test error"}
        )

        assert updated["status"] == "failed"
        assert updated["error_message"] == "Test error"

    def test_delete_model(self, storage_service, sample_model_metadata):
        """Test model deletion."""
        model_id = sample_model_metadata["model_id"]

        storage_service.save_model_metadata(model_id, sample_model_metadata)
        assert storage_service.model_exists(model_id) is True

        result = storage_service.delete_model(model_id)
        assert result is True
        assert storage_service.model_exists(model_id) is False

    def test_list_models(self, storage_service, sample_model_metadata):
        """Test listing models."""
        # Initially empty
        assert len(storage_service.list_models()) == 0

        # Add some models
        storage_service.save_model_metadata("model1", sample_model_metadata)
        meta2 = sample_model_metadata.copy()
        meta2["model_id"] = "model2"
        meta2["name"] = "Second Model"
        storage_service.save_model_metadata("model2", meta2)

        models = storage_service.list_models()
        assert len(models) == 2


class TestResultsStorage:
    """Tests for results storage functionality."""

    def test_save_and_load_results(self, storage_service):
        """Test saving and loading results."""
        model_id = "test_model"
        results_type = "summary"
        results = {
            "model_id": model_id,
            "r2": 0.95,
            "mape": 5.2,
        }

        storage_service.save_results(model_id, results_type, results)

        loaded = storage_service.load_results(model_id, results_type)
        assert loaded["r2"] == 0.95
        assert loaded["mape"] == 5.2

    def test_load_nonexistent_results(self, storage_service):
        """Test loading non-existent results."""
        from storage import StorageError

        with pytest.raises(StorageError):
            storage_service.load_results("nonexistent", "summary")
