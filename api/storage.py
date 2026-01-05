"""
Storage service for data, configurations, and models.

Supports local filesystem and S3 backends.
"""

from __future__ import annotations

import hashlib
import json
import cloudpickle as pickle
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
import mmm_framework

import pandas as pd

from config import Settings, get_settings


class StorageError(Exception):
    """Storage operation error."""

    pass


class StorageService:
    """
    Storage service for managing data, configs, and models.

    Supports local filesystem storage with optional S3 backend.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure storage directories exist."""
        if self.settings.storage_backend == "local":
            for subdir in ["data", "configs", "models", "results"]:
                (self.settings.storage_path / subdir).mkdir(parents=True, exist_ok=True)

    def _get_path(self, category: str, item_id: str, ext: str = "") -> Path:
        """Get storage path for an item."""
        return self.settings.storage_path / category / f"{item_id}{ext}"

    @staticmethod
    def generate_id() -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())[:12]

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()[:16]

    # =========================================================================
    # Data Storage
    # =========================================================================

    def save_data(
        self,
        file_content: bytes,
        filename: str,
        data_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Save uploaded data file.

        Parameters
        ----------
        file_content : bytes
            Raw file content.
        filename : str
            Original filename.
        data_id : str, optional
            Custom data ID. Generated if not provided.

        Returns
        -------
        dict
            Metadata about saved data.
        """
        data_id = data_id or self.generate_id()

        # Determine format from filename
        if filename.endswith(".csv"):
            ext = ".csv"
            df = pd.read_csv(pd.io.common.BytesIO(file_content))
        elif filename.endswith(".parquet"):
            ext = ".parquet"
            df = pd.read_parquet(pd.io.common.BytesIO(file_content))
        elif filename.endswith((".xls", ".xlsx")):
            ext = ".xlsx"
            df = pd.read_excel(pd.io.common.BytesIO(file_content))
        else:
            raise StorageError(f"Unsupported file format: {filename}")

        # Save the raw file
        data_path = self._get_path("data", data_id, ext)
        data_path.write_bytes(file_content)

        # Extract metadata
        variables = []
        dimensions = {}

        if "VariableName" in df.columns:
            variables = sorted(df["VariableName"].dropna().unique().tolist())

        dim_cols = ["Geography", "Product", "Campaign", "Outlet", "Creative"]
        for col in dim_cols:
            if col in df.columns:
                values = df[col].dropna().unique().tolist()
                values = [v for v in values if v != "" and pd.notna(v)]
                if values:
                    dimensions[col] = sorted(values)

        # Save metadata
        metadata = {
            "data_id": data_id,
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "variables": variables,
            "dimensions": dimensions,
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": len(file_content),
            "file_ext": ext,
            "hash": self.compute_hash(file_content),
        }

        meta_path = self._get_path("data", f"{data_id}_meta", ".json")
        meta_path.write_text(json.dumps(metadata, indent=2))

        return metadata

    def load_data(self, data_id: str) -> pd.DataFrame:
        """Load data as DataFrame."""
        # Find the data file
        for ext in [".csv", ".parquet", ".xlsx"]:
            data_path = self._get_path("data", data_id, ext)
            if data_path.exists():
                if ext == ".csv":
                    return pd.read_csv(data_path)
                elif ext == ".parquet":
                    return pd.read_parquet(data_path)
                elif ext == ".xlsx":
                    return pd.read_excel(data_path)

        raise StorageError(f"Data not found: {data_id}")

    def get_data_info(self, data_id: str) -> dict[str, Any]:
        """Get data metadata."""
        meta_path = self._get_path("data", f"{data_id}_meta", ".json")
        if not meta_path.exists():
            raise StorageError(f"Data not found: {data_id}")
        return json.loads(meta_path.read_text())

    def list_data(self) -> list[dict[str, Any]]:
        """List all stored datasets."""
        data_dir = self.settings.storage_path / "data"
        datasets = []

        for meta_file in data_dir.glob("*_meta.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                datasets.append(metadata)
            except Exception:
                continue

        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_data(self, data_id: str) -> bool:
        """Delete data and metadata."""
        deleted = False

        # Delete data file
        for ext in [".csv", ".parquet", ".xlsx"]:
            data_path = self._get_path("data", data_id, ext)
            if data_path.exists():
                data_path.unlink()
                deleted = True

        # Delete metadata
        meta_path = self._get_path("data", f"{data_id}_meta", ".json")
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        return deleted

    def data_exists(self, data_id: str) -> bool:
        """Check if data exists."""
        meta_path = self._get_path("data", f"{data_id}_meta", ".json")
        return meta_path.exists()

    # =========================================================================
    # Config Storage
    # =========================================================================

    def save_config(
        self,
        config_data: dict[str, Any],
        config_id: str | None = None,
    ) -> dict[str, Any]:
        """Save model configuration."""
        config_id = config_id or self.generate_id()
        now = datetime.utcnow().isoformat()

        # Add timestamps
        if "created_at" not in config_data:
            config_data["created_at"] = now
        config_data["updated_at"] = now
        config_data["config_id"] = config_id

        # Save config
        config_path = self._get_path("configs", config_id, ".json")
        config_path.write_text(json.dumps(config_data, indent=2, default=str))

        return config_data

    def load_config(self, config_id: str) -> dict[str, Any]:
        """Load configuration."""
        config_path = self._get_path("configs", config_id, ".json")
        if not config_path.exists():
            raise StorageError(f"Config not found: {config_id}")
        return json.loads(config_path.read_text())

    def update_config(self, config_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update configuration."""
        config = self.load_config(config_id)

        # Deep merge updates
        for key, value in updates.items():
            if value is not None:
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value

        return self.save_config(config, config_id)

    def list_configs(self) -> list[dict[str, Any]]:
        """List all configurations."""
        config_dir = self.settings.storage_path / "configs"
        configs = []

        for config_file in config_dir.glob("*.json"):
            try:
                config = json.loads(config_file.read_text())
                configs.append(config)
            except Exception:
                continue

        return sorted(configs, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_config(self, config_id: str) -> bool:
        """Delete configuration."""
        config_path = self._get_path("configs", config_id, ".json")
        if config_path.exists():
            config_path.unlink()
            return True
        return False

    def config_exists(self, config_id: str) -> bool:
        """Check if config exists."""
        config_path = self._get_path("configs", config_id, ".json")
        return config_path.exists()

    # =========================================================================
    # Model Storage
    # =========================================================================

    def save_model_metadata(
        self,
        model_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Save model metadata."""
        metadata["model_id"] = model_id
        meta_path = self._get_path("models", f"{model_id}_meta", ".json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    def update_model_metadata(
        self, model_id: str, updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Update model metadata."""
        metadata = self.get_model_metadata(model_id)
        metadata.update(updates)
        self.save_model_metadata(model_id, metadata)
        return metadata

    def get_model_metadata(self, model_id: str) -> dict[str, Any]:
        """Get model metadata."""
        meta_path = self._get_path("models", f"{model_id}_meta", ".json")
        if not meta_path.exists():
            raise StorageError(f"Model not found: {model_id}")
        return json.loads(meta_path.read_text())

    def save_model_artifact(
        self,
        model_id: str,
        artifact_name: str,
        data: Any,
    ) -> Path:
        """Save a model artifact (e.g., trace, results)."""
        artifact_dir = self.settings.storage_path / "models" / model_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = artifact_dir / f"{artifact_name}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(data, f)

        return artifact_path

    def load_model_artifact(self, model_id: str, artifact_name: str) -> Any:
        """Load a model artifact."""
        artifact_path = (
            self.settings.storage_path / "models" / model_id / f"{artifact_name}.pkl"
        )
        if not artifact_path.exists():
            raise StorageError(f"Artifact not found: {model_id}/{artifact_name}")

        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    def get_model_artifact_path(self, model_id: str, artifact_name: str) -> Path:
        """Get path to model artifact for streaming."""
        artifact_path = (
            self.settings.storage_path / "models" / model_id / f"{artifact_name}.pkl"
        )
        if not artifact_path.exists():
            raise StorageError(f"Artifact not found: {model_id}/{artifact_name}")
        return artifact_path

    def list_models(self) -> list[dict[str, Any]]:
        """List all models."""
        models_dir = self.settings.storage_path / "models"
        models = []

        for meta_file in models_dir.glob("*_meta.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                models.append(metadata)
            except Exception:
                continue

        return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_model(self, model_id: str) -> bool:
        """Delete model and all artifacts."""
        deleted = False

        # Delete metadata
        meta_path = self._get_path("models", f"{model_id}_meta", ".json")
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        # Delete artifacts directory
        artifact_dir = self.settings.storage_path / "models" / model_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
            deleted = True

        return deleted

    def model_exists(self, model_id: str) -> bool:
        """Check if model exists."""
        meta_path = self._get_path("models", f"{model_id}_meta", ".json")
        return meta_path.exists()

    # =========================================================================
    # Results Storage
    # =========================================================================

    def save_results(
        self,
        model_id: str,
        results_type: str,
        results: dict[str, Any],
    ) -> None:
        """Save analysis results."""
        results_dir = self.settings.storage_path / "results" / model_id
        results_dir.mkdir(parents=True, exist_ok=True)

        results_path = results_dir / f"{results_type}.json"
        results_path.write_text(json.dumps(results, indent=2, default=str))

    def load_results(self, model_id: str, results_type: str) -> dict[str, Any]:
        """Load analysis results."""
        results_path = (
            self.settings.storage_path / "results" / model_id / f"{results_type}.json"
        )
        if not results_path.exists():
            raise StorageError(f"Results not found: {model_id}/{results_type}")
        return json.loads(results_path.read_text())


# Global storage instance
_storage: StorageService | None = None


def get_storage() -> StorageService:
    """Get global storage service instance."""
    global _storage
    if _storage is None:
        _storage = StorageService()
    return _storage
