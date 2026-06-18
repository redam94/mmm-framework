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

# Mirrors mmm_framework.auth.store.DEFAULT_ORG_ID. Records written before tenant
# scoping existed (or while auth was disabled) have no ``org_id`` and are treated
# as belonging to this default org.
DEFAULT_ORG_ID = "org_default"


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
            for subdir in [
                "data",
                "configs",
                "models",
                "results",
                "projects",
                "budget_plans",
            ]:
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
    # Budget Plan Storage
    # =========================================================================

    def save_budget_plan(
        self,
        name: str,
        model_id: str,
        spend_changes: dict[str, Any],
        baseline_outcome: float,
        scenario_outcome: float,
        outcome_change: float,
        outcome_change_pct: float,
        channel_details: dict[str, Any] | None = None,
        description: str | None = None,
        project_id: str | None = None,
        plan_id: str | None = None,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Save a named budget plan with its scenario result."""
        plan_id = plan_id or self.generate_id()
        now = datetime.utcnow().isoformat()
        metadata: dict[str, Any] = {
            "plan_id": plan_id,
            "name": name,
            "description": description,
            "model_id": model_id,
            "spend_changes": spend_changes,
            "baseline_outcome": baseline_outcome,
            "scenario_outcome": scenario_outcome,
            "outcome_change": outcome_change,
            "outcome_change_pct": outcome_change_pct,
            "channel_details": channel_details or {},
            "created_at": now,
            "project_id": project_id,
        }
        if org_id is not None:
            metadata["org_id"] = org_id
        meta_path = self._get_path("budget_plans", f"{plan_id}", ".json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        return metadata

    def get_budget_plan(self, plan_id: str) -> dict[str, Any]:
        """Get a budget plan by ID."""
        meta_path = self._get_path("budget_plans", plan_id, ".json")
        if not meta_path.exists():
            raise StorageError(f"Budget plan not found: {plan_id}")
        return json.loads(meta_path.read_text())

    def list_budget_plans(
        self,
        model_id: str | None = None,
        project_id: str | None = None,
        org_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List budget plans, optionally filtered by model, project, or org."""
        plans_dir = self.settings.storage_path / "budget_plans"
        plans = []
        for plan_file in plans_dir.glob("*.json"):
            try:
                plan = json.loads(plan_file.read_text())
                if model_id is not None and plan.get("model_id") != model_id:
                    continue
                if project_id is not None and plan.get("project_id") != project_id:
                    continue
                if (
                    org_id is not None
                    and (plan.get("org_id") or DEFAULT_ORG_ID) != org_id
                ):
                    continue
                plans.append(plan)
            except Exception:
                continue
        return sorted(plans, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_budget_plan(self, plan_id: str) -> bool:
        """Delete a budget plan."""
        meta_path = self._get_path("budget_plans", plan_id, ".json")
        if meta_path.exists():
            meta_path.unlink()
            return True
        return False

    def budget_plan_exists(self, plan_id: str) -> bool:
        """Check if a budget plan exists."""
        return self._get_path("budget_plans", plan_id, ".json").exists()

    # =========================================================================
    # Project Storage
    # =========================================================================

    def save_project(
        self,
        name: str,
        description: str | None = None,
        project_id: str | None = None,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or overwrite a project record."""
        project_id = project_id or self.generate_id()
        now = datetime.utcnow().isoformat()
        metadata: dict[str, Any] = {
            "project_id": project_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
        }
        if org_id is not None:
            metadata["org_id"] = org_id
        meta_path = self._get_path("projects", f"{project_id}_meta", ".json")
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def get_project_metadata(self, project_id: str) -> dict[str, Any]:
        """Get project metadata."""
        meta_path = self._get_path("projects", f"{project_id}_meta", ".json")
        if not meta_path.exists():
            raise StorageError(f"Project not found: {project_id}")
        return json.loads(meta_path.read_text())

    def update_project_metadata(
        self, project_id: str, updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Update project fields."""
        metadata = self.get_project_metadata(project_id)
        for key, value in updates.items():
            if value is not None:
                metadata[key] = value
        metadata["updated_at"] = datetime.utcnow().isoformat()
        meta_path = self._get_path("projects", f"{project_id}_meta", ".json")
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def list_projects(self, org_id: str | None = None) -> list[dict[str, Any]]:
        """List all projects, optionally filtered by org."""
        projects_dir = self.settings.storage_path / "projects"
        projects = []
        for meta_file in projects_dir.glob("*_meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
                if (
                    org_id is not None
                    and (meta.get("org_id") or DEFAULT_ORG_ID) != org_id
                ):
                    continue
                projects.append(meta)
            except Exception:
                continue
        return sorted(projects, key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete_project(self, project_id: str) -> bool:
        """Delete a project record (does not cascade-delete member resources)."""
        meta_path = self._get_path("projects", f"{project_id}_meta", ".json")
        if meta_path.exists():
            meta_path.unlink()
            return True
        return False

    def project_exists(self, project_id: str) -> bool:
        """Check if a project exists."""
        return self._get_path("projects", f"{project_id}_meta", ".json").exists()

    def count_by_project(self, project_id: str) -> dict[str, int]:
        """Count data/config/model resources that belong to a project."""
        counts: dict[str, int] = {"data_count": 0, "config_count": 0, "model_count": 0}
        for item in self.list_data():
            if item.get("project_id") == project_id:
                counts["data_count"] += 1
        for item in self.list_configs():
            if item.get("project_id") == project_id:
                counts["config_count"] += 1
        for item in self.list_models():
            if item.get("project_id") == project_id:
                counts["model_count"] += 1
        return counts

    # =========================================================================
    # Data Storage
    # =========================================================================

    def save_data(
        self,
        file_content: bytes,
        filename: str,
        data_id: str | None = None,
        project_id: str | None = None,
        org_id: str | None = None,
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
            "project_id": project_id,
        }
        if org_id is not None:
            metadata["org_id"] = org_id

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

    def list_data(
        self, project_id: str | None = None, org_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List stored datasets, optionally filtered by project or org."""
        data_dir = self.settings.storage_path / "data"
        datasets = []

        for meta_file in data_dir.glob("*_meta.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                if project_id is not None and metadata.get("project_id") != project_id:
                    continue
                if (
                    org_id is not None
                    and (metadata.get("org_id") or DEFAULT_ORG_ID) != org_id
                ):
                    continue
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
        project_id: str | None = None,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Save model configuration."""
        config_id = config_id or self.generate_id()
        now = datetime.utcnow().isoformat()

        # Add timestamps
        if "created_at" not in config_data:
            config_data["created_at"] = now
        config_data["updated_at"] = now
        config_data["config_id"] = config_id
        if project_id is not None:
            config_data["project_id"] = project_id
        if org_id is not None:
            config_data["org_id"] = org_id

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

    def list_configs(
        self, project_id: str | None = None, org_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List configurations, optionally filtered by project or org."""
        config_dir = self.settings.storage_path / "configs"
        configs = []

        for config_file in config_dir.glob("*.json"):
            try:
                config = json.loads(config_file.read_text())
                if project_id is not None and config.get("project_id") != project_id:
                    continue
                if (
                    org_id is not None
                    and (config.get("org_id") or DEFAULT_ORG_ID) != org_id
                ):
                    continue
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
        project_id: str | None = None,
        org_id: str | None = None,
    ) -> None:
        """Save model metadata."""
        metadata["model_id"] = model_id
        if project_id is not None:
            metadata["project_id"] = project_id
        if org_id is not None:
            metadata["org_id"] = org_id
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

    def list_models(
        self, project_id: str | None = None, org_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List models, optionally filtered by project or org."""
        models_dir = self.settings.storage_path / "models"
        models = []

        for meta_file in models_dir.glob("*_meta.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                if project_id is not None and metadata.get("project_id") != project_id:
                    continue
                if (
                    org_id is not None
                    and (metadata.get("org_id") or DEFAULT_ORG_ID) != org_id
                ):
                    continue
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


def org_scope(principal) -> str | None:
    """Return the org id to scope a query to, or None for the dev principal.

    ``principal`` is an ``mmm_framework.auth.models.AuthContext``. When auth is
    disabled (dev/single-tenant) ``is_dev`` is True and scoping is skipped so
    existing flows are unchanged.
    """
    return None if getattr(principal, "is_dev", False) else principal.org_id


def assert_org_owns(stored_org: str | None, org: str | None) -> None:
    """Raise 404 unless ``org`` (when set) owns the record.

    Records with no ``org_id`` are treated as belonging to ``DEFAULT_ORG_ID``.
    Imported lazily so ``storage`` stays importable without fastapi at module
    load if ever needed; fastapi is already a hard dep of the API though.
    """
    if org is not None and (stored_org or DEFAULT_ORG_ID) != org:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


def backfill_org_id(storage: "StorageService", org_id: str) -> dict[str, int]:
    """Stamp ``org_id`` into every org-less on-disk record.

    One-time migration to run BEFORE enabling auth on an existing single-tenant
    install, so its pre-existing data stays visible to its tenant org (records
    written before tenant scoping have no ``org_id`` and otherwise default to
    ``DEFAULT_ORG_ID``). Idempotent: only records missing an ``org_id`` are
    touched. Returns a per-category count of records stamped.

    Run via: ``python api/backfill_org.py <org_id>`` (from the api/ dir).
    """
    sp = storage.settings.storage_path
    # (subdir, glob) — data/models/projects keep metadata in *_meta.json;
    # configs/budget_plans are the JSON record itself.
    specs = [
        ("data", "*_meta.json"),
        ("configs", "*.json"),
        ("models", "*_meta.json"),
        ("projects", "*_meta.json"),
        ("budget_plans", "*.json"),
    ]
    counts: dict[str, int] = {}
    for sub, pat in specs:
        d = sp / sub
        n = 0
        if d.exists():
            for f in sorted(d.glob(pat)):
                try:
                    meta = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(meta, dict) and not meta.get("org_id"):
                    meta["org_id"] = org_id
                    f.write_text(json.dumps(meta, indent=2, default=str))
                    n += 1
        counts[sub] = n
    return counts


# Global storage instance
_storage: StorageService | None = None


def get_storage() -> StorageService:
    """Get global storage service instance."""
    global _storage
    if _storage is None:
        _storage = StorageService()
    return _storage
