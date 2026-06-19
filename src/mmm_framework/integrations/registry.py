"""Registry + catalog for external data sources.

``list_data_sources`` powers the Settings "Data connections" UI (what's
available + whether the SDK is installed). ``build_data_source`` instantiates a
connector from a ``kind`` + a config dict, validated through its pydantic model.
"""

from __future__ import annotations

from typing import Any

from .base import DataSource, IntegrationError, dependency_installed
from .bigquery import BigQueryConfig, BigQueryDataSource
from .gcs import GCSConfig, GCSDataSource

_REGISTRY: dict[str, dict[str, Any]] = {
    "gcs": {
        "label": "Google Cloud Storage",
        "description": "Read CSV or Parquet objects from a GCS bucket.",
        "source_cls": GCSDataSource,
        "config_cls": GCSConfig,
        "probe_module": "google.cloud.storage",
        "auth": "Application Default Credentials (ADC), or a service-account JSON key.",
        "fields": [
            {
                "name": "bucket",
                "label": "Bucket",
                "required": True,
                "placeholder": "my-mmm-data",
            },
            {
                "name": "prefix",
                "label": "Object prefix",
                "required": False,
                "placeholder": "exports/",
            },
            {
                "name": "project",
                "label": "GCP project",
                "required": False,
                "placeholder": "(ADC default)",
            },
            {
                "name": "credentials_path",
                "label": "Service-account key path",
                "required": False,
                "placeholder": "(empty = ADC)",
            },
        ],
    },
    "bigquery": {
        "label": "BigQuery",
        "description": "Run a query or read a table into a DataFrame.",
        "source_cls": BigQueryDataSource,
        "config_cls": BigQueryConfig,
        "probe_module": "google.cloud.bigquery",
        "auth": "Application Default Credentials (ADC), or a service-account JSON key.",
        "fields": [
            {
                "name": "project",
                "label": "GCP project",
                "required": False,
                "placeholder": "(ADC default)",
            },
            {
                "name": "dataset",
                "label": "Default dataset",
                "required": False,
                "placeholder": "marketing",
            },
            {
                "name": "location",
                "label": "Location",
                "required": False,
                "placeholder": "US",
            },
            {
                "name": "credentials_path",
                "label": "Service-account key path",
                "required": False,
                "placeholder": "(empty = ADC)",
            },
        ],
    },
}


def data_source_kinds() -> list[str]:
    return list(_REGISTRY)


def build_data_source(
    kind: str, config: dict[str, Any] | Any, *, client: Any | None = None
) -> DataSource:
    """Instantiate a connector from a ``kind`` and a config dict/model."""
    spec = _REGISTRY.get(kind)
    if spec is None:
        raise IntegrationError(
            f"Unknown data source {kind!r}; known kinds: {', '.join(_REGISTRY)}"
        )
    cfg_cls = spec["config_cls"]
    cfg = config if isinstance(config, cfg_cls) else cfg_cls(**(config or {}))
    return spec["source_cls"](cfg, client=client)


def list_data_sources() -> list[dict[str, Any]]:
    """Catalog metadata for every registered data source (UI-friendly)."""
    out: list[dict[str, Any]] = []
    for kind, spec in _REGISTRY.items():
        out.append(
            {
                "kind": kind,
                "label": spec["label"],
                "description": spec["description"],
                "auth": spec["auth"],
                "installed": dependency_installed(spec["probe_module"]),
                "install_extra": "gcp",
                "fields": spec["fields"],
            }
        )
    return out
