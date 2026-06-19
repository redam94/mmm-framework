"""External data integrations for the MMM framework.

Connectors pull tabular data from cloud systems into pandas; the DataFrame then
feeds the existing MFF loader / agent workspace unchanged. Google Cloud SDKs are
optional (``pip install 'mmm-framework[gcp]'``) and imported lazily.

    from mmm_framework.integrations import build_data_source
    src = build_data_source("bigquery", {"project": "acme", "dataset": "mmm"})
    df = src.read_dataframe(query="SELECT * FROM mmm.weekly_spend")
"""

from __future__ import annotations

from .base import (
    ConnectionStatus,
    DataSource,
    IntegrationAuthError,
    IntegrationError,
    MissingDependencyError,
    ObjectInfo,
    dependency_installed,
    require_dependency,
)
from .bigquery import BigQueryConfig, BigQueryDataSource
from .connections import read_connection_dataframe, probe_connection
from .credentials import load_gcp_credentials, resolve_project
from .gcs import GCSConfig, GCSDataSource
from .registry import build_data_source, data_source_kinds, list_data_sources

__all__ = [
    "ConnectionStatus",
    "DataSource",
    "ObjectInfo",
    "IntegrationError",
    "IntegrationAuthError",
    "MissingDependencyError",
    "require_dependency",
    "dependency_installed",
    "GCSConfig",
    "GCSDataSource",
    "BigQueryConfig",
    "BigQueryDataSource",
    "load_gcp_credentials",
    "resolve_project",
    "build_data_source",
    "data_source_kinds",
    "list_data_sources",
    "read_connection_dataframe",
    "probe_connection",
]
