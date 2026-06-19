"""BigQuery data source: run a query or read a table into pandas.

Auth is ADC-first (shared with GCS + the Vertex LLM). The client is built lazily
or injected for testing. A ``maximum_bytes_billed`` guard is applied to every
query so a careless ``SELECT *`` cannot run away with cost.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from .base import (
    ConnectionStatus,
    DataSource,
    IntegrationError,
    ObjectInfo,
    require_dependency,
)
from .credentials import load_gcp_credentials, resolve_project

# Fully-qualified or dataset.table identifier (BigQuery legal chars only).
_TABLE_RE = re.compile(r"^[A-Za-z0-9_\-]+(\.[A-Za-z0-9_\-]+){1,2}$")


class BigQueryConfig(BaseModel):
    """Connection settings for BigQuery."""

    project: str | None = Field(
        None, description="GCP project (defaults to ADC project)"
    )
    dataset: str | None = Field(
        None, description="Default dataset for table reads/listing"
    )
    location: str | None = Field(
        None, description="BigQuery location, e.g. US / EU / us-central1"
    )
    credentials_path: str | None = Field(
        None, description="Service-account JSON path; empty = ADC"
    )
    maximum_bytes_billed: int | None = Field(
        2 * 1024 * 1024 * 1024,
        description="Per-query byte ceiling (default 2 GiB); None disables the guard",
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> BigQueryConfig:
        data = {
            "project": resolve_project(),
            "dataset": os.environ.get("MMM_BIGQUERY_DATASET"),
            "location": os.environ.get("MMM_BIGQUERY_LOCATION"),
            "credentials_path": os.environ.get("MMM_GCP_CREDENTIALS_PATH"),
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class BigQueryDataSource(DataSource):
    """Read query results / tables from BigQuery as DataFrames."""

    kind = "bigquery"

    def __init__(
        self, config: BigQueryConfig | None = None, *, client: Any | None = None
    ) -> None:
        self.config = config or BigQueryConfig()
        self._client = client

    # -- client -----------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            bq = require_dependency("google.cloud.bigquery", purpose="query BigQuery")
            self._client = bq.Client(
                project=resolve_project(self.config.project),
                credentials=load_gcp_credentials(self.config.credentials_path),
                location=self.config.location,
            )
        return self._client

    def _job_config(self, **kwargs: Any) -> Any:
        # Soft import: when the SDK is present we attach a real cost guard; when
        # it is absent (e.g. an injected test client), pass job_config=None.
        import importlib

        try:
            bq = importlib.import_module("google.cloud.bigquery")
        except ImportError:
            return None
        cfg = bq.QueryJobConfig(**kwargs)
        if self.config.maximum_bytes_billed is not None:
            cfg.maximum_bytes_billed = self.config.maximum_bytes_billed
        return cfg

    # -- operations -------------------------------------------------------------
    def test_connection(self) -> ConnectionStatus:
        target = self.config.project or "(ADC project)"
        try:
            job = self.client.query("SELECT 1 AS ok", job_config=self._job_config())
            list(job.result())  # force execution
            return ConnectionStatus(
                True, "Reachable; SELECT 1 succeeded", self.kind, target
            )
        except Exception as exc:  # noqa: BLE001
            return ConnectionStatus(
                False, f"{type(exc).__name__}: {exc}", self.kind, target
            )

    def list_objects(
        self, dataset: str | None = None, max_results: int = 200
    ) -> list[ObjectInfo]:
        ds = dataset or self.config.dataset
        if not ds:
            raise IntegrationError(
                "list_objects requires a dataset (config.dataset or dataset=…)"
            )
        tables = self.client.list_tables(ds, max_results=max_results)
        out: list[ObjectInfo] = []
        for t in tables:
            tid = getattr(t, "table_id", None) or str(t)
            out.append(
                ObjectInfo(
                    name=f"{ds}.{tid}",
                    kind="table",
                    rows=getattr(t, "num_rows", None),
                )
            )
        return out

    def read_dataframe(
        self,
        ref: str | None = None,
        *,
        query: str | None = None,
        max_rows: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read from a ``query=`` or a table ``ref`` (``dataset.table``).

        A bare table ref becomes ``SELECT * FROM `ref` [LIMIT max_rows]``.
        """
        sql = self._resolve_sql(ref, query, max_rows)
        job = self.client.query(sql, job_config=self._job_config())
        return job.result().to_dataframe(**kwargs)

    def _resolve_sql(
        self, ref: str | None, query: str | None, max_rows: int | None
    ) -> str:
        if query and ref:
            raise IntegrationError("Pass either ref=<table> or query=<sql>, not both")
        if query:
            return query
        if not ref:
            raise IntegrationError("read_dataframe requires ref=<table> or query=<sql>")
        table = ref
        if "." not in table and self.config.dataset:
            table = f"{self.config.dataset}.{table}"
        if not _TABLE_RE.match(table):
            raise IntegrationError(
                f"{ref!r} is not a valid table id; pass query=<sql> for arbitrary SQL"
            )
        limit = f" LIMIT {int(max_rows)}" if max_rows else ""
        return f"SELECT * FROM `{table}`{limit}"
