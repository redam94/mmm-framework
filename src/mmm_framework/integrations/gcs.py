"""Google Cloud Storage data source: read CSV/Parquet objects into pandas.

Reads are streamed into memory (``download_as_bytes``) and parsed by pandas
based on the object extension. The client is built lazily with ADC-first auth,
or injected for testing.
"""

from __future__ import annotations

import io
import os
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

_PARQUET_EXT = (".parquet", ".pq")
_CSV_EXT = (".csv", ".csv.gz", ".tsv", ".txt")


class GCSConfig(BaseModel):
    """Connection settings for a GCS bucket."""

    bucket: str = Field(..., description="GCS bucket name (no gs:// prefix)")
    prefix: str = Field("", description="Optional object-name prefix to scope listings")
    project: str | None = Field(
        None, description="GCP project (defaults to ADC project)"
    )
    credentials_path: str | None = Field(
        None, description="Service-account JSON path; empty = ADC"
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> GCSConfig:
        """Build from ``MMM_GCS_*`` / ``MMM_GCP_*`` env vars, with overrides."""
        data = {
            "bucket": os.environ.get("MMM_GCS_BUCKET", ""),
            "prefix": os.environ.get("MMM_GCS_PREFIX", ""),
            "project": resolve_project(),
            "credentials_path": os.environ.get("MMM_GCP_CREDENTIALS_PATH"),
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class GCSDataSource(DataSource):
    """Read objects from a GCS bucket as DataFrames."""

    kind = "gcs"

    def __init__(self, config: GCSConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    # -- client -----------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            storage = require_dependency(
                "google.cloud.storage", purpose="read from Google Cloud Storage"
            )
            self._client = storage.Client(
                project=resolve_project(self.config.project),
                credentials=load_gcp_credentials(self.config.credentials_path),
            )
        return self._client

    # -- operations -------------------------------------------------------------
    def test_connection(self) -> ConnectionStatus:
        try:
            # A bounded listing is the cheapest reachability + auth probe.
            blobs = list(
                self.client.list_blobs(
                    self.config.bucket, prefix=self.config.prefix or None, max_results=1
                )
            )
            detail = "Reachable" + (
                f"; sample object: {blobs[0].name}" if blobs else "; bucket empty"
            )
            return ConnectionStatus(
                True, detail, self.kind, f"gs://{self.config.bucket}"
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/auth error uniformly
            return ConnectionStatus(
                False,
                f"{type(exc).__name__}: {exc}",
                self.kind,
                f"gs://{self.config.bucket}",
            )

    def list_objects(
        self, prefix: str | None = None, max_results: int = 200
    ) -> list[ObjectInfo]:
        eff_prefix = prefix if prefix is not None else (self.config.prefix or None)
        blobs = self.client.list_blobs(
            self.config.bucket, prefix=eff_prefix, max_results=max_results
        )
        out: list[ObjectInfo] = []
        for b in blobs:
            name = b.name
            if name.endswith("/"):  # skip directory placeholders
                continue
            out.append(
                ObjectInfo(
                    name=name,
                    kind=_infer_fmt(name) or "",
                    size_bytes=getattr(b, "size", None),
                    updated=_iso(getattr(b, "updated", None)),
                )
            )
        return out

    def read_dataframe(
        self,
        ref: str | None = None,
        *,
        fmt: str | None = None,
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        if not ref:
            raise IntegrationError("read_dataframe(ref=…) requires an object path")
        blob = self.client.bucket(self.config.bucket).blob(ref)
        raw = blob.download_as_bytes()
        resolved = (fmt or _infer_fmt(ref) or "csv").lower()
        buf = io.BytesIO(raw)
        if resolved == "parquet":
            return pd.read_parquet(buf, **read_kwargs)
        if resolved == "tsv":
            read_kwargs.setdefault("sep", "\t")
        return pd.read_csv(buf, **read_kwargs)


def _infer_fmt(name: str) -> str | None:
    low = name.lower()
    if low.endswith(_PARQUET_EXT):
        return "parquet"
    if low.endswith(".tsv"):
        return "tsv"
    if low.endswith(_CSV_EXT):
        return "csv"
    return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)
