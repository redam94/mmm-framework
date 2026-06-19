"""Shared contracts for external data-source connectors (GCS, BigQuery, …).

A :class:`DataSource` knows how to authenticate to an external system, list the
objects/tables available there, and pull one of them into a pandas DataFrame.
That DataFrame is the only contract the rest of the framework cares about: it is
handed to :func:`mmm_framework.load_mff` (with an ``MFFConfig`` that maps the
source columns to MFF) or written to the agent workspace as the session dataset.

Connectors keep heavy Google SDKs out of import time — they are pulled in lazily
via :func:`require_dependency`, which raises a *helpful* error naming the
optional extra (``mmm-framework[gcp]``) when the SDK is absent. Every connector
also accepts an injected ``client`` so the read/list logic is unit-testable
without real cloud credentials.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


class IntegrationError(RuntimeError):
    """Base class for all data-integration failures."""


class IntegrationAuthError(IntegrationError):
    """Authentication / credential resolution failed."""


class MissingDependencyError(IntegrationError):
    """An optional cloud SDK is needed but not installed."""


def require_dependency(module: str, *, extra: str = "gcp", purpose: str = "") -> Any:
    """Import ``module`` lazily, raising a clear error that names the extra.

    Mirrors the lazy-import convention used by ``agents/llm.py`` for Vertex —
    the core framework must import cleanly without the Google SDKs present.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        tail = f" to {purpose}" if purpose else ""
        raise MissingDependencyError(
            f"{module!r} is required{tail}, but it is not installed. "
            f"Install the optional dependency group:  "
            f"pip install 'mmm-framework[{extra}]'  (or:  uv sync --extra {extra})."
        ) from exc


def scrub_cloud_error(text: str) -> str:
    """Redact identifiers cloud SDK errors echo back — project ids, service-
    account emails, credential file paths — before surfacing them to a client."""
    import re

    text = re.sub(r"projects/[\w-]+", "projects/***", text)
    text = re.sub(
        r"[\w.+-]+@[\w.-]+\.iam\.gserviceaccount\.com",
        "***@***.iam.gserviceaccount.com",
        text,
    )
    text = re.sub(r"(?:/[^\s'\"]+)+\.json", "/***.json", text)
    return text


def dependency_installed(module: str) -> bool:
    """Whether ``module`` can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - namespace edge cases
        return False


@dataclass(frozen=True)
class ConnectionStatus:
    """Result of a connector ``test_connection`` probe."""

    ok: bool
    detail: str
    source: str  # "gcs" | "bigquery"
    target: str = ""  # human description of what was reached (bucket / dataset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "detail": self.detail,
            "source": self.source,
            "target": self.target,
        }


@dataclass(frozen=True)
class ObjectInfo:
    """A listable item in a source: a GCS blob or a BigQuery table."""

    name: str
    kind: str = ""  # "csv" | "parquet" | "table" | …
    size_bytes: int | None = None
    updated: str | None = None
    rows: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "updated": self.updated,
            "rows": self.rows,
        }


class DataSource(ABC):
    """A read-only external tabular data source.

    Subclasses pull data into pandas; everything downstream (MFF loading,
    standardization, modeling) is unchanged. Connectors are deliberately
    stateless beyond their config + (optionally injected) cloud client.
    """

    #: short stable identifier used by the registry / catalog ("gcs", "bigquery")
    kind: str = ""

    @abstractmethod
    def test_connection(self) -> ConnectionStatus:
        """Cheaply verify credentials + reachability."""

    @abstractmethod
    def list_objects(self, **kwargs: Any) -> list[ObjectInfo]:
        """List the blobs / tables available to read."""

    @abstractmethod
    def read_dataframe(self, ref: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Pull one object/table/query into a DataFrame.

        ``ref`` is an object path (GCS) or a table id (BigQuery); BigQuery also
        accepts a ``query=`` keyword instead of ``ref``.
        """
