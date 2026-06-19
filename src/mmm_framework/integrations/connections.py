"""Read / test a *saved* data-source connection.

A saved connection is ``{kind, config}`` where ``config`` is a directly-readable
reference: for BigQuery an embedded ``query`` or ``table``; for GCS an
``object``. These helpers split that reference out of the connector config and
drive the underlying :class:`DataSource`. They take an injectable ``client`` so
the routing logic is unit-testable without real cloud credentials.

No credentials live in ``config`` — auth is ambient (ADC / the server's
``MMM_GCP_CREDENTIALS_PATH``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import ConnectionStatus, IntegrationError
from .registry import build_data_source

# Reference keys that select WHAT to read — stripped before building the config.
_REF_KEYS = ("query", "table", "object", "object_path")


def _split_ref(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(config or {})
    ref = {k: cfg.pop(k) for k in _REF_KEYS if k in cfg}
    return cfg, ref


def read_connection_dataframe(
    kind: str,
    config: dict[str, Any],
    *,
    max_rows: int | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Pull the DataFrame a saved connection points at."""
    cfg, ref = _split_ref(config)
    if kind == "bigquery":
        src = build_data_source("bigquery", cfg, client=client)
        return src.read_dataframe(
            ref.get("table"), query=ref.get("query"), max_rows=max_rows
        )
    if kind == "gcs":
        obj = ref.get("object") or ref.get("object_path")
        if not obj:
            raise IntegrationError(
                "GCS connection needs an 'object' (the object path to read)"
            )
        src = build_data_source("gcs", cfg, client=client)
        return src.read_dataframe(obj)
    raise IntegrationError(f"Cannot read from a connection of kind {kind!r}")


def probe_connection(
    kind: str, config: dict[str, Any], *, client: Any | None = None
) -> ConnectionStatus:
    """Probe a saved connection's reachability (ignores the read reference)."""
    cfg, _ = _split_ref(config)
    return build_data_source(kind, cfg, client=client).test_connection()
