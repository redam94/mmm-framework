"""Wiring: /integrations/catalog endpoint + the load_from_* agent tools."""

from __future__ import annotations

import asyncio

from langgraph.types import Command

from mmm_framework.agents.tools import TOOLS, load_from_bigquery, load_from_gcs
from mmm_framework.integrations import dependency_installed


def _message(result: Command) -> str:
    assert isinstance(result, Command)
    return result.update["messages"][0].content


def test_catalog_endpoint_payload():
    from mmm_framework.api import main as M

    out = asyncio.run(M.integrations_catalog_endpoint())
    kinds = {d["kind"] for d in out["data_sources"]}
    assert kinds == {"gcs", "bigquery"}
    platforms = {p["platform"] for p in out["ad_platforms"]}
    assert platforms == {"google_ads", "meta_ads", "tiktok_ads"}


def test_tools_registered():
    names = {t.name for t in TOOLS}
    assert {"load_from_bigquery", "load_from_gcs"} <= names


def test_bigquery_tool_degrades_without_sdk():
    result = load_from_bigquery.func(
        state={}, query="SELECT 1", tool_call_id="t1", config=None
    )
    msg = _message(result)
    # No dataset was set on the error path.
    assert "dataset_path" not in result.update
    if not dependency_installed("google.cloud.bigquery"):
        assert "mmm-framework[gcp]" in msg


def test_gcs_tool_requires_bucket(monkeypatch):
    monkeypatch.delenv("MMM_GCS_BUCKET", raising=False)
    result = load_from_gcs.func(
        state={}, object_path="x.csv", tool_call_id="t2", config=None
    )
    msg = _message(result)
    assert "bucket" in msg.lower()


def test_gcs_tool_degrades_without_sdk():
    result = load_from_gcs.func(
        state={}, object_path="x.csv", bucket="b", tool_call_id="t3", config=None
    )
    msg = _message(result)
    if not dependency_installed("google.cloud.storage"):
        assert "mmm-framework[gcp]" in msg
