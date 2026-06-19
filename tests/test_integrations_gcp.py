"""Unit tests for the GCS + BigQuery data-source connectors.

These run WITHOUT the Google SDKs installed: connectors accept an injected
client, so the read/list/test logic is exercised against fakes. Only the
parquet path needs pyarrow and is skipped when absent.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from mmm_framework.integrations import (
    BigQueryConfig,
    BigQueryDataSource,
    GCSConfig,
    GCSDataSource,
    IntegrationAuthError,
    IntegrationError,
    MissingDependencyError,
    build_data_source,
    list_data_sources,
    load_gcp_credentials,
    require_dependency,
    resolve_project,
)

# ── Fakes ──────────────────────────────────────────────────────────────────────


class _FakeBlob:
    def __init__(self, name: str, data: bytes = b"", size: int | None = None):
        self.name = name
        self._data = data
        self.size = size if size is not None else len(data)
        self.updated = None

    def download_as_bytes(self) -> bytes:
        return self._data


class _FakeBucketHandle:
    def __init__(self, blobs: dict[str, _FakeBlob]):
        self._blobs = blobs

    def blob(self, name: str) -> _FakeBlob:
        return self._blobs[name]


class _FakeStorageClient:
    def __init__(self, blobs: list[_FakeBlob], *, raise_on_list: bool = False):
        self._blobs = blobs
        self._raise = raise_on_list

    def list_blobs(self, bucket, prefix=None, max_results=None):
        if self._raise:
            raise RuntimeError("boom")
        items = [b for b in self._blobs if prefix is None or b.name.startswith(prefix)]
        return items[:max_results] if max_results else items

    def bucket(self, name):
        return _FakeBucketHandle({b.name: b for b in self._blobs})


class _FakeResult:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __iter__(self):
        return iter(self._df.to_dict("records"))

    def to_dataframe(self, **kwargs):
        return self._df


class _FakeJob:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def result(self):
        return _FakeResult(self._df)


class _FakeTable:
    def __init__(self, table_id: str, num_rows: int):
        self.table_id = table_id
        self.num_rows = num_rows


class _FakeBQClient:
    def __init__(self, df: pd.DataFrame, *, tables=None, raise_on_query: bool = False):
        self._df = df
        self._tables = tables or []
        self._raise = raise_on_query
        self.last_sql: str | None = None

    def query(self, sql, job_config=None):
        if self._raise:
            raise RuntimeError("denied")
        self.last_sql = sql
        return _FakeJob(self._df)

    def list_tables(self, dataset, max_results=None):
        return self._tables[:max_results] if max_results else self._tables


# ── Catalog / registry ──────────────────────────────────────────────────────────


def test_catalog_lists_both_sources():
    cat = {c["kind"]: c for c in list_data_sources()}
    assert set(cat) == {"gcs", "bigquery"}
    for c in cat.values():
        assert isinstance(c["installed"], bool)
        assert c["install_extra"] == "gcp"
        assert c["fields"] and all("name" in f for f in c["fields"])


def test_build_data_source_factory():
    src = build_data_source("gcs", {"bucket": "b"})
    assert isinstance(src, GCSDataSource) and src.config.bucket == "b"
    bq = build_data_source("bigquery", {"project": "p", "dataset": "d"})
    assert isinstance(bq, BigQueryDataSource)
    with pytest.raises(IntegrationError):
        build_data_source("snowflake", {})


# ── GCS ─────────────────────────────────────────────────────────────────────────


def test_gcs_read_csv_with_injected_client():
    csv = b"Period,VariableName,VariableValue\n2023-01-02,TV,5000\n"
    client = _FakeStorageClient([_FakeBlob("exports/data.csv", csv)])
    src = GCSDataSource(GCSConfig(bucket="b", prefix="exports/"), client=client)
    df = src.read_dataframe("exports/data.csv")
    assert list(df.columns) == ["Period", "VariableName", "VariableValue"]
    assert df.iloc[0]["VariableValue"] == 5000


def test_gcs_read_parquet_with_injected_client():
    pytest.importorskip("pyarrow")
    df_in = pd.DataFrame({"Period": ["2023-01-02"], "VariableValue": [5000]})
    buf = io.BytesIO()
    df_in.to_parquet(buf)
    client = _FakeStorageClient([_FakeBlob("d.parquet", buf.getvalue())])
    src = GCSDataSource(GCSConfig(bucket="b"), client=client)
    df = src.read_dataframe("d.parquet")
    assert df.iloc[0]["VariableValue"] == 5000


def test_gcs_list_objects_and_infers_fmt():
    client = _FakeStorageClient(
        [_FakeBlob("a.csv", b"x"), _FakeBlob("b.parquet", b"y"), _FakeBlob("dir/", b"")]
    )
    src = GCSDataSource(GCSConfig(bucket="b"), client=client)
    objs = {o.name: o for o in src.list_objects()}
    assert objs["a.csv"].kind == "csv"
    assert objs["b.parquet"].kind == "parquet"
    assert "dir/" not in objs  # directory placeholder skipped


def test_gcs_read_requires_ref():
    src = GCSDataSource(GCSConfig(bucket="b"), client=_FakeStorageClient([]))
    with pytest.raises(IntegrationError):
        src.read_dataframe()


def test_gcs_test_connection_ok_and_error():
    ok = GCSDataSource(
        GCSConfig(bucket="b"), client=_FakeStorageClient([_FakeBlob("a.csv", b"x")])
    )
    status = ok.test_connection()
    assert status.ok and status.source == "gcs" and status.target == "gs://b"

    bad = GCSDataSource(
        GCSConfig(bucket="b"), client=_FakeStorageClient([], raise_on_list=True)
    )
    status = bad.test_connection()
    assert not status.ok and "boom" in status.detail


# ── BigQuery ────────────────────────────────────────────────────────────────────


def test_bigquery_read_table_builds_select(monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    client = _FakeBQClient(df)
    src = BigQueryDataSource(BigQueryConfig(dataset="mmm"), client=client)
    out = src.read_dataframe("weekly")  # dataset prefilled
    assert client.last_sql == "SELECT * FROM `mmm.weekly`"
    assert out.equals(df)


def test_bigquery_read_query_passthrough_and_limit():
    df = pd.DataFrame({"a": [1]})
    client = _FakeBQClient(df)
    src = BigQueryDataSource(BigQueryConfig(), client=client)
    src.read_dataframe(query="SELECT 1")
    assert client.last_sql == "SELECT 1"

    src.read_dataframe("proj.ds.tbl", max_rows=10)
    assert client.last_sql == "SELECT * FROM `proj.ds.tbl` LIMIT 10"


def test_bigquery_rejects_both_ref_and_query_and_bad_table():
    src = BigQueryDataSource(BigQueryConfig(), client=_FakeBQClient(pd.DataFrame()))
    with pytest.raises(IntegrationError):
        src.read_dataframe("t", query="SELECT 1")
    with pytest.raises(IntegrationError):
        src.read_dataframe("not a table name")  # no dataset, invalid id
    with pytest.raises(IntegrationError):
        src.read_dataframe()  # neither


def test_bigquery_rejects_newline_and_negative_limit():
    src = BigQueryDataSource(
        BigQueryConfig(dataset="mmm"), client=_FakeBQClient(pd.DataFrame())
    )
    # Trailing newline must NOT pass the \Z-anchored regex (SQL-injection vector).
    with pytest.raises(IntegrationError):
        src.read_dataframe("mmm.weekly\n")
    with pytest.raises(IntegrationError):
        src.read_dataframe("weekly\nUNION SELECT 1")
    # Negative LIMIT is rejected before it reaches SQL.
    with pytest.raises(IntegrationError):
        src.read_dataframe("weekly", max_rows=-1)


def test_bigquery_list_tables():
    client = _FakeBQClient(
        pd.DataFrame(), tables=[_FakeTable("weekly", 52), _FakeTable("geo", 520)]
    )
    src = BigQueryDataSource(BigQueryConfig(dataset="mmm"), client=client)
    names = [o.name for o in src.list_objects()]
    assert names == ["mmm.weekly", "mmm.geo"]
    with pytest.raises(IntegrationError):
        BigQueryDataSource(BigQueryConfig(), client=client).list_objects()  # no dataset


def test_bigquery_test_connection_ok_and_error():
    ok = BigQueryDataSource(
        BigQueryConfig(project="p"), client=_FakeBQClient(pd.DataFrame({"ok": [1]}))
    )
    assert ok.test_connection().ok
    bad = BigQueryDataSource(
        BigQueryConfig(project="p"),
        client=_FakeBQClient(pd.DataFrame(), raise_on_query=True),
    )
    status = bad.test_connection()
    assert not status.ok and "denied" in status.detail


# ── Credentials / deps ──────────────────────────────────────────────────────────


def test_load_credentials_adc_when_no_path(monkeypatch):
    monkeypatch.delenv("MMM_GCP_CREDENTIALS_PATH", raising=False)
    assert load_gcp_credentials(None) is None  # ADC


def test_load_credentials_missing_file_raises(monkeypatch):
    monkeypatch.delenv("MMM_GCP_CREDENTIALS_PATH", raising=False)
    with pytest.raises(IntegrationAuthError):
        load_gcp_credentials("/nonexistent/key.json")


def test_resolve_project_env_precedence(monkeypatch):
    for v in (
        "MMM_GCP_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GCP_PROJECT",
    ):
        monkeypatch.delenv(v, raising=False)
    assert resolve_project() is None
    assert resolve_project("explicit") == "explicit"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-env")
    assert resolve_project() == "from-env"
    monkeypatch.setenv("MMM_GCP_PROJECT", "preferred")
    assert resolve_project() == "preferred"


def test_require_dependency_helpful_error():
    with pytest.raises(MissingDependencyError) as ei:
        require_dependency("definitely_not_a_real_module_xyz", purpose="do a thing")
    assert "mmm-framework[gcp]" in str(ei.value)
