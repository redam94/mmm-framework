"""Object-store backends: local + S3 (via moto) (P1b)."""

from __future__ import annotations

import pytest

from mmm_framework.storage import (
    LocalObjectStore,
    S3ObjectStore,
    get_object_store,
)


def _roundtrip(store):
    store.put("data/x.csv", b"col\n1\n")
    assert store.exists("data/x.csv")
    assert store.get("data/x.csv") == b"col\n1\n"
    assert "data/x.csv" in store.list()
    assert "data/x.csv" in store.list(prefix="data/")
    assert store.list(prefix="nope/") == []
    store.delete("data/x.csv")
    assert not store.exists("data/x.csv")
    store.delete("data/x.csv")  # idempotent
    with pytest.raises(FileNotFoundError):
        store.get("data/x.csv")


def test_local_roundtrip(tmp_path):
    _roundtrip(LocalObjectStore(tmp_path / "store"))


def test_s3_roundtrip():
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="mmm-test")
        _roundtrip(S3ObjectStore("mmm-test", prefix="artifacts", client=client))


def test_s3_prefix_namespacing():
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="mmm-test")
        store = S3ObjectStore("mmm-test", prefix="tenantA", client=client)
        store.put("m.bin", b"x")
        # The object is actually stored under the prefix in the bucket...
        keys = [o["Key"] for o in client.list_objects_v2(Bucket="mmm-test")["Contents"]]
        assert keys == ["tenantA/m.bin"]
        # ...but the store API returns the prefix-stripped key.
        assert store.list() == ["m.bin"]


def test_factory():
    s = get_object_store("local", root="./_unused_storage_test")
    assert isinstance(s, LocalObjectStore)
    with pytest.raises(ValueError):
        get_object_store("s3")  # missing bucket
    with pytest.raises(ValueError):
        get_object_store("ftp")  # unknown backend
