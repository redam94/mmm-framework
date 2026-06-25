"""Content-addressed dataset lineage (G1)."""

from __future__ import annotations

import pytest

from mmm_framework.lineage import DatasetLineage, content_hash
from mmm_framework.storage import LocalObjectStore


def _lin(tmp_path):
    return DatasetLineage(LocalObjectStore(tmp_path / "lineage"))


def test_snapshot_and_resolve_roundtrip(tmp_path):
    lin = _lin(tmp_path)
    data = b"Period,Sales\n2021-01-01,100\n"
    h = lin.snapshot(data)
    assert h == content_hash(data)
    assert lin.exists(h)
    assert lin.resolve(h) == data
    assert lin.verify(h)


def test_dedup_identical_content(tmp_path):
    lin = _lin(tmp_path)
    data = b"same bytes"
    h1 = lin.snapshot(data)
    h2 = lin.snapshot(data)  # second snapshot of identical content
    assert h1 == h2
    # Stored exactly once (dedup).
    assert lin.store.list() == [DatasetLineage._key(h1)]


def test_different_content_different_hash(tmp_path):
    lin = _lin(tmp_path)
    assert lin.snapshot(b"a") != lin.snapshot(b"b")


def test_snapshot_from_path(tmp_path):
    lin = _lin(tmp_path)
    f = tmp_path / "data.csv"
    f.write_bytes(b"x,y\n1,2\n")
    h = lin.snapshot(f)
    assert lin.resolve(h) == b"x,y\n1,2\n"


def test_resolve_missing_raises(tmp_path):
    lin = _lin(tmp_path)
    with pytest.raises(FileNotFoundError):
        lin.resolve("0" * 64)
    assert lin.verify("0" * 64) is False


def test_overwrite_source_still_resolves_old_bytes(tmp_path):
    """The point of G1: the model's snapshot survives the file being overwritten."""
    lin = _lin(tmp_path)
    f = tmp_path / "client.csv"
    f.write_bytes(b"v1 data")
    h = lin.snapshot(f)
    f.write_bytes(b"v2 data REDELIVERED")  # client re-delivers under the same name
    assert lin.resolve(h) == b"v1 data"  # old number still reproducible
