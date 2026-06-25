"""Pluggable object store: local filesystem or S3.

A small, dependency-light abstraction so artifacts (datasets, models, reports)
can live on shared object storage in a multi-replica deployment instead of a
pod-local disk — closing the "replicas:2 against pod-local files" gap. Local is
the default and needs nothing; S3 is opt-in (the ``s3`` extra: ``uv sync
--extra s3``) and lazy-imports boto3, so importing this module never requires it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStore(ABC):
    """Byte blob store keyed by string path (``a/b/c.bin``)."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the blob; raise ``FileNotFoundError`` if absent."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the blob (no error if it does not exist)."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Keys under ``prefix`` (sorted)."""


class LocalObjectStore(ObjectStore):
    """Filesystem-backed store rooted at ``root``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise FileNotFoundError(key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_file():
            p.unlink()

    def list(self, prefix: str = "") -> list[str]:
        out = []
        for f in self.root.rglob("*"):
            if f.is_file():
                rel = f.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)


class S3ObjectStore(ObjectStore):
    """S3-backed store. Pass an existing boto3 client (e.g. for tests) or let it
    construct one from region/credentials. Keys are namespaced under ``prefix``."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        client=None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover - exercised via the extra
                raise ImportError(
                    "S3ObjectStore requires boto3 — install the 's3' extra: "
                    "uv sync --extra s3 (or pip install 'mmm-framework[s3]')."
                ) from e
            kwargs: dict = {}
            if region:
                kwargs["region_name"] = region
            if access_key and secret_key:
                kwargs["aws_access_key_id"] = access_key
                kwargs["aws_secret_access_key"] = secret_key
            self._client = boto3.client("s3", **kwargs)

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _strip(self, full_key: str) -> str:
        if self.prefix and full_key.startswith(self.prefix + "/"):
            return full_key[len(self.prefix) + 1 :]
        return full_key

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._full(key), Body=data)

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._full(key))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NoSuchBucket"):
                raise FileNotFoundError(key) from e
            raise
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self._full(key))

    def list(self, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        out = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._full(prefix)):
            for obj in page.get("Contents", []):
                out.append(self._strip(obj["Key"]))
        return sorted(out)


def get_object_store(
    backend: str = "local",
    *,
    root: str | Path = "./storage",
    bucket: str | None = None,
    prefix: str = "",
    region: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> ObjectStore:
    """Construct an :class:`ObjectStore` for ``backend`` ('local' | 's3')."""
    if backend == "local":
        return LocalObjectStore(root)
    if backend == "s3":
        if not bucket:
            raise ValueError("S3 storage backend requires a bucket name.")
        return S3ObjectStore(
            bucket,
            prefix=prefix,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
        )
    raise ValueError(f"Unknown storage backend: {backend!r} (expected 'local'|'s3').")
