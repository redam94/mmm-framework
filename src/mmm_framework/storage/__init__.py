"""Pluggable object storage (local filesystem / S3) for artifacts."""

from __future__ import annotations

from .object_store import (
    LocalObjectStore,
    ObjectStore,
    S3ObjectStore,
    get_object_store,
)

__all__ = [
    "ObjectStore",
    "LocalObjectStore",
    "S3ObjectStore",
    "get_object_store",
]
