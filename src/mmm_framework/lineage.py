"""Content-addressed dataset lineage.

So a fitted model resolves to the EXACT training bytes even when the source file
is later overwritten under the same name (the analyst's "reprove last quarter's
number" problem). The key IS the sha256 of the content, which gives automatic
**dedup** — re-snapshotting identical data stores nothing new and two models
trained on the same bytes share one snapshot.

Backed by the pluggable object store (:mod:`mmm_framework.storage`), so snapshots
can live on local disk or shared S3 in a multi-replica deployment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .storage import LocalObjectStore, ObjectStore


def content_hash(data: bytes) -> str:
    """sha256 hex digest of ``data`` — the content address."""
    return hashlib.sha256(data).hexdigest()


class DatasetLineage:
    """Store/resolve dataset bytes by content hash (immutable, deduplicated)."""

    def __init__(self, store: ObjectStore | None = None, *, root: str | Path = "./lineage"):
        self.store = store if store is not None else LocalObjectStore(root)

    @staticmethod
    def _key(h: str) -> str:
        # Shard by the first two hex chars so a directory/bucket doesn't accumulate
        # a single flat namespace of millions of keys.
        return f"{h[:2]}/{h}"

    def snapshot(self, data: bytes | str | Path) -> str:
        """Store ``data`` (bytes, or a path to read) content-addressed; return its
        hash. Idempotent: identical content is stored at most once (dedup)."""
        if isinstance(data, (str, Path)):
            data = Path(data).read_bytes()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("snapshot() takes bytes or a path")
        data = bytes(data)
        h = content_hash(data)
        key = self._key(h)
        if not self.store.exists(key):  # dedup
            self.store.put(key, data)
        return h

    def resolve(self, h: str) -> bytes:
        """Return the exact bytes for content hash ``h`` (``FileNotFoundError`` if
        the snapshot is absent)."""
        return self.store.get(self._key(h))

    def exists(self, h: str) -> bool:
        return self.store.exists(self._key(h))

    def verify(self, h: str) -> bool:
        """True if the stored bytes for ``h`` still hash to ``h`` (tamper check)."""
        try:
            return content_hash(self.resolve(h)) == h
        except FileNotFoundError:
            return False
