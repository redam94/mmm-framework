"""Security & governance helpers (PII screening, encryption-at-rest).

Dependency-light building blocks for the hosted posture. Each is opt-in and
independently testable; nothing here is wired into a hot path by default.
"""

from __future__ import annotations

from .pii import PIIFinding, scan_dataframe_for_pii, scan_text

__all__ = ["PIIFinding", "scan_dataframe_for_pii", "scan_text"]

try:  # encryption needs `cryptography`, which is already a core dep (scrypt)
    from .encryption import (  # noqa: F401
        DatasetEncryptor,
        EncryptionError,
        encryption_enabled,
    )

    __all__ += ["DatasetEncryptor", "EncryptionError", "encryption_enabled"]
except Exception:  # pragma: no cover - cryptography always present in practice
    pass
