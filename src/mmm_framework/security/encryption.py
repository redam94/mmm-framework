"""Opt-in encryption-at-rest for stored client data.

Client datasets are written to disk as plaintext today. This provides authenticated
symmetric encryption (Fernet / AES-128-CBC + HMAC) keyed from an environment
secret, so a hosted deployment can encrypt artifacts at rest with tamper
detection. Opt-in: inert unless ``MMM_ENCRYPTION_KEY`` is set.

Generate a key once and store it in your secret manager::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

# Files we encrypt start with this marker so decrypt() is a no-op on plaintext
# written before encryption was enabled (smooth migration).
_MAGIC = b"MMMENC1:"


class EncryptionError(Exception):
    """Raised when encryption/decryption fails (bad key or tampered ciphertext)."""


def encryption_enabled() -> bool:
    """True if an encryption key is configured (``MMM_ENCRYPTION_KEY``)."""
    return bool(os.environ.get("MMM_ENCRYPTION_KEY", "").strip())


class DatasetEncryptor:
    """Encrypt/decrypt bytes with Fernet.

    Supports key rotation: ``MMM_ENCRYPTION_KEY`` may be a comma-separated list
    (the FIRST key encrypts; ALL are tried for decrypt), so a new key can be
    rolled in while old ciphertext still decrypts.
    """

    def __init__(self, key: str | bytes | None = None):
        raw = key if key is not None else os.environ.get("MMM_ENCRYPTION_KEY", "")
        if isinstance(raw, bytes):
            raw = raw.decode()
        keys = [k.strip() for k in str(raw).split(",") if k.strip()]
        if not keys:
            raise EncryptionError(
                "No encryption key (set MMM_ENCRYPTION_KEY to a Fernet key)."
            )
        try:
            self._fernet = MultiFernet([Fernet(k.encode()) for k in keys])
        except Exception as e:  # noqa: BLE001
            raise EncryptionError(f"Invalid encryption key: {e}") from e

    def encrypt(self, data: bytes) -> bytes:
        if not isinstance(data, (bytes, bytearray)):
            raise EncryptionError("encrypt() requires bytes")
        return _MAGIC + self._fernet.encrypt(bytes(data))

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypt, tolerating plaintext written before encryption was enabled."""
        if not isinstance(blob, (bytes, bytearray)):
            raise EncryptionError("decrypt() requires bytes")
        blob = bytes(blob)
        if not blob.startswith(_MAGIC):
            return blob  # legacy plaintext — pass through unchanged
        try:
            return self._fernet.decrypt(blob[len(_MAGIC) :])
        except InvalidToken as e:
            raise EncryptionError(
                "Decryption failed: wrong key or tampered ciphertext."
            ) from e

    @staticmethod
    def is_encrypted(blob: bytes) -> bool:
        return isinstance(blob, (bytes, bytearray)) and bytes(blob).startswith(_MAGIC)
