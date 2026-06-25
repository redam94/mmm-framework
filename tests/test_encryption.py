"""Encryption-at-rest (Phase 3 / G2)."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from mmm_framework.security.encryption import (
    DatasetEncryptor,
    EncryptionError,
    encryption_enabled,
)


def _key() -> str:
    return Fernet.generate_key().decode()


def test_round_trip():
    enc = DatasetEncryptor(_key())
    data = b"sensitive,bytes\n1,2\n"
    blob = enc.encrypt(data)
    assert DatasetEncryptor.is_encrypted(blob)
    assert enc.decrypt(blob) == data


def test_legacy_plaintext_passthrough():
    enc = DatasetEncryptor(_key())
    assert enc.decrypt(b"plaintext written before encryption") == (
        b"plaintext written before encryption"
    )


def test_tamper_is_detected():
    enc = DatasetEncryptor(_key())
    blob = bytearray(enc.encrypt(b"hello"))
    blob[-1] ^= 0x01
    with pytest.raises(EncryptionError):
        enc.decrypt(bytes(blob))


def test_wrong_key_fails():
    blob = DatasetEncryptor(_key()).encrypt(b"x")
    with pytest.raises(EncryptionError):
        DatasetEncryptor(_key()).decrypt(blob)


def test_key_rotation_decrypts_old_ciphertext():
    k1, k2 = _key(), _key()
    blob = DatasetEncryptor(k1).encrypt(b"data")
    # New config lists the new key first but keeps the old for decrypt.
    rotated = DatasetEncryptor(f"{k2},{k1}")
    assert rotated.decrypt(blob) == b"data"


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("MMM_ENCRYPTION_KEY", raising=False)
    assert encryption_enabled() is False
    monkeypatch.setenv("MMM_ENCRYPTION_KEY", _key())
    assert encryption_enabled() is True


def test_no_key_raises(monkeypatch):
    monkeypatch.delenv("MMM_ENCRYPTION_KEY", raising=False)
    with pytest.raises(EncryptionError):
        DatasetEncryptor()
