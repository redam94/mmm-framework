"""Password hashing — scrypt via ``cryptography`` (no extra dependency).

We deliberately avoid ``passlib``/``bcrypt``/``argon2-cffi`` to keep the auth
foundation dependency-free: ``cryptography`` is already a transitive dependency,
and its scrypt KDF is a sound memory-hard choice for password storage.

Encoded format (single self-describing string, like ``passlib``'s)::

    scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>

so verification reads its own parameters and stays forward-compatible if we
raise the work factor later.
"""

from __future__ import annotations

import base64
import hmac
import os

from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

# scrypt cost parameters. n must be a power of two; (n=2**15, r=8, p=1) targets
# ~tens of milliseconds per hash on commodity hardware in 2026 — strong enough
# for interactive login without making it a DoS vector.
_DEFAULT_N = 2**15
_DEFAULT_R = 8
_DEFAULT_P = 1
_KEY_LEN = 32
_SALT_LEN = 16


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    kdf = Scrypt(salt=salt, length=_KEY_LEN, n=n, r=r, p=p)
    return kdf.derive(password.encode("utf-8"))


def hash_password(
    password: str,
    *,
    n: int = _DEFAULT_N,
    r: int = _DEFAULT_R,
    p: int = _DEFAULT_P,
) -> str:
    """Hash ``password`` with a fresh random salt.

    Returns the self-describing ``scrypt$n$r$p$salt$hash`` encoding.
    """
    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(_SALT_LEN)
    dk = _derive(password, salt, n, r, p)
    return f"scrypt${n}${r}${p}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, encoded: str | None) -> bool:
    """Constant-time check of ``password`` against an ``encoded`` hash.

    Returns ``False`` (never raises) on any malformed/empty/None hash so callers
    can treat "no password set" and "wrong password" identically.
    """
    if not encoded or not password:
        return False
    try:
        scheme, n_s, r_s, p_s, salt_b64, hash_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
    except (ValueError, TypeError):
        return False
    try:
        candidate = _derive(password, salt, n, r, p)
    except Exception:
        return False
    return hmac.compare_digest(candidate, expected)
