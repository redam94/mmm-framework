"""Authentication, organizations, and tenant access control.

Dependency-free foundation (HS256 JWT via stdlib, scrypt via ``cryptography``)
structured so an external IdP plugs into :class:`TokenVerifier` later. The
``deps``/``routes`` submodules require ``fastapi`` and are imported by the API
apps only — importing this package does **not** import fastapi.
"""

from __future__ import annotations

from .config import AuthSettings, get_auth_settings
from .models import (
    AuthContext,
    Role,
    role_rank,
    role_satisfies,
)
from .passwords import hash_password, verify_password
from .tokens import (
    AuthError,
    ExpiredToken,
    InvalidToken,
    LocalJWTVerifier,
    TokenVerifier,
    build_verifier,
    decode_jwt,
    encode_jwt,
)

__all__ = [
    "AuthSettings",
    "get_auth_settings",
    "AuthContext",
    "Role",
    "role_rank",
    "role_satisfies",
    "hash_password",
    "verify_password",
    "AuthError",
    "ExpiredToken",
    "InvalidToken",
    "LocalJWTVerifier",
    "TokenVerifier",
    "build_verifier",
    "decode_jwt",
    "encode_jwt",
]
