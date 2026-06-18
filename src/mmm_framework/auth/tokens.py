"""JWT issuance + verification, with a pluggable verifier.

Built-in tokens are **HS256** signed with a shared secret, implemented on the
stdlib (``hmac``/``hashlib``/``base64``) so the foundation needs no JWT library.
Everything that consumes a token goes through the :class:`TokenVerifier`
protocol, so swapping the built-in signer for an external IdP (OIDC/JWKS,
RS256) later is a single ``build_verifier`` change — call sites don't move.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Protocol


class AuthError(Exception):
    """Base class for token problems (auth failures)."""


class InvalidToken(AuthError):
    """Token is malformed, mis-signed, or fails a claim check."""


class ExpiredToken(InvalidToken):
    """Token signature is valid but it is past ``exp``."""


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def encode_jwt(payload: dict[str, Any], secret: str) -> str:
    """Sign ``payload`` as a compact HS256 JWT."""
    header = {"alg": "HS256", "typ": "JWT"}
    segs = [
        _b64url(json.dumps(header, separators=(",", ":")).encode()),
        _b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segs).encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    segs.append(_b64url(sig))
    return ".".join(segs)


def decode_jwt(
    token: str,
    secret: str,
    *,
    verify_exp: bool = True,
    audience: str | None = None,
    issuer: str | None = None,
    leeway: float = 0.0,
) -> dict[str, Any]:
    """Verify an HS256 JWT signature and standard claims, return the payload.

    Raises :class:`ExpiredToken` / :class:`InvalidToken` on any failure.
    """
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise InvalidToken("malformed token") from exc

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = _b64url(
        hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    )
    if not hmac.compare_digest(expected, sig_b64):
        raise InvalidToken("bad signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidToken("undecodable payload") from exc

    now = time.time()
    if verify_exp and "exp" in payload and now > float(payload["exp"]) + leeway:
        raise ExpiredToken("token expired")
    if "nbf" in payload and now + leeway < float(payload["nbf"]):
        raise InvalidToken("token not yet valid")
    if audience is not None and payload.get("aud") != audience:
        raise InvalidToken("audience mismatch")
    if issuer is not None and payload.get("iss") != issuer:
        raise InvalidToken("issuer mismatch")
    return payload


def make_claims(
    *,
    subject: str,
    org_id: str,
    org_role: str,
    email: str,
    token_type: str,
    ttl_seconds: float,
    issuer: str,
    audience: str,
    token_version: int = 0,
) -> dict[str, Any]:
    """Assemble a standard claim set for an access/refresh token.

    ``token_version`` (claim ``tv``) is the user's current token generation.
    Bumping the stored version (on deactivate / password reset / "sign out
    everywhere") invalidates every token minted at an older version on its next
    use — giving immediate, stateless revocation of even live access tokens.
    """
    now = time.time()
    return {
        "sub": subject,
        "org": org_id,
        "role": org_role,
        "email": email,
        "typ": token_type,  # "access" | "refresh"
        "tv": token_version,
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex,
    }


class TokenVerifier(Protocol):
    """Anything that can turn a bearer token string into verified claims."""

    def verify(self, token: str) -> dict[str, Any]:
        """Return verified claims or raise :class:`AuthError`."""
        ...


class LocalJWTVerifier:
    """Verifies HS256 tokens this service signed itself."""

    def __init__(self, secret: str, *, issuer: str, audience: str) -> None:
        self._secret = secret
        self._issuer = issuer
        self._audience = audience

    def verify(self, token: str) -> dict[str, Any]:
        return decode_jwt(
            token,
            self._secret,
            audience=self._audience,
            issuer=self._issuer,
        )


class OIDCVerifier:
    """Placeholder for external-IdP verification (OIDC/JWKS, RS256).

    Wiring this up later means: fetch+cache the IdP's JWKS, select the key by
    ``kid``, verify an RS256 signature, then map the IdP claims onto our
    ``sub``/``org``/``role`` shape. Intentionally not implemented in the
    dependency-free foundation; ``build_verifier`` raises a clear error if a
    deployment selects ``provider=oidc`` before this lands.
    """

    def __init__(self, **_: Any) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "OIDC verification is not yet implemented. Use MMM_AUTH_PROVIDER=local, "
            "or implement OIDCVerifier (JWKS fetch + RS256) and add the dependency."
        )

    def verify(self, token: str) -> dict[str, Any]:  # pragma: no cover - stub
        raise NotImplementedError


def build_verifier(settings: Any) -> TokenVerifier:
    """Construct the verifier for the configured provider.

    ``settings`` is an :class:`~mmm_framework.auth.config.AuthSettings`.
    """
    provider = getattr(settings, "provider", "local")
    if provider == "local":
        return LocalJWTVerifier(
            settings.require_secret(),
            issuer=settings.issuer,
            audience=settings.audience,
        )
    if provider == "oidc":
        return OIDCVerifier(
            jwks_url=settings.oidc_jwks_url,
            issuer=settings.oidc_issuer or settings.issuer,
            audience=settings.oidc_audience or settings.audience,
        )
    raise ValueError(f"unknown auth provider: {provider!r}")
