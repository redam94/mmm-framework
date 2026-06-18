"""High-level auth operations: signup, authenticate, token issuance, bootstrap."""

from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path
from typing import Any

from . import store
from .audit import audit_event
from .config import AuthSettings, get_auth_settings
from .models import AuthContext, Role, TokenResponse, normalize_email
from .passwords import hash_password, verify_password
from .tokens import make_claims, encode_jwt, decode_jwt, InvalidToken

logger = logging.getLogger("mmm_auth")


def _now() -> float:
    return time.time()


class AuthServiceError(Exception):
    """User-facing auth failure (bad credentials, duplicate email, weak password)."""


def signup_organization(
    *,
    organization: str,
    email: str,
    password: str,
    name: str | None = None,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], TokenResponse]:
    """Create an org + its first user (owner) and return (user, tokens)."""
    settings = settings or get_auth_settings()
    email = normalize_email(email)
    if len(password) < settings.min_password_length:
        raise AuthServiceError(
            f"password must be at least {settings.min_password_length} characters"
        )
    if store.get_user_by_email(email, db_path=db_path):
        raise AuthServiceError("an account with that email already exists")

    org = store.create_organization(organization, db_path=db_path)
    user = store.create_user(
        email=email,
        password_hash=hash_password(password),
        org_id=org["org_id"],
        name=name,
        role=Role.OWNER.value,
        db_path=db_path,
    )
    store.add_org_member(
        org["org_id"], user["user_id"], Role.OWNER.value, db_path=db_path
    )
    user_full = {**user, "org_role": Role.OWNER.value, "name": name or email}
    tokens = issue_tokens(
        user_id=user["user_id"],
        org_id=org["org_id"],
        email=email,
        org_role=Role.OWNER.value,
        settings=settings,
    )
    return user_full, tokens


def authenticate(
    email: str,
    password: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Verify credentials; return the user row or raise ``AuthServiceError``."""
    email = normalize_email(email)
    user = store.get_user_by_email(email, db_path=db_path)
    # Verify even on missing user (constant-ish work) to blunt user enumeration.
    ok = verify_password(password, user.get("password_hash") if user else None)
    if not user or not ok:
        raise AuthServiceError("invalid email or password")
    if user.get("status", "active") != "active":
        raise AuthServiceError("account is disabled")
    store.touch_last_login(user["user_id"], db_path=db_path)
    audit_event("auth.login", user_id=user["user_id"], org_id=user.get("org_id"))
    return user


def issue_tokens(
    *,
    user_id: str,
    org_id: str,
    email: str,
    org_role: str,
    token_version: int = 0,
    settings: AuthSettings | None = None,
) -> TokenResponse:
    """Mint an access + refresh token pair stamped with ``token_version``."""
    settings = settings or get_auth_settings()
    secret = settings.require_secret()
    access = encode_jwt(
        make_claims(
            subject=user_id,
            org_id=org_id,
            org_role=org_role,
            email=email,
            token_type="access",
            ttl_seconds=settings.access_ttl_seconds,
            issuer=settings.issuer,
            audience=settings.audience,
            token_version=token_version,
        ),
        secret,
    )
    refresh = encode_jwt(
        make_claims(
            subject=user_id,
            org_id=org_id,
            org_role=org_role,
            email=email,
            token_type="refresh",
            ttl_seconds=settings.refresh_ttl_seconds,
            issuer=settings.issuer,
            audience=settings.audience,
            token_version=token_version,
        ),
        secret,
    )
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_ttl_seconds,
    )


def refresh_tokens(
    refresh_token: str,
    *,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> TokenResponse:
    """Exchange a valid refresh token for a fresh access + refresh pair."""
    settings = settings or get_auth_settings()
    try:
        claims = decode_jwt(
            refresh_token,
            settings.require_secret(),
            audience=settings.audience,
            issuer=settings.issuer,
        )
    except InvalidToken as exc:
        raise AuthServiceError("invalid or expired refresh token") from exc
    if claims.get("typ") != "refresh":
        raise AuthServiceError("not a refresh token")
    if store.is_token_revoked(claims.get("jti", ""), db_path=db_path):
        raise AuthServiceError("refresh token has been revoked")
    user = store.get_user(claims["sub"], db_path=db_path)
    if not user or user.get("status", "active") != "active":
        raise AuthServiceError("account no longer active")
    current_tv = int(user.get("token_version") or 0)
    if int(claims.get("tv", 0)) != current_tv:
        # Token predates a deactivate / password-reset / sign-out-everywhere bump.
        raise AuthServiceError("token superseded — please sign in again")
    role = (
        store.get_org_role(claims["org"], claims["sub"], db_path=db_path)
        or claims["role"]
    )
    # Rotation: a refresh token is single-use. Revoke the presented one so a
    # leaked/replayed refresh token can't be used again, then mint a fresh pair.
    store.revoke_token(
        claims.get("jti", ""),
        claims.get("exp", _now()),
        user_id=claims["sub"],
        db_path=db_path,
    )
    audit_event(
        "auth.token_refreshed",
        user_id=claims["sub"],
        org_id=claims["org"],
        jti=claims.get("jti"),
    )
    return issue_tokens(
        user_id=claims["sub"],
        org_id=claims["org"],
        email=claims.get("email", user.get("email", "")),
        org_role=role,
        token_version=current_tv,
        settings=settings,
    )


def logout(
    refresh_token: str,
    *,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Revoke a refresh token (idempotent; never raises on a bad token)."""
    settings = settings or get_auth_settings()
    try:
        claims = decode_jwt(
            refresh_token,
            settings.require_secret(),
            verify_exp=False,  # allow revoking an already-expired token cleanly
            audience=settings.audience,
            issuer=settings.issuer,
        )
    except InvalidToken:
        return
    store.revoke_token(
        claims.get("jti", ""),
        claims.get("exp", _now()),
        user_id=claims.get("sub"),
        db_path=db_path,
    )
    audit_event("auth.logout", user_id=claims.get("sub"), org_id=claims.get("org"))


def create_invite(
    *,
    org_id: str,
    email: str,
    role: str,
    invited_by: str | None,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Create an org invite; returns the invite (incl. opaque token to deliver)."""
    settings = settings or get_auth_settings()
    email = normalize_email(email)
    if store.get_user_by_email(email, db_path=db_path):
        raise AuthServiceError("a user with that email already exists")
    token = secrets.token_urlsafe(32)
    expires_at = _now() + settings.invite_ttl_seconds
    store.create_invite(
        token=token,
        org_id=org_id,
        email=email,
        role=role,
        invited_by=invited_by,
        expires_at=expires_at,
        db_path=db_path,
    )
    audit_event(
        "auth.invite_created", user_id=invited_by, org_id=org_id, email=email, role=role
    )
    return {
        "token": token,
        "email": email,
        "role": role,
        "org_id": org_id,
        "expires_at": expires_at,
    }


def accept_invite(
    *,
    token: str,
    password: str,
    name: str | None = None,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any], TokenResponse]:
    """Redeem an invite: create the user in the org, return (user, tokens)."""
    settings = settings or get_auth_settings()
    inv = store.get_invite(token, db_path=db_path)
    if inv is None or inv.get("accepted_at"):
        raise AuthServiceError("invalid or already-used invite")
    if inv["expires_at"] < _now():
        raise AuthServiceError("invite has expired")
    if len(password) < settings.min_password_length:
        raise AuthServiceError(
            f"password must be at least {settings.min_password_length} characters"
        )
    if store.get_user_by_email(inv["email"], db_path=db_path):
        raise AuthServiceError("a user with that email already exists")
    user = store.create_user(
        email=inv["email"],
        password_hash=hash_password(password),
        org_id=inv["org_id"],
        name=name,
        role=inv["role"],
        db_path=db_path,
    )
    store.add_org_member(inv["org_id"], user["user_id"], inv["role"], db_path=db_path)
    store.mark_invite_accepted(token, db_path=db_path)
    audit_event(
        "auth.invite_accepted",
        user_id=user["user_id"],
        org_id=inv["org_id"],
        email=inv["email"],
    )
    tokens = issue_tokens(
        user_id=user["user_id"],
        org_id=inv["org_id"],
        email=inv["email"],
        org_role=inv["role"],
        settings=settings,
    )
    return {**user, "org_role": inv["role"]}, tokens


def request_password_reset(
    email: str,
    *,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> str | None:
    """Create a reset token for ``email`` if it exists. Returns the token (to be
    delivered out-of-band) or ``None`` — callers must NOT leak which, to avoid
    user enumeration."""
    settings = settings or get_auth_settings()
    email = normalize_email(email)
    user = store.get_user_by_email(email, db_path=db_path)
    if not user:
        return None
    token = secrets.token_urlsafe(32)
    store.create_password_reset(
        token=token,
        user_id=user["user_id"],
        expires_at=_now() + settings.reset_ttl_seconds,
        db_path=db_path,
    )
    audit_event(
        "auth.password_reset_requested",
        user_id=user["user_id"],
        org_id=user.get("org_id"),
    )
    return token


def confirm_password_reset(
    *,
    token: str,
    new_password: str,
    settings: AuthSettings | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Set a new password from a valid reset token, then revoke nothing else but
    mark the token used (single-use)."""
    settings = settings or get_auth_settings()
    rec = store.get_password_reset(token, db_path=db_path)
    if rec is None or rec.get("used_at"):
        raise AuthServiceError("invalid or already-used reset token")
    if rec["expires_at"] < _now():
        raise AuthServiceError("reset token has expired")
    if len(new_password) < settings.min_password_length:
        raise AuthServiceError(
            f"password must be at least {settings.min_password_length} characters"
        )
    store.set_password_hash(
        rec["user_id"], hash_password(new_password), db_path=db_path
    )
    store.mark_reset_used(token, db_path=db_path)
    # A password change kills every outstanding session for that user.
    store.bump_token_version(rec["user_id"], db_path=db_path)
    audit_event("auth.password_reset", user_id=rec["user_id"])


def deactivate_user(
    user_id: str,
    *,
    actor: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Disable an account. Bumping the token version kills its live access tokens
    immediately (not just the next refresh)."""
    store.set_user_status(user_id, "disabled", db_path=db_path)
    store.bump_token_version(user_id, db_path=db_path)
    audit_event("auth.user_deactivated", user_id=user_id, actor=actor)


def revoke_all_sessions(user_id: str, *, db_path: Path | str | None = None) -> None:
    """ "Sign out everywhere": invalidate all of a user's tokens immediately."""
    store.bump_token_version(user_id, db_path=db_path)
    audit_event("auth.sessions_revoked", user_id=user_id)


def initialize_auth(
    settings: AuthSettings | None = None, db_path: Path | str | None = None
) -> str:
    """Idempotent startup hook: schema + optional bootstrap + orphan backfill.

    Safe to call on every boot, with auth enabled or not. Steps:
      1. ensure the org/auth schema exists;
      2. if bootstrap env vars are set and the user is absent, create that owner
         in a new org (no token issued — this runs without a live request);
      3. attach every pre-existing org-less project/user to the primary org, so
         turning enforcement on later doesn't orphan existing data.

    Returns the primary org id (the bootstrap org, or the canonical default).
    """
    settings = settings or get_auth_settings()
    store.init_auth_schema(db_path)

    primary: str | None = None
    if settings.bootstrap_email and settings.bootstrap_password:
        email = normalize_email(settings.bootstrap_email)
        existing = store.get_user_by_email(email, db_path=db_path)
        if existing:
            primary = existing.get("org_id")
        elif len(settings.bootstrap_password) >= settings.min_password_length:
            org = store.create_organization(
                settings.bootstrap_org or "Default Organization", db_path=db_path
            )
            user = store.create_user(
                email=email,
                password_hash=hash_password(settings.bootstrap_password),
                org_id=org["org_id"],
                role=Role.OWNER.value,
                db_path=db_path,
            )
            store.add_org_member(
                org["org_id"], user["user_id"], Role.OWNER.value, db_path=db_path
            )
            primary = org["org_id"]
            logger.info("auth: bootstrapped owner %s in org %s", email, primary)
        else:
            logger.warning(
                "auth: MMM_AUTH_BOOTSTRAP_PASSWORD shorter than %d chars; skipping",
                settings.min_password_length,
            )

    if not primary:
        primary = store.ensure_default_organization(db_path=db_path)

    counts = store.attach_orphans_to_org(primary, db_path=db_path)
    if counts["projects"] or counts["users"]:
        logger.info(
            "auth: backfilled %d project(s) and %d user(s) into org %s",
            counts["projects"],
            counts["users"],
            primary,
        )
    return primary
