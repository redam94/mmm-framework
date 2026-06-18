"""Auth configuration (``MMM_AUTH_*`` env vars).

Separate from the API ``Settings`` so the same auth layer serves both backends
(root ``api/`` and the agent app) with one source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """Authentication settings, loaded from the environment.

    When ``enabled`` is false the dependency layer returns a single-tenant dev
    principal, so existing local/dev workflows keep working with no token.
    """

    model_config = SettingsConfigDict(
        env_prefix="MMM_AUTH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    provider: Literal["local", "oidc"] = "local"

    # Built-in (local) HS256 signing secret. Required when enabled + local.
    secret: str = ""
    issuer: str = "mmm-framework"
    audience: str = "mmm-api"

    access_ttl_seconds: int = 60 * 60  # 1 hour
    refresh_ttl_seconds: int = 60 * 60 * 24 * 14  # 14 days
    invite_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 days
    reset_ttl_seconds: int = 60 * 60  # 1 hour

    # External IdP (used only when provider == "oidc"; not yet implemented).
    oidc_jwks_url: str = ""
    oidc_issuer: str = ""
    oidc_audience: str = ""

    # Optional one-shot admin bootstrap (creates an org + owner on startup if the
    # email does not already exist). Handy for first-run / demos.
    bootstrap_org: str = ""
    bootstrap_email: str = ""
    bootstrap_password: str = ""

    # Dev principal used when auth is disabled (keeps single-tenant flows intact).
    dev_org_id: str = "dev-org"
    dev_user_id: str = "dev-user"
    dev_email: str = "dev@localhost"

    min_password_length: int = Field(default=10, ge=6)

    def require_secret(self) -> str:
        """Return the signing secret or raise if misconfigured."""
        if not self.secret:
            raise RuntimeError(
                "MMM_AUTH_SECRET is required when auth is enabled with the local "
                "provider. Generate one with: python -c "
                "'import secrets;print(secrets.token_urlsafe(48))'"
            )
        return self.secret


@lru_cache
def get_auth_settings() -> AuthSettings:
    """Cached auth settings instance."""
    return AuthSettings()
