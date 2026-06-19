"""Google Cloud credential + project resolution, shared by GCS and BigQuery.

Reuses the project's established ADC-first posture (see ``agents/llm.py``):
a ``None`` credentials path means Application Default Credentials — the VM
service account, ``gcloud auth application-default login``, or the
``GOOGLE_APPLICATION_CREDENTIALS`` env var — so a deployment already running a
Vertex LLM needs **no** extra key management. An explicit path loads a
service-account JSON.
"""

from __future__ import annotations

import os
from typing import Any

from .base import IntegrationAuthError, require_dependency

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

#: env vars consulted (in order) when a project is not passed explicitly.
_PROJECT_ENV_VARS = (
    "MMM_GCP_PROJECT",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GCP_PROJECT",
)


def resolve_project(explicit: str | None = None) -> str | None:
    """First non-empty of the explicit value then the standard GCP env vars."""
    if explicit:
        return explicit
    for var in _PROJECT_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


def load_gcp_credentials(credentials_path: str | None = None) -> Any | None:
    """Return google credentials for a service-account JSON, or ``None`` for ADC.

    Mirrors ``agents/llm.py:_load_credentials`` so GCS/BigQuery and the LLM share
    one authentication story.
    """
    path = credentials_path or os.environ.get("MMM_GCP_CREDENTIALS_PATH")
    if not path:
        return None  # ADC: google.auth.default() resolves the ambient identity.
    if not os.path.exists(path):
        raise IntegrationAuthError(
            f"Service-account key not found at {path!r}. Leave the credentials "
            f"path empty to use Application Default Credentials instead."
        )
    sa = require_dependency(
        "google.oauth2.service_account", purpose="load a service-account key"
    )
    return sa.Credentials.from_service_account_file(path, scopes=_SCOPES)
