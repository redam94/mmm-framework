"""Deployment profile (Phase 3 PR-F.6).

A single switch, ``MMM_AGENT_HOSTED``, flips the agent from the single-user dev
posture to the **hosted multi-user** posture. It is deliberately *inert* until the
Tier-2 container sandbox exists (it would be more dangerous half-applied — §4), so
turning it on assumes PR-F.1..F.5 are in place. When hosted:

  - the kernel defaults to the ``container`` impl and **requires** a complete
    sandbox (``ContainerKernel._verify_isolation`` fail-closes),
  - egress is denied (the container default),
  - ``Path.cwd()`` is dropped from the download allow-roots (``workspace.allowed_roots``),
  - agent reports are written per-session under the workspace (not the shared CWD),
  - the API refuses guessable / client-invented ``thread_id``s — only server-minted
    uuid4 sessions (``POST /sessions``) are accepted.
"""

from __future__ import annotations

import os

_FALSE = ("0", "false", "no", "")


def is_hosted() -> bool:
    """True in the hosted multi-user posture (``MMM_AGENT_HOSTED``)."""
    return os.environ.get("MMM_AGENT_HOSTED", "0").strip().lower() not in _FALSE


def default_kernel_impl() -> str:
    """The kernel implementation to use when ``MMM_AGENT_KERNEL`` is unset: a
    sandboxed ``container`` when hosted, else the in-process warm kernel."""
    explicit = os.environ.get("MMM_AGENT_KERNEL")
    if explicit:
        return explicit
    return "container" if is_hosted() else "inprocess"
