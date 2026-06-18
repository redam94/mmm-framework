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

import logging
import os

_FALSE = ("0", "false", "no", "")

# The only kernel impls that actually sandbox untrusted cell code. `inprocess`
# (API process) and `subprocess` (local ipykernel, no container) are NOT sandboxed.
SANDBOXED_IMPLS = frozenset({"container"})


def is_hosted() -> bool:
    """True in the hosted multi-user posture (``MMM_AGENT_HOSTED``)."""
    return os.environ.get("MMM_AGENT_HOSTED", "0").strip().lower() not in _FALSE


def default_kernel_impl() -> str:
    """The kernel implementation to use when ``MMM_AGENT_KERNEL`` is unset: a
    sandboxed ``container`` when hosted, else the in-process warm kernel.

    Hosted REQUIRES a sandbox, so a non-sandboxed explicit ``MMM_AGENT_KERNEL`` is
    a misconfiguration that is **force-upgraded** to ``container`` (fail-safe) —
    otherwise ``MMM_AGENT_HOSTED=1 MMM_AGENT_KERNEL=inprocess`` would run untrusted
    code unsandboxed while *behaving* hosted (the §4 partial-enablement trap). The
    sandbox itself fail-closes at spawn if the image/runtime is missing."""
    explicit = os.environ.get("MMM_AGENT_KERNEL")
    if is_hosted():
        if explicit and explicit not in SANDBOXED_IMPLS:
            logging.getLogger("mmm_audit").warning(
                "hosted: ignoring MMM_AGENT_KERNEL=%s and forcing a sandboxed "
                "container kernel (hosted requires a sandbox)",
                explicit,
            )
        return "container"
    return explicit or "inprocess"


def assert_hosted_sandbox(impl: str) -> None:
    """Fail-closed profile guard (PR-F.6, hardened): refuse to run hosted on a
    non-sandboxed kernel. Raises ``RuntimeError`` — wire into app startup so a
    misconfigured hosted deployment refuses to serve rather than silently running
    untrusted code in-process (§4: a half-applied sandbox is worse than none)."""
    if is_hosted() and impl not in SANDBOXED_IMPLS:
        raise RuntimeError(
            f"hosted profile requires a sandboxed kernel (container), got {impl!r} "
            "— refusing to start (set MMM_AGENT_HOSTED=0 for single-user, or build "
            "the kernel image and use the container kernel)"
        )
