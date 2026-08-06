"""Lean-core packaging gate.

``pip install mmm-framework`` (no extras) must give a working modeling library:
no module in the core import surface may pull the web stack (fastapi/uvicorn)
or the LLM stack (langgraph/langchain-*) — those live in the separate
``mmm-framework-server`` workspace package and the ``[agents]`` extra
respectively (see the 2026-07-24 packaging split in pyproject.toml).

The dev environment installs everything, so this gate simulates the lean
environment inside a subprocess: a ``sys.meta_path`` blocker makes importing
any optional-stack distribution raise ImportError, then imports the core
surface. A regression (someone adding a module-level ``import fastapi`` or
``from langchain_core...`` to core code) fails here long before a user hits it
in a clean venv.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Distributions that are NOT part of the lean core install. Keep in sync with
# pyproject.toml: [agents] extra + server/pyproject.toml dependencies.
BLOCKED_PACKAGES = [
    # Optional [frequentist] extra: the constrained estimator imports it lazily,
    # so the design matrix and the closed-form ridge must work without it.
    "cvxpy",
    "fastapi",
    "starlette",
    "uvicorn",
    "slowapi",
    "redis",
    "httpx",
    "pypdf",
    "docx",
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "langchain_openai",
    "langchain_google_genai",
    "langchain_google_vertexai",
]

# The core import surface a lean install must serve. Submodules that core code
# imports lazily from mmm_framework.agents are included: they are the reason
# agents/__init__.py must stay lazy.
CORE_IMPORTS = [
    "mmm_framework",
    "mmm_framework.builders",
    "mmm_framework.model",
    "mmm_framework.transforms",
    "mmm_framework.reporting",
    "mmm_framework.reporting.prefit",
    "mmm_framework.reporting.interactive",
    "mmm_framework.validation",
    "mmm_framework.validation.spec_curve",
    "mmm_framework.planning",
    "mmm_framework.planning.history",
    "mmm_framework.estimands",
    "mmm_framework.finance",
    # The frequentist package's own invariant: cvxpy is a function-local
    # import (constrained.py), so this import must survive the cvxpy blocker.
    # Absent from this list, the BLOCKED_PACKAGES cvxpy entry guarded nothing
    # (#228).
    "mmm_framework.frequentist",
    "mmm_framework.diagnostics",
    "mmm_framework.calibration",
    "mmm_framework.mmm_extensions",
    "mmm_framework.synth",
    "mmm_framework.eda",
    "mmm_framework.data_studio.transforms",
    "mmm_framework.garden",
    "mmm_framework.excel_config",
    "mmm_framework.serialization",
    "mmm_framework.analysis",
    "mmm_framework.dag_model_builder",
    "mmm_framework.continuous_learning",
    "mmm_framework.ltv",
    # Platform services (ex-api service layer) are dependency-light by design.
    "mmm_framework.platform.sessions",
    "mmm_framework.platform.history",
    "mmm_framework.platform.runs",
    "mmm_framework.platform.pacing",
    # Realized-KPI actuals parsing/reconciliation (#227): numpy + stdlib.
    "mmm_framework.platform.actuals",
    # Variance-to-plan input assembly (#227): store reads + numpy, no web deps.
    "mmm_framework.platform.variance",
    # The commitment gate is stdlib-only by design — the server and the agent
    # both import it, and neither should pull the reporting stack to get it.
    "mmm_framework.platform.plan_of_record",
    "mmm_framework.platform.triangulation",
    # Shares the estimand grading rule with the reporting stack via
    # finance.evidence; it must not drag the reporting stack in to get it.
    "mmm_framework.platform.estimands",
    # Auth core is stdlib + cryptography (a declared core dep); fastapi lives
    # only in auth.deps/auth.routes, which the server imports.
    "mmm_framework.auth",
    # langchain-free agents service modules used by core code paths.
    "mmm_framework.agents.fitting",
    "mmm_framework.agents.workspace",
    "mmm_framework.agents.tables",
    "mmm_framework.agents.estimand_rows",
    "mmm_framework.agents.spec_locks",
    "mmm_framework.agents.spec_normalize",
    "mmm_framework.agents.model_ops",
    "mmm_framework.agents.report_builder",
    "mmm_framework.agents.branding",
]

_BLOCKER_TEMPLATE = r"""
import sys

BLOCKED = {blocked!r}


class _LeanGateBlocker:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            # What a genuinely absent distribution raises, so production
            # except-clauses (e.g. agents/__init__) behave identically.
            raise ModuleNotFoundError(
                f"lean-core gate: {{fullname!r}} is an optional dependency "
                "(server package or [agents] extra) and must not be imported "
                "by the core surface at module level",
                name=fullname,
            )
        return None


sys.meta_path.insert(0, _LeanGateBlocker())

failures = []
for mod in {imports!r}:
    for cached in [m for m in sys.modules if m == mod or m.startswith(mod + ".")]:
        del sys.modules[cached]
    try:
        __import__(mod)
    except ImportError as exc:
        failures.append(f"{{mod}}: {{exc}}")

if failures:
    print("LEAN-GATE FAILURES:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("lean gate ok")
"""


def test_core_surface_imports_without_web_or_llm_stack():
    """The whole core surface imports with fastapi/langchain/etc. blocked."""
    code = _BLOCKER_TEMPLATE.format(blocked=BLOCKED_PACKAGES, imports=CORE_IMPORTS)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "Core import surface leaked an optional dependency:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_agents_init_gives_install_hint_without_llm_stack():
    """Accessing an LLM-backed agents export without the extra names the fix."""
    code = _BLOCKER_TEMPLATE.format(blocked=BLOCKED_PACKAGES, imports=[]) + r"""
import mmm_framework.agents as agents

try:
    agents.create_agent_graph
except ModuleNotFoundError as exc:
    assert "mmm-framework[agents]" in str(exc), f"unhelpful error: {exc}"
    print("hint ok")
else:
    raise SystemExit("expected ModuleNotFoundError for create_agent_graph")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "hint ok" in result.stdout


@pytest.mark.parametrize("dead_dep", ["slowapi"])
def test_dead_dependencies_stay_out_of_pyproject(dead_dep):
    """slowapi was dropped as unused — keep it out of every dependency list."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("pyproject.toml", "server/pyproject.toml"):
        text = (root / rel).read_text()
        assert dead_dep not in text, f"{dead_dep} reappeared in {rel}"
