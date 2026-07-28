"""The frontend's copy of the inference-method enum must not drift (#221).

The frontend hand-copied this enum twice — a label list plus an
``EXACT_METHODS`` set in ``ModelSpecWidget.tsx``, and an inline three-branch
test in ``ArtifactsPanel.tsx`` — and both were written before the frequentist
path shipped. Both spelled ``is_approximate`` as ``method not in (nuts, smc)``,
so a ``frequentist_ridge`` spec rendered an amber "approximate" badge and the
instruction to "re-fit with NUTS", contradicting the shipped rule that
``approximate`` stays False for a frequentist fit (#188).

Nothing caught it, because nothing was comparing the copies. This is that
comparison. A hand-copied enum in the client is fine; a hand-copied enum with no
gate is how this shipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mmm_framework.config.enums import FitMethod, InferenceMethod
from mmm_framework.config.inference_methods import (
    BAYESIAN,
    FREQUENTIST,
    INFERENCE_METHODS,
    frequentist_method_values,
    method_info,
    method_values,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# the registry covers, and agrees with, the enums it unions
# ---------------------------------------------------------------------------


def test_registry_covers_every_fit_method_and_frequentist_member():
    """A new enum member must be described here, not silently unrenderable."""
    values = method_values()
    for m in FitMethod:
        assert m.value in values, f"{m} missing from INFERENCE_METHODS"
    for m in (InferenceMethod.FREQUENTIST_RIDGE, InferenceMethod.FREQUENTIST_CVXPY):
        assert m.value in values, f"{m} missing from INFERENCE_METHODS"
    assert len(INFERENCE_METHODS) == len(values), "duplicate value in the registry"


@pytest.mark.parametrize("method", list(FitMethod))
def test_approximate_flag_is_read_off_the_enum(method):
    info = method_info(method.value)
    assert info is not None
    assert info.approximate is method.is_approximate
    assert info.paradigm == BAYESIAN


def test_smc_is_exact():
    """The rule the frontend's `!(nuts|smc)` spelling got right, pinned."""
    assert method_info("smc").approximate is False
    assert method_info("nuts").approximate is False
    assert method_info("map").approximate is True


@pytest.mark.parametrize("value", sorted(frequentist_method_values()))
def test_frequentist_is_not_approximate(value):
    """The bug, stated as an assertion. A penalized point estimate with
    bootstrap CIs is not an approximation of a posterior."""
    info = method_info(value)
    assert info.paradigm == FREQUENTIST
    assert info.approximate is False
    assert info.interval_kind == "confidence"
    # and it carries its OWN caveat, not the re-fit-with-NUTS one
    assert "re-fit with NUTS" not in (info.caveat or "")
    assert "bootstrap" in (info.caveat or "").lower()


def test_paradigm_names_match_the_provenance_module():
    """`diagnostics.provenance` is the authority on the two family names; this
    registry restates them for a lower layer, so they have to be equal."""
    from mmm_framework.diagnostics import provenance

    assert BAYESIAN == provenance.BAYESIAN
    assert FREQUENTIST == provenance.FREQUENTIST


def test_unknown_method_returns_none_rather_than_guessing():
    """The guess is the bug: `!(nuts|smc)` called every unknown value
    approximate."""
    assert method_info("nuts_but_faster") is None
    assert method_info("") is None
    assert method_info(None) is None


def test_agent_spec_validation_derives_from_the_registry():
    from mmm_framework.agents import fitting

    assert fitting._INFERENCE_METHODS == method_values()
    assert fitting._FREQUENTIST_METHODS == frequentist_method_values()


# ---------------------------------------------------------------------------
# the generated frontend mirror
# ---------------------------------------------------------------------------


def test_generated_typescript_is_up_to_date():
    """Regenerate with: uv run python scripts/gen_fe_enums.py"""
    result = subprocess.run(
        [sys.executable, "scripts/gen_fe_enums.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_generated_file_carries_every_method_and_the_right_flags():
    """Read the emitted TS as text — if the generator itself regressed, the
    --check comparison above would still pass (it compares to its own output)."""
    ts = (REPO_ROOT / "frontend/src/api/generated/inferenceMethods.ts").read_text()
    for info in INFERENCE_METHODS:
        assert f'"value": "{info.value}"' in ts, f"{info.value} missing from the mirror"
    # The specific regression: the frequentist entries must be present and NOT
    # flagged approximate.
    ridge = ts.split('"value": "frequentist_ridge"', 1)[1].split("}", 1)[0]
    assert '"approximate": false' in ridge
    assert '"paradigm": "frequentist"' in ridge
    assert '"intervalKind": "confidence"' in ridge


def test_no_hand_rolled_approximate_test_survives_in_the_frontend():
    """A negative test: the two mirrors must be gone, not merely supplemented.

    Both spelled the rule as "not nuts and not smc". If either is reintroduced
    the badge silently goes wrong again for any method added later, and the
    gate above would not notice — it only checks the generated file.
    """
    fe = REPO_ROOT / "frontend/src"
    offenders = []
    for path in fe.rglob("*.tsx"):
        # Tests legitimately quote the old string to assert it is gone.
        if path.name.endswith((".test.tsx", ".spec.tsx")):
            continue
        text = path.read_text()
        if "EXACT_METHODS" in text:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: EXACT_METHODS")
        if "(approximate)" in text and "methodInfo" not in text:
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}: '(approximate)' without methodInfo"
            )
    assert not offenders, (
        "hand-rolled approximate test found; read it from "
        "api/generated/inferenceMethods instead:\n  " + "\n  ".join(offenders)
    )
