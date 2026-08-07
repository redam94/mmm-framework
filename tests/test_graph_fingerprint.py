"""Graph-fingerprint contract over the model matrix (src-refactor PR 0.1).

UNMARKED (fast tier). Each matrix case is built unfitted, fingerprinted with
``numeric=True``, and compared as a FULL DICT against its golden JSON under
``tests/contracts/graph_fingerprints/`` — so a failure is a readable diff, not
an opaque hash mismatch. On mismatch the assertion message carries a unified
json diff plus ``pymc.printing.str_for_model(model)`` (``Model.str_repr`` does
NOT exist in pymc 6.0.1).

Why the numeric block is mandatory: ``michaelis_menten`` and ``tanh``
saturation COLLIDE on every structural key (same RVs, same priors — only the
Deterministic formula differs) and separate only numerically. A structure-only
contract would wave a swapped saturation family straight through; the test
proving that false negative is below.

Golden regeneration is gated behind ``MMM_REGEN_FINGERPRINTS=1``:

    MMM_REGEN_FINGERPRINTS=1 uv run pytest tests/test_graph_fingerprint.py -q
"""

from __future__ import annotations

import difflib
import functools
import json
import os
import sys
import warnings
from pathlib import Path

import pytest

_CONTRACTS = Path(__file__).resolve().parent / "contracts"
sys.path.insert(0, str(_CONTRACTS))

import model_matrix  # noqa: E402
from graph_fingerprint import model_fingerprint  # noqa: E402

GOLDEN_DIR = _CONTRACTS / "graph_fingerprints"
REGEN = os.environ.get("MMM_REGEN_FINGERPRINTS") == "1"

# Backend note (measured): the goldens are BACKEND-INDEPENDENT — recomputing
# the numerics-heavy cases (GP trend, multiplicative, per-geo betas, both
# extension models, weibull adstock, levers) under ``cxx=""`` (the pure-python
# linker, tests/test_model.py's trick) reproduced the C-compiled goldens
# byte-for-byte at the engine's 9-decimal rounding. The default backend is
# still used here because python-mode Scan execution is ~5-10x slower per
# case, which more than cancels the saved clang time.


def _dump(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _normalize(payload: dict) -> dict:
    """Round-trip through the golden serialization so the comparison sees
    exactly what the JSON file can store (tuples -> lists, -0.0 -> 0.0, ...)."""
    return json.loads(json.dumps(payload, sort_keys=True))


def _build_model(name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model_matrix.CASES[name]()


@functools.lru_cache(maxsize=None)
def _observed(name: str) -> str:
    """The case's golden-comparable payload, as canonical JSON text (cached —
    hashable, and several tests read the same case)."""
    if name in model_matrix.REFUSAL_CASES:
        try:
            _build_model(name)
        except Exception as exc:  # noqa: BLE001 — the refusal IS the contract
            payload = {
                "refusal": {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            }
        else:
            raise AssertionError(
                f"{name}: expected the builder to refuse, but a model was built"
            )
    else:
        payload = model_fingerprint(_build_model(name), numeric=True)
    return _dump(_normalize(payload))


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _maybe_regen(path: Path, text: str) -> None:
    if REGEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _failure_artifact(name: str) -> str:
    """str_for_model output for the mismatch message (spec: attach it; and
    ``hasattr(pm.Model, "str_repr")`` is False in pymc 6.0.1)."""
    if name in model_matrix.REFUSAL_CASES:
        return "(refusal case — no model graph to print)"
    try:
        import pymc.printing

        model = _build_model(name)
        return pymc.printing.str_for_model(getattr(model, "model", model))
    except Exception as exc:  # noqa: BLE001 — the artifact must never mask the diff
        return f"(str_for_model unavailable: {type(exc).__name__}: {exc})"


# ── the golden contract, one test per case ───────────────────────────────────


@pytest.mark.parametrize("name", sorted(model_matrix.CASES))
def test_case_matches_golden(name):
    observed_text = _observed(name)
    path = _golden_path(name)
    _maybe_regen(path, observed_text)
    assert path.exists(), (
        f"no golden for case {name!r} at {path} — run once with "
        "MMM_REGEN_FINGERPRINTS=1 to create it"
    )
    golden_text = path.read_text()
    observed = json.loads(observed_text)
    golden = json.loads(golden_text)
    if observed != golden:
        diff = "\n".join(
            difflib.unified_diff(
                golden_text.splitlines(),
                observed_text.splitlines(),
                fromfile=f"golden/{name}.json",
                tofile=f"observed/{name}",
                lineterm="",
            )
        )
        pytest.fail(
            f"fingerprint drift for case {name!r}:\n{diff}\n\n"
            f"--- model graph (pymc.printing.str_for_model) ---\n"
            f"{_failure_artifact(name)}\n\n"
            "If the change is INTENDED, regenerate the golden with "
            "MMM_REGEN_FINGERPRINTS=1 and commit the diff."
        )


def test_unbuildable_registry_matches_golden():
    """Absence must be explicit: the UNBUILDABLE dict is golden-pinned, so a
    capability silently dropping out of CASES cannot go unnoticed."""
    text = _dump(_normalize({"unbuildable": model_matrix.UNBUILDABLE}))
    path = GOLDEN_DIR / "_unbuildable.json"
    _maybe_regen(path, text)
    assert json.loads(path.read_text()) == json.loads(text)


# ── the reason the numeric block exists ──────────────────────────────────────

_STRUCTURAL_KEYS = (
    "free_RVs",
    "observed_RVs",
    "deterministics",
    "potentials",
    "data_vars",
    "coords",
    "dims",
)


def test_michaelis_menten_and_tanh_collide_structurally_but_not_numerically():
    """The measured false negative that makes ``numeric=True`` mandatory: both
    families emit ``sat_half_<ch>`` with the same Beta prior, and only the
    Deterministic formula differs — every structural key is IDENTICAL, and the
    two separate only in the numeric block."""
    mm = json.loads(_observed("saturation_michaelis_menten"))
    th = json.loads(_observed("saturation_tanh"))
    for key in _STRUCTURAL_KEYS:
        assert mm[key] == th[key], (
            f"structural key {key!r} unexpectedly differs — the false-negative "
            "premise changed; re-verify whether the numeric block is still the "
            "only separator"
        )
    assert mm["deterministic_values"] != th["deterministic_values"], (
        "michaelis_menten and tanh produced identical Deterministic values at "
        "the probe point — the numeric block no longer separates them and the "
        "contract has a false negative"
    )
    assert mm != th


# ── determinism ──────────────────────────────────────────────────────────────


def test_fingerprint_is_deterministic_across_rebuilds():
    """One case, built twice from scratch, must fingerprint identically —
    the property every golden depends on (name-derived probe offsets,
    ``replace_rvs_by_values``, seed-invariant ``initial_point``)."""
    fp_a = _normalize(model_fingerprint(_build_model("default_national")))
    fp_b = _normalize(model_fingerprint(_build_model("default_national")))
    assert fp_a == fp_b


# ── the refusal contract ─────────────────────────────────────────────────────


def test_binomial_refused_raises_the_contracted_refusal():
    """The refusal IS the contract: the binomial-family config must refuse at
    graph build with the exact exception type + message the matrix declares."""
    exc_name, fragment = model_matrix.REFUSAL_CASES["binomial_refused"]
    assert exc_name == "NotImplementedError"
    with pytest.raises(NotImplementedError, match=fragment):
        _build_model("binomial_refused")


def test_refusal_cases_are_registered_cases():
    missing = set(model_matrix.REFUSAL_CASES) - set(model_matrix.CASES)
    assert not missing, f"REFUSAL_CASES not in CASES: {sorted(missing)}"
