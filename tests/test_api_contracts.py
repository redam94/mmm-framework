"""v1.0 contract gates.

From v1.0.0 the project follows strict SemVer: the REST route set, the core
Python surface, the model-spec keys, and the config enum values are FROZEN
public contracts (docs/api-contracts.html). These tests pin them so a breaking
change fails CI instead of shipping silently.

Updating a snapshot here is a *deliberate act*: additions are minor-version
territory; removals/renames are major-version territory and must come with a
changelog entry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CONTRACTS_DIR = Path(__file__).parent / "contracts"


# ── REST API ──────────────────────────────────────────────────────────────────


def _live_routes() -> set[tuple[str, str]]:
    from mmm_framework_server.main import app

    return {
        (method.upper(), path)
        for path, ops in app.openapi()["paths"].items()
        for method in ops
    }


def test_rest_route_set_matches_snapshot():
    """Every (method, path) pair in the server matches tests/contracts/rest_routes.json.

    Removals or renames are BREAKING (major version). Additions are fine but
    must be recorded: regenerate the snapshot and mention the new endpoints in
    the changelog.
    """
    snapshot = {
        (r["method"], r["path"])
        for r in json.loads((CONTRACTS_DIR / "rest_routes.json").read_text())
    }
    live = _live_routes()

    removed = snapshot - live
    added = live - snapshot
    msg = []
    if removed:
        msg.append(
            "REMOVED endpoints (breaking — major version + changelog required):\n  "
            + "\n  ".join(f"{m} {p}" for m, p in sorted(removed))
        )
    if added:
        msg.append(
            "NEW endpoints not in the snapshot (update tests/contracts/"
            "rest_routes.json deliberately + changelog):\n  "
            + "\n  ".join(f"{m} {p}" for m, p in sorted(added))
        )
    assert not msg, "\n\n".join(msg)


# ── Core Python surface ───────────────────────────────────────────────────────

#: Symbols the v1.0 docs promise. Import failure or absence = breaking change.
PUBLIC_SURFACE = {
    "mmm_framework": [
        "BayesianMMM",
        "MFFLoader",
        "Dataset",
        "load_dataset",
        "load_example",
        "__version__",
    ],
    "mmm_framework.model": ["BayesianMMM"],
    "mmm_framework.model.results": ["MMMResults", "PredictionResults"],
    "mmm_framework.builders": [
        "ModelConfigBuilder",
        "MediaChannelConfigBuilder",
        "ControlVariableConfigBuilder",
        "KPIConfigBuilder",
    ],
    "mmm_framework.data_loader": ["MFFLoader", "PanelDataset", "PanelCoordinates"],
    "mmm_framework.serialization": ["MMMSerializer"],
    "mmm_framework.analysis": [
        "MMMAnalyzer",
        "MarginalAnalysisResult",
        "compute_contribution_summary",
    ],
    "mmm_framework.estimands": ["Estimand"],
    # v1.3.1 documented these as "exported from mmm_framework.validation" and
    # three of the five were not — nothing pinned the set, and the docs-snippet
    # gate only reads code fences, so a prose claim about the API surface had
    # nothing checking it.
    "mmm_framework.validation": [
        "BacktestConfig",
        "BacktestResult",
        "ForecastUnsupportedError",
        "PosteriorForecaster",
        "TrendExtrapolation",
        "audit_forward_pass",
        "audit_refit",
        "rebuild_like",
        "run_backtest",
    ],
    "mmm_framework.reporting": ["MMMReportGenerator", "ReportConfig"],
    "mmm_framework.platform.sessions": ["resolve_db_path", "init_db"],
    "mmm_framework.synth": ["generate_mff"],
    # Lean-importable agents service modules (see test_lean_imports.py).
    "mmm_framework.agents.fitting": ["build_model", "build_and_fit"],
}


@pytest.mark.parametrize("module_name", sorted(PUBLIC_SURFACE))
def test_public_symbols_present(module_name):
    import importlib

    module = importlib.import_module(module_name)
    missing = [n for n in PUBLIC_SURFACE[module_name] if not hasattr(module, n)]
    assert not missing, (
        f"{module_name} lost public symbols {missing} — breaking change "
        "(major version + changelog) or a bug."
    )


def test_package_version_is_semver_1x():
    import mmm_framework

    major = int(mmm_framework.__version__.split(".")[0])
    assert major >= 1


def test_package_version_matches_pyproject():
    """``__version__`` and ``pyproject.toml`` are two sources of truth for one fact.

    They drifted during the 1.3.1 release: bumping ``pyproject.toml`` leaves the
    hardcoded ``mmm_framework.__version__`` behind, and nothing noticed —
    ``docs/troubleshooting.html`` tells users to check exactly that attribute.
    """
    import tomllib
    from pathlib import Path

    import mmm_framework

    root = Path(__file__).resolve().parents[1]
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "version"
    ]
    assert mmm_framework.__version__ == declared, (
        f"mmm_framework.__version__ is {mmm_framework.__version__!r} but "
        f"pyproject.toml declares {declared!r}. Bump both — see the release "
        "checklist in CLAUDE.md."
    )


# ── Model-spec keys (agent spec registry) ─────────────────────────────────────

#: A plain (non-DAG) spec context — channel/control entries are {"name": ...}
#: dicts, the shape every agent tool writes.
_SPEC_CTX = {
    "kpi": "sales",
    "media_channels": [{"name": "tv"}],
    "control_variables": [{"name": "price"}],
}

#: Spec paths the v1.0 contract documents as consumed/writable for a plain
#: model spec. `unconsumed_spec_path` returning None means "consumed/valid".
#: Whole-list keys (media_channels, control_variables) are deliberately NOT
#: writable — the registry demands per-field addressing.
CONSUMED_SPEC_PATHS = [
    (["kpi"], "sales"),
    (["kpi_level"], "national"),
    (["time_granularity"], "weekly"),
    (["trend"], "linear"),
    (["media_prior_mode"], "roi"),
    (["inference", "method"], "nuts"),
    (["inference", "draws"], 500),
    (["inference", "chains"], 2),
    (["likelihood"], {"family": "normal"}),
    (["estimands"], []),
    (["experiments"], []),
    (["model_params"], {}),
    (["seasonality"], {"yearly": 2}),
    (["specification"], "multiplicative"),
    (["media_channels", "tv", "adstock", "type"], "geometric"),
    (["media_channels", "tv", "saturation", "type"], "logistic"),
]


@pytest.mark.parametrize(
    "parts,value",
    CONSUMED_SPEC_PATHS,
    ids=[".".join(p) for p, _ in CONSUMED_SPEC_PATHS],
)
def test_documented_spec_paths_stay_consumed(parts, value):
    from mmm_framework.agents.fitting import unconsumed_spec_path

    verdict = unconsumed_spec_path(parts, value, dict(_SPEC_CTX))
    assert verdict is None, (
        f"spec path {'.'.join(parts)} is now rejected as unconsumed: {verdict!r} — "
        "this breaks the documented v1.0 spec contract."
    )


def test_prior_paths_stay_consumed():
    from mmm_framework.agents.fitting import unconsumed_prior_path

    cases = [
        (["priors", "media", "tv", "roi"], {"median": 1.0, "sigma": 0.6}),
        (["priors", "media", "tv", "coefficient"], {"sigma": 0.5}),
        (["priors", "media_default", "roi_mu"], 0.0),
    ]
    for parts, value in cases:
        verdict = unconsumed_prior_path(parts, value, dict(_SPEC_CTX))
        assert verdict is None, (
            f"prior path {'.'.join(parts)} rejected: {verdict!r} — breaks the "
            "documented v1.0 prior contract."
        )


# ── Config enum values ────────────────────────────────────────────────────────

#: Enum VALUES are serialized into specs, saved configs, and run metadata —
#: removing/renaming one breaks stored artifacts. Additions are fine.
FROZEN_ENUM_VALUES = {
    "FitMethod": {
        "nuts",
        "map",
        "advi",
        "fullrank_advi",
        "pathfinder",
        "laplace",
        "smc",
    },
    "SaturationType": {
        "logistic",
        "hill",
        "michaelis_menten",
        "tanh",
        "root",
        "none",
    },
    "AdstockType": {"geometric", "delayed", "weibull", "none"},
    # InferenceMethod carries the PARADIGM (bayesian backend vs frequentist
    # estimator) and is serialized into specs, configs.json and run metrics —
    # losing a value strands stored runs. Note frequentist_ridge/cvxpy are
    # InferenceMethod values, NOT FitMethod (#228: the obvious edit was a
    # no-op).
    "InferenceMethod": {
        "bayesian_pymc",
        "bayesian_numpyro",
        "bayesian_nutpie",
        "frequentist_ridge",
        "frequentist_cvxpy",
    },
    "LikelihoodFamily": {
        "normal",
        "student_t",
        "lognormal",
        "binomial",
        "beta_binomial",
        "poisson",
        "negative_binomial",
        "beta",
    },
    "LinkFunction": {"identity", "logit", "log"},
    # Serialized into report/estimand payloads (measurement descriptors).
    "MeasurementUnit": {"spend", "impressions", "clicks", "other"},
}


@pytest.mark.parametrize("enum_name", sorted(FROZEN_ENUM_VALUES))
def test_enum_values_stay_available(enum_name):
    import mmm_framework.config as config

    enum_cls = getattr(config, enum_name)
    available = {e.value for e in enum_cls}
    missing = FROZEN_ENUM_VALUES[enum_name] - available
    assert not missing, (
        f"{enum_name} lost values {missing} — stored specs/configs referencing "
        "them will fail to load. Breaking change (major version)."
    )


# ── Built-in estimand names ───────────────────────────────────────────────────


def test_user_rows_never_expose_auth_columns(tmp_path, monkeypatch):
    """The users table mixes roster + auth columns (password_hash, token_version)
    since the auth layer augments it. The roster surface (`_user_row_to_dict`,
    consumed by /users) must project explicitly — leaking a hash is a security
    contract violation, not just an API break."""
    from mmm_framework.platform import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    from mmm_framework.auth import store as auth_store

    auth_store.init_auth_schema(db_path=tmp_path / "sessions.db")
    user = S.create_user("Contract Probe", email="probe@example.com")
    rows = S.list_users()
    assert rows, "probe user missing"
    forbidden = {"password_hash", "token_version"}
    for row in rows:
        leaked = forbidden & set(row)
        assert not leaked, f"user roster row leaks auth columns: {leaked}"
    assert set(user) == {
        "user_id",
        "name",
        "email",
        "role",
        "created_at",
        "updated_at",
    }


def test_builtin_estimand_names_stay_registered():
    from mmm_framework.estimands.registry import BUILTINS

    expected = {
        "contribution_roi",
        "counterfactual_roi",
        "marginal_roas",
        "contribution",
    }
    missing = expected - set(BUILTINS)
    assert not missing, (
        f"built-in estimands {missing} disappeared from the registry — specs "
        "and saved models reference them by name."
    )


# ---------------------------------------------------------------------------
# The private authoring surface (#279 PR 0.4)
#
# agents/garden_authoring.py promises a specific member list to the LLM as the
# garden authoring API, and until this contract NO test asserted any of those
# names existed. Phase 4's "the facade preserves the public surface, therefore
# minor" is unverifiable without pinning the ~dozen PRIVATE members registered
# garden models actually call.
# ---------------------------------------------------------------------------

#: (owner, name): "method" on BayesianMMM, or "module" in model.base. The
#: signature snapshot lives in tests/contracts/authoring_surface.json.
AUTHORING_SURFACE = {
    ("method", "_build_coords"),
    ("method", "_prepare_raw_media_for_model"),
    ("method", "_build_channel_saturation"),
    ("method", "_build_trend_component"),
    ("method", "_build_control_betas"),
    ("module", "_apply_saturation_pt"),
    ("module", "_sample_from_prior_config"),
}

#: Instance attributes garden_authoring.py names as present after __init__.
AUTHORING_ATTRS = [
    "channel_names",
    "y_mean",
    "y_std",
    "_media_raw_max",
    "panel",
    "model_config",
    "has_geo",
    "has_product",
    "n_obs",
    "n_channels",
    "n_controls",
    "time_idx",
    "X_controls",
    "n_periods",
    "seasonality_features",
]

#: pm.Data node names predict()/sample_channel_contributions() swap on.
AUTHORING_DATA_NODES = ["X_media_raw", "X_controls", "time_idx"]

_AUTHORING_SNAPSHOT = CONTRACTS_DIR / "authoring_surface.json"


def _authoring_signatures() -> dict[str, str]:
    import inspect

    import mmm_framework.model.base as MB
    from mmm_framework.model.base import BayesianMMM

    out = {}
    for owner, name in sorted(AUTHORING_SURFACE):
        obj = getattr(BayesianMMM if owner == "method" else MB, name, None)
        assert obj is not None, (
            f"authoring surface member {name!r} ({owner}) is GONE — "
            "agents/garden_authoring.py promises it to every garden author; "
            "restore it or update the doc AND this contract together"
        )
        out[f"{owner}:{name}"] = str(inspect.signature(obj))
    return out


def test_authoring_surface_signatures_match_snapshot():
    """Existence + inspect.signature equality against the checked-in snapshot.

    A signature change breaks every registered garden model that calls the
    member — that is a contract event, not an internal refactor. Regenerate
    deliberately with MMM_REGEN_AUTHORING_SURFACE=1 and review the diff.
    """
    import json
    import os

    observed = _authoring_signatures()
    if os.environ.get("MMM_REGEN_AUTHORING_SURFACE") == "1" or (
        not _AUTHORING_SNAPSHOT.exists()
    ):
        _AUTHORING_SNAPSHOT.write_text(json.dumps(observed, indent=2) + "\n")
        if os.environ.get("MMM_REGEN_AUTHORING_SURFACE") == "1":
            pytest.skip("snapshot regenerated; rerun without the env var")
    snapshot = json.loads(_AUTHORING_SNAPSHOT.read_text())
    assert observed == snapshot, (
        "private authoring-surface signatures drifted from "
        "tests/contracts/authoring_surface.json — every registered garden "
        "model calling the changed member breaks. Deliberate? "
        "MMM_REGEN_AUTHORING_SURFACE=1 and review."
    )


def test_authoring_surface_is_what_the_doc_promises():
    """The doc and the pin may not drift apart: every pinned name appears in
    agents/garden_authoring.py, and the doc says where the module-level
    helpers live."""
    root = Path(__file__).resolve().parents[1]
    doc = (root / "src" / "mmm_framework" / "agents" / "garden_authoring.py").read_text(
        encoding="utf-8"
    )
    missing = [n for _o, n in sorted(AUTHORING_SURFACE) if n not in doc]
    assert not missing, (
        f"pinned members {missing} no longer appear in garden_authoring.py — "
        "the pin and the promise must move together"
    )
    for attr in AUTHORING_ATTRS[:8]:
        assert attr in doc, f"promised attribute {attr!r} gone from the doc"


def test_authoring_attrs_and_data_nodes_exist_on_a_real_model():
    """The promised instance attributes exist after a real __init__, and the
    pm.Data nodes carry the promised names on the built graph."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "contracts"))
    import model_matrix as M

    m = M.CASES["default_national"]()
    missing = [a for a in AUTHORING_ATTRS if not hasattr(m, a)]
    assert not missing, f"promised instance attributes missing: {missing}"
    named = set(m.model.named_vars)
    gone = [n for n in AUTHORING_DATA_NODES if n not in named]
    assert not gone, (
        f"promised pm.Data nodes missing from the graph: {gone} — "
        "predict()/sample_channel_contributions() swap on these names"
    )
