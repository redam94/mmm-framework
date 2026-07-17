"""DAG-routed extension models through the agent fit path.

``build_model_from_dag`` stamps ``dag_model_type`` (+ ``dag_spec`` for
extension types) onto the model spec; ``agents/fitting.build_model`` must
route those to DAGModelBuilder — previously it ignored ``dag_model_type`` and
silently fit a plain BayesianMMM that dropped the mediation / multi-outcome
structure the DAG was validated for.
"""

import numpy as np
import pandas as pd
import pytest

from mmm_framework.agents.fitting import build_and_fit, build_model
from mmm_framework.dag_model_builder.frontend_adapter import (
    create_mediation_dag,
)


@pytest.fixture(scope="module")
def mediated_mff(tmp_path_factory) -> str:
    """A tiny national MFF dataset with a mediator path TV→Awareness→Sales."""
    rng = np.random.default_rng(7)
    periods = pd.date_range("2024-01-01", periods=50, freq="W-MON")
    rows = []
    tv = np.abs(rng.normal(100, 20, len(periods)))
    digital = np.abs(rng.normal(60, 15, len(periods)))
    awareness = 40 + 0.3 * tv + rng.normal(0, 4, len(periods))
    sales = 1000 + 4.0 * awareness + 2.0 * digital + rng.normal(0, 40, len(periods))
    for i, p in enumerate(periods):
        for var, val in [
            ("Sales", sales[i]),
            ("TV", tv[i]),
            ("Digital", digital[i]),
            ("Awareness", awareness[i]),
        ]:
            rows.append(
                {
                    "Geography": None,
                    "Product": None,
                    "Campaign": None,
                    "Outlet": None,
                    "Creative": None,
                    "Period": p.strftime("%Y-%m-%d"),
                    "VariableName": var,
                    "VariableValue": val,
                }
            )
    path = tmp_path_factory.mktemp("ext_fit") / "mff.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _nested_spec() -> dict:
    dag = create_mediation_dag(
        kpi_name="Sales",
        media_names=["TV", "Digital"],
        mediator_name="Awareness",
        include_direct_effects=True,
    )
    return {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "time_granularity": "weekly",
        "trend": {"type": "linear"},
        "dag_model_type": "nested_mmm",
        "dag_spec": dag.model_dump(),
        "inference": {"chains": 1, "draws": 40, "tune": 40, "target_accept": 0.8},
    }


def test_build_model_routes_to_nested(mediated_mff):
    mmm = build_model(_nested_spec(), mediated_mff)
    assert type(mmm).__name__ == "NestedMMM"
    assert mmm.n_channels == 2


def test_missing_dag_spec_is_a_clear_error(mediated_mff):
    spec = _nested_spec()
    spec.pop("dag_spec")
    with pytest.raises(ValueError, match="dag_spec"):
        build_model(spec, mediated_mff)


def test_basic_dag_type_keeps_the_plain_path(mediated_mff):
    spec = _nested_spec()
    spec["dag_model_type"] = "bayesian_mmm"  # basic — the stamp is informational
    spec.pop("dag_spec")
    mmm = build_model(spec, mediated_mff)
    assert type(mmm).__name__ == "BayesianMMM"


def test_map_fit_supported_for_extensions(mediated_mff):
    """Extension models now share the base model's approximate engine — a MAP
    fit returns a drop-in approximate result instead of raising."""
    mmm = build_model(_nested_spec(), mediated_mff)
    assert type(mmm).__name__ == "NestedMMM"
    results = mmm.fit(method="map", random_seed=0)
    assert results.approximate is True
    assert results.diagnostics["fit_method"] == "map"
    # R-hat / ESS are undefined for a single-path approximation.
    assert results.diagnostics["rhat_max"] is None
    assert results.converged is None  # "not assessable", not False
    # The trace is a drop-in for a NUTS trace: (chain, draw) posterior present,
    # with the model's deterministics, and az summary works off it.
    assert results.trace.posterior.sizes["draw"] >= 1
    assert not results.summary().empty


@pytest.mark.slow
def test_advi_fit_supported_for_extensions(mediated_mff):
    """ADVI (variational) also works for an extension model."""
    mmm = build_model(_nested_spec(), mediated_mff)
    results = mmm.fit(method="advi", draws=50, random_seed=0, n=2000)
    assert results.approximate is True
    assert results.diagnostics["fit_method"] == "advi"
    assert results.trace.posterior.sizes["draw"] == 50


@pytest.mark.slow
def test_smc_fit_supported_for_extensions(mediated_mff):
    """SMC (exact Sequential Monte Carlo) works for an extension model and is
    NOT flagged approximate — R-hat/ESS across the independent runs plus the
    log marginal likelihood land in the diagnostics."""
    import warnings

    mmm = build_model(_nested_spec(), mediated_mff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny particle count -> low-ESS noise
        results = mmm.fit(
            method="smc", draws=80, chains=2, random_seed=0, cores=1, progressbar=False
        )
    assert results.approximate is False
    d = results.diagnostics
    assert d["fit_method"] == "smc"
    assert d["rhat_max"] is not None
    assert "log_marginal_likelihood" in d
    assert results.trace.posterior.sizes["chain"] == 2
    assert results.trace.posterior.sizes["draw"] == 80


def test_approximate_method_flows_through_the_agent_fit_path(mediated_mff):
    """The agent build_model → fit path passes method='map' to the extension
    model instead of raising (the old behavior)."""
    spec = _nested_spec()
    spec["inference"]["method"] = "map"
    mmm = build_model(spec, mediated_mff)
    results = mmm.fit(method=spec["inference"]["method"], random_seed=0)
    assert results.approximate is True


def test_spec_trend_seasonality_likelihood_and_priors_reach_nested_graph(mediated_mff):
    """End-to-end: a spec's trend + seasonality + outcome-likelihood family +
    mediator prior overrides all reach the DAG-routed NestedMMM graph."""
    spec = _nested_spec()
    spec["trend"] = {"type": "linear"}
    spec["seasonality"] = {"yearly": 2}
    spec["likelihood"] = {"family": "student_t"}
    spec["priors"] = {
        "mediator": {
            "Awareness": {"media_effect_sigma": 2.5, "direct_effect_sigma": 1.5}
        }
    }
    mmm = build_model(spec, mediated_mff)
    rvs = set(mmm.model.named_vars.keys())
    assert "trend_slope" in rvs  # spec trend
    assert "seasonality_coefs" in rvs  # spec seasonality
    assert "nu_y" in rvs  # student-t outcome likelihood
    # the mediator prior override reached the model's config
    med = mmm.config.mediators[0]
    assert med.media_effect.sigma == 2.5
    assert med.direct_effect.sigma == 1.5


@pytest.mark.slow
def test_spline_trend_extension_fits_end_to_end(mediated_mff):
    """A DAG-routed NestedMMM with a spec spline trend builds + fits (MAP)."""
    spec = _nested_spec()
    spec["trend"] = {"type": "spline"}
    mmm = build_model(spec, mediated_mff)
    assert "spline_coef_raw" in set(mmm.model.named_vars.keys())
    results = mmm.fit(method="map", random_seed=0)
    assert results.approximate is True
    assert "trend_component" in results.trace.posterior


@pytest.mark.slow
def test_map_fit_artifact_marks_model_and_method(mediated_mff, tmp_path, monkeypatch):
    """A MAP fit of a DAG-routed NestedMMM records WHICH model + HOW it was fit,
    so the artifact is self-describing (reports/settings read these)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    spec = _nested_spec()
    spec["inference"]["method"] = "map"
    spec["inference"]["metrics_draws"] = 0
    mmm, results, info = build_and_fit(spec, mediated_mff)
    run = info["model_run"]
    assert run["model_class"] == "NestedMMM"
    assert run["dag_model_type"] == "nested_mmm"
    assert run["approximate"] is True  # first-class, not just method != nuts
    assert run["inference"]["method"] == "map"
    assert results.approximate is True


@pytest.mark.slow
def test_build_and_fit_nested_end_to_end(mediated_mff, tmp_path, monkeypatch):
    # keep artifacts out of the repo cwd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    spec = _nested_spec()
    spec["inference"]["metrics_draws"] = 0  # skip heavy run-metrics snapshots
    mmm, results, info = build_and_fit(spec, mediated_mff)
    assert type(mmm).__name__ == "NestedMMM"
    run = info["model_run"]
    assert run["dag_model_type"] == "nested_mmm"
    assert run["model_class"] == "NestedMMM"
    assert "NestedMMM" in info["summary"]
