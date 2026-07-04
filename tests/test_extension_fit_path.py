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


def test_approximate_methods_refused_for_extensions(mediated_mff):
    spec = _nested_spec()
    spec["inference"]["method"] = "map"
    with pytest.raises(ValueError, match="not available"):
        build_and_fit(spec, mediated_mff)


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
