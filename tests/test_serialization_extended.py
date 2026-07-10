"""MMMSerializer round-trips for the BaseExtendedMMM family.

The serializer's core save/load contract is panel-shaped (media
re-standardization, panel-compatibility gates) and used to crash on extension
models at ``model._VERSION`` — every `build_and_fit` auto-save of a
Nested/Multivariate/Combined/StructuralNested fit failed. The extended flavor
saves the pickled instance (arrays + configs; graph/trace stripped) with the
same platform-legible sidecars, and loads panel-free.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.serialization import MMMSerializer

_T = 50


def _rng():
    return np.random.default_rng(7)


def _media(rng):
    X = rng.gamma(2.0, 40.0, size=(_T, 2))
    X[rng.random((_T, 2)) < 0.15] = 0.0
    return X


def _index():
    return pd.date_range("2024-01-01", periods=_T, freq="W-MON")


def _map_fit(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(method="map", random_seed=0)


def _make_nested():
    from mmm_framework.mmm_extensions import (
        MediatorConfig,
        NestedMMM,
        NestedModelConfig,
    )

    rng = _rng()
    X = _media(rng)
    aw = 40 + 0.3 * X[:, 0] + rng.normal(0, 3, _T)
    y = 1000 + 4 * aw + 2 * X[:, 1] + rng.normal(0, 30, _T)
    cfg = NestedModelConfig(
        mediators=(MediatorConfig(name="awareness"),),
        media_to_mediator_map={"awareness": ("TV",)},
    )
    return NestedMMM(
        X, y, ["TV", "Digital"], cfg, mediator_data={"awareness": aw}, index=_index()
    )


def _make_multivariate():
    from mmm_framework.mmm_extensions import (
        MultivariateMMM,
        MultivariateModelConfig,
        OutcomeConfig,
    )

    rng = _rng()
    X = _media(rng)
    y1 = 800 + 3 * X[:, 0] + rng.normal(0, 25, _T)
    y2 = 500 + 2 * X[:, 1] + rng.normal(0, 20, _T)
    cfg = MultivariateModelConfig(
        outcomes=(
            OutcomeConfig(name="sales_a", column="sales_a"),
            OutcomeConfig(name="sales_b", column="sales_b"),
        )
    )
    return MultivariateMMM(
        X,
        {"sales_a": y1, "sales_b": y2},
        ["TV", "Digital"],
        cfg,
        index=_index(),
    )


def _make_combined():
    from mmm_framework.mmm_extensions import (
        CombinedMMM,
        CombinedModelConfig,
        MediatorConfig,
        MultivariateModelConfig,
        NestedModelConfig,
        OutcomeConfig,
    )

    rng = _rng()
    X = _media(rng)
    aw = 40 + 0.3 * X[:, 0] + rng.normal(0, 3, _T)
    y1 = 800 + 3 * aw + rng.normal(0, 25, _T)
    y2 = 500 + 2 * X[:, 1] + rng.normal(0, 20, _T)
    cfg = CombinedModelConfig(
        nested=NestedModelConfig(
            mediators=(MediatorConfig(name="awareness"),),
            media_to_mediator_map={"awareness": ("TV",)},
        ),
        multivariate=MultivariateModelConfig(
            outcomes=(
                OutcomeConfig(name="sales_a", column="sales_a"),
                OutcomeConfig(name="sales_b", column="sales_b"),
            )
        ),
    )
    return CombinedMMM(
        X,
        {"sales_a": y1, "sales_b": y2},
        ["TV", "Digital"],
        cfg,
        mediator_data={"awareness": aw},
        index=_index(),
    )


def _make_structural():
    from mmm_framework.mmm_extensions import (
        StructuralNestedConfig,
        StructuralNestedMMM,
        binary_survey_mediator,
    )
    from mmm_framework.synth.dgp import make_brand_funnel

    sc = make_brand_funnel(seed=21, n_weeks=_T)
    cfg = StructuralNestedConfig(
        mediators=(binary_survey_mediator("awareness", ["TV"], affects_outcome=True),),
    )
    return StructuralNestedMMM(
        sc.spend.to_numpy(float),
        sc.y.to_numpy(float),
        sc.channels,
        cfg,
        mediator_data={"awareness": sc.notes["awareness_counts"]},
        mediator_trials={"awareness": sc.notes["awareness_trials"]},
        index=sc.weeks,
    )


_MAKERS = {
    "NestedMMM": _make_nested,
    "MultivariateMMM": _make_multivariate,
    "CombinedMMM": _make_combined,
    "StructuralNestedMMM": _make_structural,
}


@pytest.mark.parametrize("name", sorted(_MAKERS))
def test_extended_round_trip(name, tmp_path):
    model = _MAKERS[name]()
    _map_fit(model)

    save_dir = tmp_path / "run"
    MMMSerializer.save(model, save_dir)

    # Sidecars are platform-legible
    meta = json.loads((save_dir / "metadata.json").read_text())
    assert meta["model_flavor"] == "extended"
    assert meta["model_class_qualname"].endswith(name)
    assert meta["fit_method"] == "map"
    assert meta["approximate"] is True
    assert (save_dir / "configs.json").exists()

    # Panel-free load, class + trace preserved
    from mmm_framework.utils.arviz_compat import has_group

    loaded = MMMSerializer.load(save_dir)
    assert type(loaded).__name__ == name
    assert loaded._trace is not None
    assert has_group(loaded._trace, "posterior")
    assert loaded.channel_names == model.channel_names


def test_nested_mediation_effects_after_reload(tmp_path):
    model = _make_nested()
    _map_fit(model)
    MMMSerializer.save(model, tmp_path / "run")
    loaded = MMMSerializer.load(tmp_path / "run")
    me = loaded.get_mediation_effects()
    assert set(me["channel"]) == {"TV", "Digital"}
    assert np.all(np.isfinite(me["total_effect"]))


def test_structural_counterfactuals_after_reload(tmp_path):
    """The strongest post-load check: exact counterfactual ROAS requires the
    reloaded pickle to rebuild the full PyMC graph from its arrays and run
    set_data + sample_posterior_predictive against the reattached trace."""
    model = _make_structural()
    _map_fit(model)
    MMMSerializer.save(model, tmp_path / "run")
    loaded = MMMSerializer.load(tmp_path / "run")
    roas = loaded.get_channel_roas()
    assert set(roas["channel"]) == set(model.channel_names)
    assert np.all(np.isfinite(roas["contribution"]))


def test_unfitted_extended_save_loads_without_trace(tmp_path):
    model = _make_nested()
    MMMSerializer.save(model, tmp_path / "run")
    loaded = MMMSerializer.load(tmp_path / "run", rebuild_model=False)
    assert loaded._trace is None
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert "fit_method" not in meta  # never fitted -> no provenance claimed


def test_core_load_without_panel_is_a_clear_error(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "metadata.json").write_text(json.dumps({"version": "x"}))
    with pytest.raises(TypeError, match="panel"):
        MMMSerializer.load(tmp_path / "run")


def test_extended_version_drift_warns(tmp_path):
    model = _make_nested()
    _map_fit(model)
    MMMSerializer.save(model, tmp_path / "run")
    meta_path = tmp_path / "run" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="compatibility"):
        MMMSerializer.load(tmp_path / "run", rebuild_model=False)


# ── the fit-path auto-save gap this closes (E2E) ──────────────────────────────


@pytest.mark.slow
def test_build_and_fit_autosave_and_reload_for_extension(tmp_path, monkeypatch):
    """`build_and_fit` auto-save previously FAILED for every extension model
    (MMMSerializer read `model._VERSION`, absent on BaseExtendedMMM). Now: a
    DAG-routed nested fit persists, and `load_model_core` reloads it
    panel-free (no dataset requirement for extended saves)."""
    monkeypatch.chdir(tmp_path)
    import mmm_framework.reporting.generator as gen

    monkeypatch.setattr(
        gen,
        "ReportBuilder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped in test")),
    )

    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.agents.tools import load_model_core
    from mmm_framework.dag_model_builder.frontend_adapter import (
        create_mediation_dag,
    )

    # Mediated MFF (TV -> Awareness -> Sales)
    rng = np.random.default_rng(7)
    periods = pd.date_range("2024-01-01", periods=50, freq="W-MON")
    tv = np.abs(rng.normal(100, 20, len(periods)))
    digital = np.abs(rng.normal(60, 15, len(periods)))
    awareness = 40 + 0.3 * tv + rng.normal(0, 4, len(periods))
    sales = 1000 + 4.0 * awareness + 2.0 * digital + rng.normal(0, 40, len(periods))
    rows = []
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
    dataset_path = str(tmp_path / "mff.csv")
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    dag = create_mediation_dag(
        kpi_name="Sales",
        media_names=["TV", "Digital"],
        mediator_name="Awareness",
        include_direct_effects=True,
    )
    spec = {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "time_granularity": "weekly",
        "dag_model_type": "nested_mmm",
        "dag_spec": dag.model_dump(),
        "inference": {"method": "map", "metrics_draws": 0},
    }
    mmm, results, info = build_and_fit(spec, dataset_path)
    assert type(mmm).__name__ == "NestedMMM"
    assert results.approximate is True
    # THE gap: auto-save must now succeed for extension models.
    assert info["model_run"]["model_path"] is not None, info["model_run"]
    run_name = info["model_run"]["run_name"]

    # Panel-free reload through the agent surface: no spec, no dataset.
    res = load_model_core(None, run_name, None, None)
    assert res["ok"], res["message"]
    assert "loaded" in res["message"]


def test_legacy_base_extended_save_dir_loads(tmp_path):
    """An old-flavor BaseExtendedMMM.save directory (model.pkl + trace.nc, no
    metadata sidecars) loads through MMMSerializer instead of a bare
    metadata.json FileNotFoundError."""
    model = _make_nested()
    _map_fit(model)
    model.save(tmp_path / "legacy")  # the old flavor
    loaded = MMMSerializer.load(tmp_path / "legacy", rebuild_model=False)
    assert type(loaded).__name__ == "NestedMMM"
    assert loaded._trace is not None


def test_cross_flavor_base_load_reads_gzipped_trace(tmp_path):
    """BaseExtendedMMM.load on a serializer save (which gzips the trace by
    default) must not silently drop the posterior."""
    from mmm_framework.mmm_extensions import NestedMMM

    model = _make_nested()
    _map_fit(model)
    MMMSerializer.save(model, tmp_path / "run")
    assert (tmp_path / "run" / "trace.nc.gz").exists()
    loaded = NestedMMM.load(tmp_path / "run")
    assert loaded._trace is not None


def test_reload_preserves_approximate_provenance(tmp_path):
    """load_model_core's results rebuild: a reloaded MAP fit keeps flagging
    approximate (the review found fit_results=None dropped the provenance)."""
    from mmm_framework.agents.tools import _extended_results_from_disk

    model = _make_nested()
    _map_fit(model)
    MMMSerializer.save(model, tmp_path / "run")
    loaded = MMMSerializer.load(tmp_path / "run", rebuild_model=False)
    results = _extended_results_from_disk(loaded)
    assert results is not None
    assert results.approximate is True
    assert results.diagnostics["fit_method"] == "map"
