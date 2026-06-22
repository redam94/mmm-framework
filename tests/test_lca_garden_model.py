"""End-to-end proof for a second non-MMM family — Bayesian Latent Class Analysis.
Builds, fits, recovers a planted 2-class structure (sizes + per-class item
profiles), surfaces class-size estimands, round-trips, renders the latent-structure
report section, and passes the capability-gated compat suite."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from bayesian_lca import BayesianLCA, LCAConfig, synthetic_lca_panel  # noqa: E402


def _build(n=400, **params):
    panel, sizes, profiles = synthetic_lca_panel(n=n)
    mp = {"n_classes": 2, **params}
    mmm = BayesianLCA(
        panel, ModelConfig(), TrendConfig(type=TrendType.NONE), model_params=mp
    )
    return mmm, sizes, profiles


class TestStructure:
    def test_is_non_mmm_with_config_schema(self):
        from mmm_framework.garden.contract import is_mmm_model, model_kind

        assert BayesianLCA.__garden_model_kind__ == "latent_class"
        assert model_kind(BayesianLCA) == "latent_class"
        assert not is_mmm_model(BayesianLCA)
        assert BayesianLCA.CONFIG_SCHEMA is LCAConfig

    def test_prepare_data_binarizes_items(self):
        mmm, _, _ = _build()
        assert mmm.channel_names == []
        assert mmm.n_items == 6
        assert mmm.items.shape == (400, 6)
        assert set(np.unique(mmm.items)) <= {0.0, 1.0}  # binary

    def test_passes_non_mmm_contract(self):
        from mmm_framework.garden.contract import validate_class, validate_instance

        assert validate_class(BayesianLCA) == []
        mmm, _, _ = _build()
        assert validate_instance(mmm) == []

    def test_n_classes_min_two(self):
        with pytest.raises(Exception):  # ge=2 validator
            _build(n_classes=1)


@pytest.mark.slow
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def fitted(self):
        mmm, sizes, profiles = _build(n=600)
        mmm.fit(method="map", random_seed=11)
        return mmm, sizes, profiles

    def test_recovers_class_sizes(self, fitted):
        mmm, sizes, _ = fitted
        res = mmm.evaluate_estimands()  # class_size_1, class_size_2
        s1 = res["class_size_1"].mean
        s2 = res["class_size_2"].mean
        # Ordered-by-size identification -> class 1 is the smaller (~0.35).
        assert abs(s1 - 0.35) < 0.1 and abs(s2 - 0.65) < 0.1
        assert abs((s1 + s2) - 1.0) < 1e-6

    def test_recovers_class_profiles(self, fitted):
        mmm, _, _ = fitted
        prof = mmm.class_profile_summary().pivot(
            index="item", columns="class", values="prob"
        )
        # C1 endorses q1-q3 (~0.85) and rejects q4-q6; C2 is the mirror image.
        assert prof.loc["q1", "C1"] > 0.6 and prof.loc["q1", "C2"] < 0.4
        assert prof.loc["q5", "C2"] > 0.6 and prof.loc["q5", "C1"] < 0.4

    def test_class_size_estimands(self, fitted):
        mmm, _, _ = fitted
        res = mmm.evaluate_estimands()
        assert {"class_size_1", "class_size_2"} <= set(res)
        for r in res.values():
            assert r.status == "ok" and 0.0 <= r.mean <= 1.0

    def test_capabilities_expose_class_sizes(self, fitted):
        mmm, _, _ = fitted
        caps = mmm.model_capabilities()
        assert "HAS_LATENT:class_size_1" in caps
        assert "HAS_CHANNELS" not in caps

    def test_serialization_round_trip(self, fitted, tmp_path):
        from mmm_framework.serialization import MMMSerializer

        mmm, _, _ = fitted
        path = str(tmp_path / "lca_model")
        MMMSerializer.save(mmm, path)
        loaded = MMMSerializer.load(path, mmm.panel)
        assert type(loaded).__name__ == "BayesianLCA"
        assert loaded.model_params.n_classes == 2
        assert loaded._trace is not None
        assert loaded.evaluate_estimands()["class_size_1"].status == "ok"

    def test_report_renders_latent_class_section(self, fitted):
        from mmm_framework.reporting import MMMReportGenerator

        mmm, _, _ = fitted
        html = MMMReportGenerator(model=mmm).render()
        assert "Latent Class Analysis" in html
        assert "Class item-endorsement profiles" in html
        assert 'id="channel-roi"' not in html  # MMM sections gated off


@pytest.mark.slow
def test_compat_suite_passes_for_lca():
    from mmm_framework.garden.compat import run_compatibility_check

    report = run_compatibility_check(
        BayesianLCA,
        scenarios=("clean",),
        fit_method="map",
        n_weeks=60,
        check_carryover=False,
    )
    tiers = {t["name"]: t for t in report["tiers"]}
    assert report["blocking_passed"], report.get("summary")
    for name in ("scaling", "ops_smoke"):
        assert tiers[name]["skipped"], (name, tiers.get(name))
    for name in ("static", "build", "fit", "instance", "trace"):
        assert tiers[name]["passed"] and not tiers[name]["skipped"], (
            name,
            tiers.get(name),
        )
