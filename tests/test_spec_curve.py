"""Tests for the spec-curve / model-averaging engine (issue #103).

Fast tests exercise variant application, the runner, BMA, and robustness with
injected (no-MCMC) fit/roi functions, plus the report section. A slow E2E fits a
real 2-spec set with LOO-stacking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.validation.spec_curve import (
    SpecSet,
    SpecVariant,
    apply_variant,
    default_spec_variants,
    run_spec_curve,
)


def _base_spec():
    return {
        "kpi": "Sales",
        "media_channels": [
            {
                "name": "TV",
                "adstock": {"type": "geometric"},
                "saturation": {"type": "hill"},
            },
            {
                "name": "Search",
                "adstock": {"type": "geometric"},
                "saturation": {"type": "hill"},
            },
        ],
        "control_variables": [{"name": "price"}],
        "kpi_level": "national",
    }


# ---------------------------------------------------------------------------
# Variant application
# ---------------------------------------------------------------------------
class TestApplyVariant:
    def test_forms_applied_to_all_channels(self):
        v = SpecVariant(name="w×log", adstock="weibull", saturation="logistic")
        s = apply_variant(_base_spec(), v)
        assert all(m["adstock"]["type"] == "weibull" for m in s["media_channels"])
        assert all(m["saturation"]["type"] == "logistic" for m in s["media_channels"])

    def test_base_spec_untouched(self):
        base = _base_spec()
        apply_variant(base, SpecVariant(name="x", adstock="weibull"))
        assert base["media_channels"][0]["adstock"]["type"] == "geometric"

    def test_control_set_replaced(self):
        s = apply_variant(
            _base_spec(), SpecVariant(name="x", controls=["price", "promo"])
        )
        assert [c["name"] for c in s["control_variables"]] == ["price", "promo"]

    def test_pooling_prior_mode_trend_seasonality(self):
        v = SpecVariant(
            name="x",
            kpi_level="geo",
            media_prior_mode="coefficient",
            trend="spline",
            seasonality={"yearly": 4},
        )
        s = apply_variant(_base_spec(), v)
        assert s["kpi_level"] == "geo"
        assert s["media_prior_mode"] == "coefficient"
        assert s["trend"]["type"] == "spline"
        assert s["seasonality"] == {"yearly": 4}

    def test_overrides_deep_merge_last(self):
        v = SpecVariant(name="x", overrides={"inference": {"draws": 500}})
        s = apply_variant(_base_spec(), v)
        assert s["inference"]["draws"] == 500


class TestDefaultVariants:
    def test_grid_and_primary(self):
        variants = default_spec_variants(_base_spec())
        names = [v.name for v in variants]
        assert names == [
            "geometric×hill",
            "geometric×logistic",
            "weibull×hill",
            "weibull×logistic",
        ]
        primaries = [v for v in variants if v.primary]
        assert len(primaries) == 1
        assert primaries[0].name == "geometric×hill"  # the base combo

    def test_include_prior_mode_adds_sibling(self):
        variants = default_spec_variants(_base_spec(), include_prior_mode=True)
        assert any(v.media_prior_mode == "coefficient" for v in variants)


class TestSpecSet:
    def test_primary_and_names(self):
        ss = SpecSet(
            variants=[
                SpecVariant(name="a"),
                SpecVariant(name="b", primary=True),
            ]
        )
        assert ss.primary_variant.name == "b"
        assert ss.names() == ["a", "b"]

    def test_primary_defaults_to_first(self):
        ss = SpecSet(variants=[SpecVariant(name="a"), SpecVariant(name="b")])
        assert ss.primary_variant.name == "a"

    def test_serializable(self):
        ss = SpecSet(variants=[SpecVariant(name="a", adstock="weibull")], rationale="r")
        d = ss.model_dump()
        again = SpecSet(**d)
        assert again.variants[0].adstock == "weibull"


# ---------------------------------------------------------------------------
# Runner with injected fits (no MCMC)
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, channels, roi):
        self.channel_names = channels
        self._roi = roi
        self._trace = None


def _fake_fit(roi_by_sat):
    def fit(spec, path):
        sat = spec["media_channels"][0]["saturation"]["type"]
        return _FakeModel(["TV", "Search"], roi_by_sat[sat])

    return fit


def _fake_roi(model, channels, max_draws=400, random_seed=42):
    rng = np.random.default_rng(random_seed)
    return {
        ch: model._roi[ch] + 0.12 * rng.standard_normal(max_draws) for ch in channels
    }


class TestRunner:
    def _run(self, compute_loo=False):
        roi_by_sat = {
            "hill": {"TV": 2.5, "Search": 1.6},
            "logistic": {"TV": 2.3, "Search": 0.6},  # Search sign-flips across specs
        }
        return run_spec_curve(
            _base_spec(),
            "dummy.csv",
            compute_loo=compute_loo,
            fit_fn=_fake_fit(roi_by_sat),
            roi_fn=_fake_roi,
            max_draws=500,
        )

    def test_collects_per_spec_roi(self):
        res = self._run()
        assert len(res.specs) == 4
        assert all("TV" in f.roi and "Search" in f.roi for f in res.fits)

    def test_weights_sum_to_one(self):
        res = self._run()
        assert abs(sum(res.weights.values()) - 1.0) < 1e-6

    def test_bma_present_for_all_channels(self):
        res = self._run()
        assert set(res.bma) == {"TV", "Search"}
        assert all("mean" in res.bma[ch] for ch in res.bma)

    def test_robust_vs_fragile(self):
        res = self._run()
        # TV clusters tightly above break-even → robust; Search flips → fragile.
        assert res.robustness["TV"]["sign_stable"] is True
        assert res.robustness["Search"]["sign_stable"] is False
        assert res.robustness["Search"]["range"] > res.robustness["TV"]["range"]

    def test_primary_marked(self):
        res = self._run()
        assert res.primary == "geometric×hill"
        assert any(f.primary and f.name == res.primary for f in res.fits)

    def test_failing_spec_does_not_sink_sweep(self):
        def flaky_fit(spec, path):
            if spec["media_channels"][0]["saturation"]["type"] == "logistic":
                raise RuntimeError("bad geometry")
            return _FakeModel(["TV", "Search"], {"TV": 2.5, "Search": 1.6})

        res = run_spec_curve(
            _base_spec(),
            "dummy.csv",
            compute_loo=False,
            fit_fn=flaky_fit,
            roi_fn=_fake_roi,
        )
        failed = [f for f in res.fits if f.error]
        ok = [f for f in res.fits if not f.error]
        assert failed and ok  # some failed, some survived
        assert all("logistic" in f.name for f in failed)
        # BMA still computed from the surviving specs.
        assert "TV" in res.bma

    def test_to_dict_is_json_shaped(self):
        res = self._run()
        d = res.to_dict()
        assert set(d) >= {
            "channels",
            "specs",
            "primary",
            "weights",
            "bma",
            "robustness",
            "per_spec",
        }
        # No raw draws leak into the payload.
        assert "roi_draws" not in str(d)


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------
class TestReportSection:
    def _payload(self):
        return TestRunner()._run().to_dict()

    def test_section_renders_when_attached(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV", "Search"]),
            config=ReportConfig(),
            spec_curve=self._payload(),
        ).render()
        assert "Specification robustness" in html
        assert "specCurvePlot" in html
        assert "Spec-fragile" in html  # Search
        assert "Model-averaged" in html

    def test_section_absent_without_data(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV"]), config=ReportConfig()
        ).render()
        assert "Specification robustness" not in html

    def test_chart_returns_html(self):
        from mmm_framework.reporting import ReportConfig
        from mmm_framework.reporting.charts import create_spec_curve_plot

        div = create_spec_curve_plot(self._payload(), ReportConfig())
        assert "Plotly.newPlot" in div or "plotly" in div.lower()


# ---------------------------------------------------------------------------
# Real-fit end-to-end (slow) — the full build_model → fit → ROI → LOO-stack path.
# ---------------------------------------------------------------------------
def _write_mff(path):
    periods = pd.date_range("2021-01-04", periods=60, freq="W-MON")
    dims = {
        "Geography": None,
        "Product": None,
        "Campaign": None,
        "Outlet": None,
        "Creative": None,
    }
    rng = np.random.default_rng(7)
    rows = []
    for i, p in enumerate(periods):
        iso = p.strftime("%Y-%m-%d")
        tv = 100 + (i % 5) * 20 + rng.normal(0, 5)
        search = 60 + (i % 3) * 15 + rng.normal(0, 4)
        sales = 1000 + 1.5 * tv + 2.0 * search + 8 * i + rng.normal(0, 20)
        for name, val in [("Sales", sales), ("TV", tv), ("Search", search)]:
            rows.append(
                {**dims, "Period": iso, "VariableName": name, "VariableValue": val}
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.slow
def test_spec_curve_real_fit_with_stacking(tmp_path):
    dataset_path = _write_mff(tmp_path / "data.csv")
    base_spec = {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Search"}],
        "control_variables": [],
        "trend": {"type": "linear"},
        "inference": {"method": "nuts", "draws": 100, "tune": 100, "chains": 2},
    }
    # Two defensible specs: geometric×hill vs geometric×logistic.
    variants = [
        SpecVariant(
            name="geometric×hill", adstock="geometric", saturation="hill", primary=True
        ),
        SpecVariant(
            name="geometric×logistic", adstock="geometric", saturation="logistic"
        ),
    ]
    res = run_spec_curve(
        base_spec,
        dataset_path,
        variants=variants,
        compute_loo=True,
        max_draws=100,
        random_seed=0,
    )
    # Both specs fit and produced ROI for both channels.
    assert not any(f.error for f in res.fits), [f.error for f in res.fits]
    assert set(res.channels) == {"TV", "Search"}
    for f in res.fits:
        assert "TV" in f.roi and "Search" in f.roi
    # LOO-stacking weights are a proper distribution over the two specs.
    assert abs(sum(res.weights.values()) - 1.0) < 1e-3
    assert all(0.0 <= w <= 1.0 for w in res.weights.values())
    # BMA estimate present with a finite credible interval.
    assert res.bma["TV"]["mean"] > 0
    assert np.isfinite(res.bma["TV"]["lower"]) and np.isfinite(res.bma["TV"]["upper"])
