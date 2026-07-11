"""Tests for the interactive MMM Results Report.

Fast tests exercise the generator/insights on canned facts (no model, no
MCMC) plus the pure helpers in ``reporting/interactive/facts.py``; one slow
test drives the real pipeline end-to-end from an ADVI fit.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import pytest

from mmm_framework.reporting.interactive import (
    INTERACTIVE_INSIGHT_SLOTS,
    InteractiveReportGenerator,
    build_interactive_insights,
)
from mmm_framework.reporting.interactive.facts import (
    _b64_f32,
    _eti,
    _half_life_draws,
    _jsafe,
    _ppc_stat_facts,
    _series_stats,
    _stat_over_rows,
    _window_specs,
)

_XSS = "<script>alert(1)</script>"

_SECTION_IDS = (
    "insights",
    "executive-summary",
    "model-fit",
    "predictive-checks",
    "channel-roi",
    "estimands",
    "response-curves",
    "carryover",
    "prior-posterior",
    "reallocation",
    "sensitivity",
    "prior-predictive",
    "assumptions",
)


# ─────────────────────────────────────────────────────────────────────────────
# Canned facts (no model)
# ─────────────────────────────────────────────────────────────────────────────
def _summary(mean=1.5, lo=1.0, hi=2.0):
    return {"mean": mean, "lower": lo, "upper": hi}


def _canned_facts(*, approximate=False, channels=None) -> dict:
    rng = np.random.default_rng(0)
    ch_names = channels or ["TV", "Digital"]
    D, P, L = 6, 12, 5
    periods = [
        str(p)[:10] for p in pd.date_range("2024-01-01", periods=P, freq="W-MON")
    ]
    draws = {ch: rng.normal(100, 10, size=(D, P)) for ch in ch_names}
    band = lambda w: {  # noqa: E731
        "lo": list(1000 - w + rng.normal(0, 1, P)),
        "hi": list(1000 + w + rng.normal(0, 1, P)),
    }
    facts = {
        "meta": {
            "kpi": "Sales",
            "channels": ch_names,
            "geos": ["National"],
            "n_periods": P,
            "n_draws": D,
            "date_start": periods[0],
            "date_end": periods[-1],
            "interval": 0.9,
            "marginal_bump_pct": 5.0,
            "fit_method": "map" if approximate else "nuts",
            "approximate": approximate,
        },
        "periods": periods,
        "actual_national": list(1000 + rng.normal(0, 30, P)),
        "fit": {
            "series": {
                "National": {
                    "actual": list(1000 + rng.normal(0, 30, P)),
                    "mean": list(1000 + rng.normal(0, 10, P)),
                    "bands": {
                        k: band(w)
                        for k, w in (("50", 20), ("80", 40), ("90", 55), ("95", 70))
                    },
                    "stats": {"r2": 0.91, "rmse": 25.0, "mape": 2.1, "coverage90": 0.9},
                }
            },
            "order": ["National"],
            "band_levels": [0.5, 0.8, 0.9, 0.95],
        },
        "ppc_stats": {
            "n_draws": 100,
            "series": "national period-summed KPI",
            "stats": [
                {
                    "key": k,
                    "label": lbl,
                    "desc": "a property",
                    "observed": 1.0,
                    "rep_mean": 1.02,
                    "bayes_p": p,
                    "extreme": bool(p < 0.05 or p > 0.95),
                    "hist": {
                        "edges": list(np.linspace(0, 2, 11)),
                        "counts": [1, 2, 5, 9, 14, 18, 14, 9, 5, 2],
                    },
                }
                for k, lbl, p in (
                    ("mean", "Mean", 0.48),
                    ("max", "Maximum", 0.61),
                    ("acf1", "Lag-1 autocorrelation", 0.44),
                )
            ],
        },
        "contrib": {
            "n_draws": D,
            "draws_b64": {ch: _b64_f32(draws[ch]) for ch in ch_names},
        },
        "marginal": {
            "bump_pct": 5.0,
            "draws_b64": {ch: _b64_f32(draws[ch] * 0.04) for ch in ch_names},
        },
        "spend": {ch: list(rng.uniform(50, 150, P)) for ch in ch_names},
        "divisor_meta": {
            ch: {
                "is_monetary": True,
                "roi_label": "ROI",
                "marginal_label": "mROAS",
                "value_units": "KPI units",
                "divisor_units": "$",
                "reference": 1.0,
            }
            for ch in ch_names
        },
        "curves": {
            "multipliers": [0.0, 0.5, 1.0, 1.5, 2.0],
            "n_draws": D,
            "draws_b64": {
                ch: _b64_f32(rng.normal(1000, 50, size=(D, L))) for ch in ch_names
            },
            "spend_total": {ch: 1200.0 for ch in ch_names},
            "n_periods": P,
        },
        "carryover": {
            ch: {
                "family": "geometric",
                "lags": list(range(5)),
                "median": [0.5, 0.25, 0.12, 0.06, 0.03],
                "lower": [0.4, 0.2, 0.1, 0.05, 0.02],
                "upper": [0.6, 0.3, 0.15, 0.08, 0.04],
                "half_life": {"mean": 1.1, "lower": 0.7, "upper": 1.8},
            }
            for ch in ch_names
        },
        "prior_posterior": {
            "interval": 0.9,
            "rows": [
                {
                    "channel": ch,
                    "label": "ROI",
                    "reference": 1.0,
                    "grid": list(np.linspace(0, 4, 50)),
                    "posterior": {
                        "density": list(rng.uniform(0.1, 1, 50)),
                        "mean": 1.5,
                        "lower": 1.1,
                        "upper": 2.0,
                    },
                    "prior": {
                        "density": list(rng.uniform(0.05, 0.4, 50)),
                        "mean": 1.0,
                        "lower": 0.2,
                        "upper": 4.0,
                    },
                }
                for ch in ch_names
            ],
        },
        "sensitivity": {
            "estimand_label": "Contribution ROI",
            "specs": ["Base (full window)", "First half", "Second half"],
            "series": {
                ch: {
                    "mean": [1.5, 1.4, 1.6],
                    "lower": [1.1, 0.9, 1.2],
                    "upper": [2.0, 1.9, 2.1],
                }
                for ch in ch_names
            },
            "references": {ch: 1.0 for ch in ch_names},
            "interval": 0.9,
            "notes": ["Window specs re-aggregate the same posterior draws."],
        },
        "ppc_prior": {
            "kpi_label": "Sales",
            "dates": periods,
            "observed": list(1000 + rng.normal(0, 30, P)),
            "bands": {
                k: list(v + rng.normal(0, 5, P))
                for k, v in (
                    ("p05", np.full(P, 700.0)),
                    ("p25", np.full(P, 900.0)),
                    ("p50", np.full(P, 1000.0)),
                    ("p75", np.full(P, 1100.0)),
                    ("p95", np.full(P, 1300.0)),
                )
            },
            "traces": [list(1000 + rng.normal(0, 60, P)) for _ in range(3)],
            "coverage_90": 0.92,
            "scale_z_abs_mean": 0.8,
            "frac_negative": 0.0,
            "rep_means": list(rng.normal(1000, 60, 100)),
            "rep_sds": list(rng.uniform(20, 80, 100)),
            "obs_mean": 1002.0,
            "obs_sd": 31.0,
            "n_draws": 100,
        },
        "headline": {
            "total_kpi": 12000.0,
            "media_total": _summary(2400, 2000, 2800),
            "blended_roi": {**_summary(1.4, 1.1, 1.8), "spend": 1700.0},
            "media_share": _summary(0.2, 0.17, 0.23),
            "channels": [
                {
                    "channel": ch,
                    "roi_mean": 1.5 + 0.2 * i,
                    "roi_lower": 0.9 + 0.2 * i,
                    "roi_upper": 2.1 + 0.2 * i,
                    "spend": 1200.0,
                    "is_monetary": True,
                    "label": "ROI",
                    "reference": 1.0,
                }
                for i, ch in enumerate(ch_names)
            ],
            "fit": {"r2": 0.91, "rmse": 25.0, "mape": 2.1, "coverage90": 0.9},
            "interval": 0.9,
        },
        "assumptions": [
            {
                "topic": "Likelihood",
                "setting": "normal",
                "detail": "identity link",
                "channels": [],
            }
        ],
        "diagnostics": {
            "fit_method": "map" if approximate else "nuts",
            "approximate": approximate,
            "rhat_max": None if approximate else 1.004,
            "ess_bulk_min": None if approximate else 512.0,
            "divergences": 0,
            "converged": None if approximate else True,
        },
    }
    return facts


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestFactsHelpers:
    def test_b64_roundtrip(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)
        raw = base64.b64decode(_b64_f32(arr))
        out = np.frombuffer(raw, dtype="<f4").reshape(3, 4)
        np.testing.assert_allclose(out, arr)

    def test_eti_central_interval(self):
        lo, hi = _eti(np.linspace(0, 1, 1001), 0.9)
        assert lo == pytest.approx(0.05, abs=0.01)
        assert hi == pytest.approx(0.95, abs=0.01)

    def test_jsafe_replaces_nonfinite(self):
        out = _jsafe({"a": np.array([1.0, np.nan]), "b": np.float64(2.0)})
        assert out == {"a": [1.0, None], "b": 2.0}
        json.dumps(out)

    def test_series_stats_perfect_fit(self):
        a = np.linspace(100, 200, 20)
        stats = _series_stats(a, a.copy(), (a - 10, a + 10))
        assert stats["r2"] == pytest.approx(1.0)
        assert stats["coverage90"] == pytest.approx(1.0)
        assert stats["mape"] == pytest.approx(0.0)

    def test_half_life_monotone_in_alpha(self):
        lags = np.arange(9, dtype=float)
        slow = np.power(0.8, lags)[None, :]
        fast = np.power(0.2, lags)[None, :]
        assert _half_life_draws(slow)[0] > _half_life_draws(fast)[0]

    def test_stat_over_rows_acf1_detects_persistence(self):
        rng = np.random.default_rng(3)
        n = 200
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
        white = rng.standard_normal((1, n))
        acf_ar = _stat_over_rows("acf1", ar[None, :])[0]
        acf_white = _stat_over_rows("acf1", white)[0]
        assert acf_ar > 0.7
        assert abs(acf_white) < 0.3

    def test_ppc_stat_facts_calibrated_replicates(self):
        # Replicates drawn from the SAME process as the observed series →
        # every Bayesian p-value should be non-extreme.
        rng = np.random.default_rng(0)
        rep = rng.normal(100, 10, size=(400, 120))
        obs = rng.normal(100, 10, size=120)
        out = _ppc_stat_facts(rep, obs)
        keys = {s["key"] for s in out["stats"]}
        assert {"mean", "sd", "min", "max", "acf1", "skew"} <= keys
        for s in out["stats"]:
            assert 0.0 <= s["bayes_p"] <= 1.0
            assert not s["extreme"], f"{s['key']} unexpectedly extreme"
            assert sum(s["hist"]["counts"]) > 0

    def test_ppc_stat_facts_flags_missing_autocorrelation(self):
        # Observed series is strongly AR(1); replicates are white noise →
        # the acf1 statistic must be flagged extreme.
        rng = np.random.default_rng(1)
        n = 150
        obs = np.zeros(n)
        for t in range(1, n):
            obs[t] = 0.9 * obs[t - 1] + rng.standard_normal()
        rep = rng.standard_normal((300, n)) * obs.std()
        out = _ppc_stat_facts(rep, obs)
        acf = [s for s in out["stats"] if s["key"] == "acf1"][0]
        assert acf["extreme"]
        assert acf["bayes_p"] < 0.05

    def test_ppc_stat_facts_degrades_on_tiny_input(self):
        out = _ppc_stat_facts(np.zeros((5, 3)), np.zeros(3))
        assert out["stats"] == []

    def test_window_specs_shapes(self):
        P = 120
        periods = [
            str(p)[:10] for p in pd.date_range("2022-01-03", periods=P, freq="W-MON")
        ]
        spend = {"TV": np.random.default_rng(0).uniform(1, 10, P)}
        specs = _window_specs(periods, P, spend, ["TV"])
        labels = [s["label"] for s in specs]
        assert labels[0] == "Base (full window)"
        assert "Excl. last quarter" in labels
        assert "Excl. top-5 spend wks" in labels
        assert any(lbl.startswith("Excl. 20") for lbl in labels)
        for s in specs:
            assert s["masks"]["TV"].shape == (P,)
        # top-spend exclusion drops exactly 5 periods
        top = [s for s in specs if s["label"] == "Excl. top-5 spend wks"][0]
        assert (~top["masks"]["TV"]).sum() == 5


# ─────────────────────────────────────────────────────────────────────────────
# Insights
# ─────────────────────────────────────────────────────────────────────────────
class _FakeReply:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _FakeReply(self.content)


class _RaisingLLM:
    def invoke(self, messages):
        raise RuntimeError("LLM down")


class TestInsights:
    def test_templated_fills_every_slot(self):
        insights = build_interactive_insights(_canned_facts())
        for slot in INTERACTIVE_INSIGHT_SLOTS:
            assert insights.get(slot), f"empty slot {slot}"

    def test_llm_replaces_labelled_slots_only(self):
        llm = _FakeLLM("STANDFIRST: The custom story.\nROI: Custom roi gloss.")
        insights = build_interactive_insights(_canned_facts(), llm=llm)
        assert insights["standfirst"] == "The custom story."
        assert insights["roi_gloss"] == "Custom roi gloss."
        assert llm.calls == 1
        # unlabelled slots keep templated text
        assert "reallocator" in insights["realloc_gloss"]

    def test_llm_failure_keeps_templated(self):
        insights = build_interactive_insights(_canned_facts(), llm=_RaisingLLM())
        for slot in INTERACTIVE_INSIGHT_SLOTS:
            assert insights.get(slot)

    def test_approximate_flag_reaches_standfirst(self):
        insights = build_interactive_insights(_canned_facts(approximate=True))
        assert "APPROXIMATE" in insights["standfirst"]


# ─────────────────────────────────────────────────────────────────────────────
# Generator (canned facts)
# ─────────────────────────────────────────────────────────────────────────────
class TestGenerator:
    def test_requires_model_or_facts(self):
        with pytest.raises(ValueError):
            InteractiveReportGenerator()

    def test_full_document_structure(self):
        html = InteractiveReportGenerator(facts=_canned_facts()).generate_report()
        for sid in _SECTION_IDS:
            assert f'href="#{sid}"' in html, f"nav missing {sid}"
            assert f'id="{sid}"' in html, f"section missing {sid}"
        assert "__IR_DATA__" in html
        assert "__IR_THEME__" in html
        assert "Plotly.newPlot" in html  # prior-predictive static charts
        assert "cdn.plot.ly" in html

    def test_payload_is_parseable_json(self):
        html = InteractiveReportGenerator(facts=_canned_facts()).generate_report()
        blob = html.split("window.__IR_DATA__ = ", 1)[1].split(";</script>", 1)[0]
        payload = json.loads(blob.replace("<\\/", "</"))
        assert payload["meta"]["channels"] == ["TV", "Digital"]
        assert set(payload["contrib"]["draws_b64"]) == {"TV", "Digital"}
        assert len(payload["periods"]) == payload["meta"]["n_periods"]

    def test_xss_channel_names_are_escaped(self):
        facts = _canned_facts(channels=["TV", _XSS])
        html = InteractiveReportGenerator(facts=facts).generate_report()
        assert _XSS not in html
        assert "&lt;script&gt;alert" in html

    def test_approximate_banner_gating(self):
        html_ok = InteractiveReportGenerator(facts=_canned_facts()).generate_report()
        assert "Approximate fit" not in html_ok
        html_ap = InteractiveReportGenerator(
            facts=_canned_facts(approximate=True)
        ).generate_report()
        assert "Approximate fit (MAP)" in html_ap
        assert "chip-approx" in html_ap

    def test_missing_data_drops_sections(self):
        facts = _canned_facts()
        facts["carryover"] = {}
        facts["prior_posterior"]["rows"] = []
        facts["ppc_prior"] = None
        facts["ppc_stats"] = {"stats": []}
        html = InteractiveReportGenerator(facts=facts).generate_report()
        for sid in (
            "carryover",
            "prior-posterior",
            "prior-predictive",
            "predictive-checks",
        ):
            assert f'href="#{sid}"' not in html
        for sid in ("insights", "channel-roi", "reallocation"):
            assert f'id="{sid}"' in html

    def test_ppc_stats_table_and_verdicts(self):
        html = InteractiveReportGenerator(facts=_canned_facts()).generate_report()
        assert 'id="predictive-checks"' in html
        assert 'id="ppcStatsGrid"' in html
        assert "Lag-1 autocorrelation" in html
        assert html.count("t-scale") >= 3  # all canned stats consistent
        assert "All test statistics are consistent" in html

    def test_ppc_stats_extreme_flagged(self):
        facts = _canned_facts()
        facts["ppc_stats"]["stats"][1]["bayes_p"] = 0.99
        facts["ppc_stats"]["stats"][1]["extreme"] = True
        insights_html = InteractiveReportGenerator(facts=facts).generate_report()
        assert "t-reduce" in insights_html
        assert "systematically fails to reproduce" in insights_html
        assert "Maximum" in insights_html
        gloss = build_interactive_insights(facts)["ppc_stats_gloss"]
        assert "1 of 3" in gloss and "Maximum" in gloss


# ─────────────────────────────────────────────────────────────────────────────
# Slow: real fit → facts → report
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def fitted(self):
        from mmm_framework.config import (
            ControlVariableConfig,
            DimensionType,
            InferenceMethod,
            KPIConfig,
            MediaChannelConfig,
            MFFConfig,
            ModelConfig,
        )
        from mmm_framework.data_loader import PanelCoordinates, PanelDataset
        from mmm_framework.model import BayesianMMM
        from mmm_framework.model.trend_config import TrendConfig, TrendType

        periods = pd.date_range("2023-01-02", periods=80, freq="W-MON")
        n = len(periods)
        rng = np.random.default_rng(7)
        tv = np.abs(rng.standard_normal(n) * 50 + 100)
        dig = np.abs(rng.standard_normal(n) * 30 + 80)
        y = pd.Series(
            1000 + 2.0 * tv + 1.2 * dig + rng.standard_normal(n) * 60,
            name="Sales",
        )
        cfg = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
            ],
            controls=[
                ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
            ],
        )
        panel = PanelDataset(
            y=y,
            X_media=pd.DataFrame({"TV": tv, "Digital": dig}),
            X_controls=pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5}),
            coords=PanelCoordinates(
                periods=periods,
                geographies=None,
                products=None,
                channels=["TV", "Digital"],
                controls=["Price"],
            ),
            index=periods,
            config=cfg,
        )
        model = BayesianMMM(
            panel,
            ModelConfig(
                inference_method=InferenceMethod.BAYESIAN_PYMC,
                n_chains=1,
                n_draws=100,
                n_tune=100,
            ),
            TrendConfig(type=TrendType.LINEAR),
        )
        results = model.fit(method="advi")
        return model, results

    def test_facts_and_report(self, fitted, tmp_path):
        model, results = fitted
        gen = InteractiveReportGenerator(
            model,
            results,
            max_draws=60,
            curve_max_draws=30,
            curve_multipliers=(0.0, 0.5, 1.0, 1.5, 2.0),
            n_prior_samples=80,
        )
        f = gen.facts
        # per-draw matrices decode to the advertised shape
        D, P = f["contrib"]["n_draws"], len(f["periods"])
        raw = base64.b64decode(f["contrib"]["draws_b64"]["TV"])
        assert np.frombuffer(raw, dtype="<f4").shape == (D * P,)
        assert P == 80
        # headline rows carry CIs
        rows = f["headline"]["channels"]
        assert {r["channel"] for r in rows} == {"TV", "Digital"}
        for r in rows:
            assert r["roi_lower"] <= r["roi_mean"] <= r["roi_upper"]
        # sensitivity includes the counterfactual estimator spec
        assert "Zero-out counterfactual" in f["sensitivity"]["specs"]
        # posterior-predictive test statistics carry valid Bayesian p-values
        stats = f["ppc_stats"]["stats"]
        assert {s["key"] for s in stats} >= {"mean", "sd", "min", "max", "acf1"}
        for s in stats:
            assert 0.0 <= s["bayes_p"] <= 1.0
        # approximate provenance surfaced
        assert f["meta"]["approximate"] is True
        out = tmp_path / "report.html"
        gen.save_report(str(out))
        html = out.read_text()
        for sid in _SECTION_IDS:
            assert f'id="{sid}"' in html
        assert "Approximate fit (ADVI)" in html

    def test_facts_require_fit(self):
        from mmm_framework.reporting.interactive import interactive_report_facts

        class _NoTrace:
            _trace = None

        with pytest.raises(ValueError):
            interactive_report_facts(_NoTrace())
