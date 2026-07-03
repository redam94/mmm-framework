"""Spec prior settings MUST reach the model graph (the silent-drop bug).

The agent writes priors at ``spec["priors"]["media"|"controls"|...]`` and
``update_model_setting`` validates those paths against the consumed-paths
registry — so every registered path must genuinely change the built graph.
Historically ``priors.media.<ch>.coefficient``, ``priors.controls.<cv>.
coefficient`` and ``allow_negative`` were accepted, wired into the configs by
``_mff_config_from_spec``, and then silently ignored by ``BayesianMMM`` (the
beta prior honored only the experiment-calibrated ``roi_prior``; controls used
fixed role-based widths) — the prefit readout's prior predictive never moved
when the agent changed them.

Contract pinned here:

- an EXPLICITLY set prior (agent spec / builder / direct constructor kwarg)
  changes the graph;
- an untouched config keeps the historical built-in priors byte-for-byte
  (``beta_<ch>`` stays the Gamma; ``beta_controls`` stays the single Normal RV).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mmm_framework.agents.fitting import build_model

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory) -> str:
    """A tiny national MFF dataset (Sales / TV / Digital / Price)."""
    rng = np.random.default_rng(0)
    periods = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    rows = []
    for p in periods:
        for var, val in [
            ("Sales", 1000 + rng.normal(0, 50)),
            ("TV", abs(rng.normal(100, 20))),
            ("Digital", abs(rng.normal(60, 15))),
            ("Price", 10 + rng.normal(0, 0.5)),
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
    path = tmp_path_factory.mktemp("prior_flow") / "mff.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


_BASE_SPEC = {
    "kpi": "Sales",
    "media_channels": [{"name": "TV"}, {"name": "Digital"}],
    "control_variables": [{"name": "Price"}],
    "time_granularity": "weekly",
    "skip_quality_gate": True,
}


def _spec(priors: dict | None = None) -> dict:
    spec = copy.deepcopy(_BASE_SPEC)
    if priors:
        spec["priors"] = priors
    return spec


def _prior_stats(model, names: list[str], n: int = 400) -> dict[str, tuple]:
    idata = model.sample_prior_predictive(n, 1)
    out = {}
    free = {rv.name for rv in model.model.free_RVs}
    for name in names:
        if name not in idata.prior:
            out[name] = None
            continue
        vals = np.asarray(idata.prior[name].values, dtype=float).reshape(-1)
        op = str(model.model[name].owner.op).split("{")[0] if name in free else "det"
        out[name] = (op, float(vals.mean()), float(vals.std()))
    return out


class TestDefaultsStayByteIdentical:
    """The LIBRARY default (media_prior_mode='coefficient') keeps the
    historical graph. The AGENT spec default is ROI mode (tested below), so the
    coefficient mode is pinned via the explicit opt-out."""

    def test_coefficient_mode_media_beta_is_the_historical_gamma(self, dataset_path):
        spec = _spec()
        spec["media_prior_mode"] = "coefficient"
        m = build_model(spec, dataset_path)
        stats = _prior_stats(m, ["beta_TV"])
        op, mean, _ = stats["beta_TV"]
        assert op == "gamma_rv"
        assert mean == pytest.approx(1.5, abs=0.15)

    def test_library_model_config_defaults_to_coefficient_mode(self):
        from mmm_framework.config import ModelConfig

        assert ModelConfig().media_prior_mode == "coefficient"

    def test_default_controls_are_the_single_normal_rv(self, dataset_path):
        m = build_model(_spec(), dataset_path)
        free = {rv.name for rv in m.model.free_RVs}
        assert "beta_controls" in free  # the historical single RV, not a Deterministic
        assert "beta_control_Price" not in free


class TestRoiModeDefaultPriors:
    """Agent-built models default to ROI-parameterized media priors: the free
    RV is the channel's ROI (LogNormal, median 1 = break-even) and beta is a
    derived Deterministic — so the DEFAULT prior lives on the decision scale
    and is comparable across channels regardless of spend."""

    def test_agent_default_is_roi_mode(self, dataset_path):
        m = build_model(_spec(), dataset_path)
        free = {rv.name for rv in m.model.free_RVs}
        assert {"roi_TV", "roi_Digital"} <= free
        assert "beta_TV" not in free  # beta is derived, not sampled
        assert "beta_TV" in [v.name for v in m.model.deterministics]

    def test_prior_roi_is_spend_scale_invariant(self, dataset_path):
        """TV spends ~10x Digital, yet both prior ROIs are the SAME LogNormal
        (median ≈ 1, P(ROI>1) ≈ 50%) — the point of the ROI parameterization."""
        from mmm_framework.reporting.helpers.prefit import (
            prior_estimand_facts,
            sample_prior,
        )

        m = build_model(_spec(), dataset_path)
        idata = sample_prior(m, 300, 3)
        est = prior_estimand_facts(m, idata)
        rows = {r["channel"]: r for r in est["channels"]}
        for ch in ("TV", "Digital"):
            r = rows[ch]
            assert 0.35 <= r["p_above_reference"] <= 0.65, ch
            assert r["lower"] < 1.0 < r["upper"], ch
        # And the reported prior ROI IS the sampled roi_<ch> prior.
        roi_tv = np.asarray(idata.prior["roi_TV"].values).reshape(-1)
        assert np.median(roi_tv) == pytest.approx(1.0, rel=0.25)

    def test_explicit_coefficient_prior_beats_roi_mode(self, dataset_path):
        m = build_model(
            _spec(
                {
                    "media": {
                        "TV": {
                            "coefficient": {
                                "distribution": "half_normal",
                                "params": {"sigma": 0.05},
                            }
                        }
                    }
                }
            ),
            dataset_path,
        )
        free = {rv.name for rv in m.model.free_RVs}
        assert "beta_TV" in free and "roi_TV" not in free
        assert "roi_Digital" in free  # untouched channel keeps the ROI default

    def test_roi_default_hyperparams_from_spec(self, dataset_path):
        spec = _spec({"media_default": {"roi_sigma": 0.2}})
        m = build_model(spec, dataset_path)
        from mmm_framework.reporting.helpers.prefit import sample_prior

        idata = sample_prior(m, 300, 3)
        roi = np.asarray(idata.prior["roi_TV"].values).reshape(-1)
        # sigma=0.2 → a much tighter prior than the default sigma=1
        assert np.std(np.log(roi)) < 0.35


class TestExplicitPriorsFlow:
    def test_media_coefficient_prior_changes_the_graph(self, dataset_path):
        m = build_model(
            _spec(
                {
                    "media": {
                        "TV": {
                            "coefficient": {
                                "distribution": "half_normal",
                                "params": {"sigma": 0.05},
                            }
                        }
                    }
                }
            ),
            dataset_path,
        )
        stats = _prior_stats(m, ["beta_TV", "beta_Digital"])
        op, mean, _ = stats["beta_TV"]
        assert op == "halfnormal_rv"
        assert mean < 0.2  # tight prior actually applied
        # The untouched channel keeps the (agent) default — ROI mode, where
        # beta is a derived Deterministic from roi_Digital.
        assert stats["beta_Digital"][0] == "det"
        assert "roi_Digital" in {rv.name for rv in m.model.free_RVs}

    def test_control_coefficient_prior_changes_the_graph(self, dataset_path):
        m = build_model(
            _spec(
                {
                    "controls": {
                        "Price": {
                            "coefficient": {
                                "distribution": "normal",
                                "params": {"mu": -2.0, "sigma": 0.1},
                            }
                        }
                    }
                }
            ),
            dataset_path,
        )
        stats = _prior_stats(m, ["beta_controls", "beta_control_Price"])
        # beta_controls survives as a Deterministic with the same name/shape.
        assert stats["beta_controls"][0] == "det"
        assert stats["beta_controls"][1] == pytest.approx(-2.0, abs=0.1)
        assert stats["beta_control_Price"][0] == "normal_rv"

    def test_control_allow_negative_false_is_positive_only(self, dataset_path):
        m = build_model(
            _spec({"controls": {"Price": {"allow_negative": False}}}), dataset_path
        )
        stats = _prior_stats(m, ["beta_control_Price"])
        op, mean, _ = stats["beta_control_Price"]
        assert op == "halfnormal_rv"
        assert mean > 0

    def test_adstock_and_saturation_priors_still_flow(self, dataset_path):
        # Regression guard for the paths that already worked.
        m = build_model(
            _spec(
                {
                    "media": {
                        "TV": {
                            "adstock_alpha": {
                                "distribution": "beta",
                                "params": {"alpha": 30, "beta": 1},
                            },
                            "saturation_kappa": {
                                "distribution": "beta",
                                "params": {"alpha": 50, "beta": 1},
                            },
                        }
                    }
                }
            ),
            dataset_path,
        )
        stats = _prior_stats(m, ["adstock_alpha_TV", "sat_half_TV"])
        assert stats["adstock_alpha_TV"][1] > 0.9  # Beta(30,1) mass near 1
        assert stats["sat_half_TV"][1] > 0.9

    def test_prefit_prior_estimands_respond_to_coefficient_prior(self, dataset_path):
        """The user-visible symptom: the prefit readout's prior ROI must move
        when the agent tightens a channel's coefficient prior."""
        from mmm_framework.reporting.helpers.prefit import (
            prior_estimand_facts,
            sample_prior,
        )

        def tv_prior_roi(spec) -> float:
            m = build_model(spec, dataset_path)
            idata = sample_prior(m, 200, 7)
            est = prior_estimand_facts(m, idata)
            row = next(r for r in est["channels"] if r["channel"] == "TV")
            return row["mean"]

        default_roi = tv_prior_roi(_spec())
        tight_roi = tv_prior_roi(
            _spec(
                {
                    "media": {
                        "TV": {
                            "coefficient": {
                                "distribution": "half_normal",
                                "params": {"sigma": 0.01},
                            }
                        }
                    }
                }
            )
        )
        assert (
            tight_roi < default_roi / 5
        )  # near-zero effect prior ⇒ near-zero prior ROI
