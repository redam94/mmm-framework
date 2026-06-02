"""Tests for the equifinality guardrails (P1-4, critique.md §3.6).

Covers the data-anchored Hill ``kappa`` mechanism: the percentile helper and the
opt-in bounded prior. The documentation pieces (adstock docstring, model module
docstring, technical note) are not asserted here.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from mmm_framework.config import SaturationConfig
from mmm_framework.mmm_extensions.components.priors import create_saturation_prior


class TestKappaBoundsFromData:
    def test_returns_requested_percentiles(self):
        x = np.arange(0, 101, dtype=float)  # 0..100
        lo, hi = SaturationConfig.compute_kappa_bounds_from_data(x, (0.1, 0.9))
        assert lo == pytest.approx(10.0, abs=1e-6)
        assert hi == pytest.approx(90.0, abs=1e-6)
        assert hi > lo

    def test_ignores_nans(self):
        x = np.array([0.0, np.nan, 5.0, 10.0, np.nan])
        lo, hi = SaturationConfig.compute_kappa_bounds_from_data(x, (0.0, 1.0))
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(10.0)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SaturationConfig.compute_kappa_bounds_from_data(np.array([]))

    def test_degenerate_range_raises(self):
        with pytest.raises(ValueError, match="no spread"):
            SaturationConfig.compute_kappa_bounds_from_data(np.zeros(50))

    def test_invalid_percentiles_raise(self):
        with pytest.raises(ValueError, match="percentiles"):
            SaturationConfig.compute_kappa_bounds_from_data(np.arange(10.0), (0.9, 0.1))

    def test_config_exposes_default_percentiles(self):
        cfg = SaturationConfig.hill()
        assert cfg.kappa_bounds_percentiles == (0.1, 0.9)


class TestAnchoredKappaPrior:
    def test_default_kappa_is_beta(self):
        with pm.Model():
            params = create_saturation_prior("sat_TV", "hill")
        assert type(params["kappa"].owner.op).__name__ == "BetaRV"

    def test_bounded_kappa_is_uniform_within_range(self):
        with pm.Model():
            params = create_saturation_prior(
                "sat_TV", "hill", kappa_lower=2.0, kappa_upper=8.0
            )
            draws = pm.draw(params["kappa"], draws=500, random_seed=0)
        assert draws.min() >= 2.0
        assert draws.max() <= 8.0

    def test_inverted_bounds_raise(self):
        with pm.Model():
            with pytest.raises(ValueError, match="kappa_upper must exceed"):
                create_saturation_prior(
                    "sat_TV", "hill", kappa_lower=8.0, kappa_upper=2.0
                )

    def test_logistic_path_unaffected_by_bounds(self):
        with pm.Model():
            params = create_saturation_prior(
                "sat_TV", "logistic", kappa_lower=2.0, kappa_upper=8.0
            )
        assert "lam" in params
        assert "kappa" not in params


class TestAnchoredKappaWiredThroughBuilder:
    """The bounds reach the Hill prior through build_media_transforms via the
    saturation_config's prior_params (the supported, opt-in path)."""

    def test_bounds_flow_through_build_media_transforms(self):
        import pytensor.tensor as pt
        from mmm_framework.config import SaturationConfig
        from mmm_framework.mmm_extensions.components.builders import (
            build_media_transforms,
        )

        rng = np.random.default_rng(0)
        spend = np.abs(rng.normal(100, 30, size=(40, 1)))
        lo, hi = SaturationConfig.compute_kappa_bounds_from_data(
            spend[:, 0], (0.1, 0.9)
        )

        with pm.Model(coords={"obs": range(40), "channel": ["tv"]}):
            result = build_media_transforms(
                X_media=pt.as_tensor_variable(spend),
                channel_names=["tv"],
                adstock_config={"l_max": 4, "prior_type": "beta"},
                saturation_config={
                    "type": "hill",
                    "prior_params": {"kappa_lower": lo, "kappa_upper": hi},
                },
                share_params=False,
            )
            kappa = result.saturation_params["tv"]["kappa"]
            draws = pm.draw(kappa, draws=300, random_seed=0)
        # Data-anchored kappa is bounded to the observed-spend percentile range.
        assert draws.min() >= lo
        assert draws.max() <= hi
        assert hi > lo
