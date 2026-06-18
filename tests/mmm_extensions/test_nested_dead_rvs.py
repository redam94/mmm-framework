"""Regression tests for dead (never-used) RVs in NestedMMM.

Bug: when a mediator was mapped to >= 2 channels, ``NestedMMM`` created an
aggregate ``beta_media_to_<mediator>`` RV *and then* per-channel
``beta_<channel>_to_<mediator>`` RVs that actually drive the mediator,
leaving the aggregate RV in the graph as a dead, never-informed parameter
(pure prior noise in posteriors and parameter-learning tables).

These tests assert:
- the aggregate RV exists only on the single-channel path where it is used,
- the per-channel RVs exist only on the multi-channel path,
- no free RV in the graph is disconnected from the likelihood (graph
  ancestry of the observed RVs covers every free RV).
"""

from __future__ import annotations

import numpy as np
import pytest
from pytensor.graph.traversal import ancestors

from mmm_framework.mmm_extensions import NestedMMM
from mmm_framework.mmm_extensions.builders import (
    MediatorConfigBuilder,
    NestedModelConfigBuilder,
)

# =============================================================================
# Fixtures
# =============================================================================

CHANNELS = ["tv", "digital", "social"]
MEDIATOR = "brand_awareness"


@pytest.fixture
def media_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.abs(rng.normal(loc=100, scale=50, size=(52, len(CHANNELS))))


@pytest.fixture
def outcome() -> np.ndarray:
    rng = np.random.default_rng(7)
    return 1000 + rng.normal(scale=100, size=52)


def _make_config(mapped_channels: list[str]):
    """Nested config with one mediator mapped to ``mapped_channels``."""
    mediator = (
        MediatorConfigBuilder(MEDIATOR)
        .partially_observed(observation_noise=0.15)
        .with_positive_media_effect(sigma=1.0)
        .build()
    )
    return (
        NestedModelConfigBuilder()
        .add_mediator(mediator)
        .map_channels_to_mediator(MEDIATOR, mapped_channels)
        .build()
    )


def _free_rv_names(pymc_model) -> set[str]:
    return {rv.name for rv in pymc_model.free_RVs}


def _dead_free_rvs(pymc_model) -> set[str]:
    """Free RVs that are NOT ancestors of any observed RV (dead parameters)."""
    reachable = set(ancestors(pymc_model.observed_RVs))
    return {rv.name for rv in pymc_model.free_RVs if rv not in reachable}


# =============================================================================
# Multi-channel mediator: per-channel betas only, no aggregate RV
# =============================================================================


class TestMultiChannelMediator:
    """Mediator mapped to two channels -> per-channel betas, no aggregate."""

    @pytest.fixture
    def pymc_model(self, media_data, outcome):
        model = NestedMMM(
            X_media=media_data,
            y=outcome,
            channel_names=CHANNELS,
            config=_make_config(["tv", "digital"]),
        )
        return model.model

    def test_aggregate_beta_absent(self, pymc_model):
        assert f"beta_media_to_{MEDIATOR}" not in pymc_model.named_vars
        assert f"beta_media_to_{MEDIATOR}" not in _free_rv_names(pymc_model)

    def test_per_channel_betas_present(self, pymc_model):
        names = _free_rv_names(pymc_model)
        assert f"beta_tv_to_{MEDIATOR}" in names
        assert f"beta_digital_to_{MEDIATOR}" in names
        # Unmapped channel gets no media->mediator beta
        assert f"beta_social_to_{MEDIATOR}" not in pymc_model.named_vars

    def test_no_dead_free_rvs(self, pymc_model):
        assert _dead_free_rvs(pymc_model) == set()


# =============================================================================
# Single-channel mediator: aggregate beta only
# =============================================================================


class TestSingleChannelMediator:
    """Mediator mapped to exactly one channel -> aggregate beta is the path."""

    @pytest.fixture
    def pymc_model(self, media_data, outcome):
        model = NestedMMM(
            X_media=media_data,
            y=outcome,
            channel_names=CHANNELS,
            config=_make_config(["tv"]),
        )
        return model.model

    def test_aggregate_beta_present(self, pymc_model):
        assert f"beta_media_to_{MEDIATOR}" in _free_rv_names(pymc_model)

    def test_per_channel_betas_absent(self, pymc_model):
        for channel in CHANNELS:
            assert f"beta_{channel}_to_{MEDIATOR}" not in pymc_model.named_vars

    def test_no_dead_free_rvs(self, pymc_model):
        assert _dead_free_rvs(pymc_model) == set()


# =============================================================================
# Tiny seeded fit + parameter learning excludes the dead RV
# =============================================================================


class TestFitAndParameterLearning:
    """A tiny fit works and the learning table has no dead aggregate beta."""

    def test_tiny_fit_and_learning_table(self, media_data, outcome):
        model = NestedMMM(
            X_media=media_data,
            y=outcome,
            channel_names=CHANNELS,
            config=_make_config(["tv", "digital"]),
        )
        model.fit(
            draws=50,
            tune=50,
            chains=1,
            cores=1,
            random_seed=0,
            progressbar=False,
        )

        table = model.compute_parameter_learning(prior_samples=100, random_seed=0)

        params = set(table["parameter"].astype(str))
        base_params = {p.split("[")[0] for p in params}
        assert f"beta_media_to_{MEDIATOR}" not in base_params
        assert f"beta_tv_to_{MEDIATOR}" in base_params
        assert f"beta_digital_to_{MEDIATOR}" in base_params
