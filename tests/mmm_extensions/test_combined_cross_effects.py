"""Tests for CombinedMMM cross-effect handling.

Regression tests for the bug where ``CombinedMMM`` sampled a full
unconstrained ``psi`` matrix with duplicate dims ``("outcome", "outcome")``:

- arviz/xarray cannot compute ESS/r-hat over a variable with the same dim
  twice, so PyMC's post-sampling convergence checks (and
  ``compute_parameter_learning``) crashed;
- configured ``CrossEffectConfig``s (e.g. ``.with_cannibalization()``) were
  silently ignored — both off-diagonal directions AND the two meaningless
  diagonal self-effect entries were sampled as dead knobs.

CombinedMMM must now build its cross-effect matrix exactly the way
MultivariateMMM does (``build_cross_effect_matrix``): only configured
source -> target directions get a free RV, everything else is a structural
zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.mmm_extensions.builders import (
    CombinedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    cannibalization_effect,
)
from mmm_framework.mmm_extensions.models import CombinedMMM, MultivariateMMM

CHANNELS = ["tv", "search"]
N_OBS = 60


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def media_data(rng):
    return rng.gamma(2.0, 10.0, size=(N_OBS, len(CHANNELS)))


@pytest.fixture(scope="module")
def outcome_data(rng):
    return {
        "prod_a": 10 + rng.normal(0, 1.0, N_OBS),
        "prod_b": 5 + rng.normal(0, 1.0, N_OBS),
    }


def _combined_config(with_cross_effect: bool = True):
    builder = (
        CombinedModelConfigBuilder()
        .with_awareness_mediator("awareness")
        .map_channels_to_mediator("awareness", ["tv"])
        .with_outcomes("prod_a", "prod_b")
    )
    if with_cross_effect:
        # source=prod_b (index 1) cannibalizes target=prod_a (index 0)
        builder = builder.with_cannibalization("prod_b", "prod_a")
    return builder.build()


def _combined_model(media_data, outcome_data, with_cross_effect: bool = True):
    return CombinedMMM(
        X_media=media_data,
        outcome_data=outcome_data,
        channel_names=CHANNELS,
        config=_combined_config(with_cross_effect),
    )


def _mv_model(media_data, outcome_data):
    cfg = (
        MultivariateModelConfigBuilder()
        .with_outcomes("prod_a", "prod_b")
        .with_cannibalization("prod_b", "prod_a")
        .build()
    )
    return MultivariateMMM(
        X_media=media_data,
        outcome_data=outcome_data,
        channel_names=CHANNELS,
        config=cfg,
    )


# =============================================================================
# Graph structure
# =============================================================================


class TestCombinedCrossEffectGraph:
    def test_configured_cannibalization_creates_signed_rv(
        self, media_data, outcome_data
    ):
        """prod_b (idx 1) -> prod_a (idx 0) cannibalization => psi_1_0_raw."""
        model = _combined_model(media_data, outcome_data)
        free_names = {rv.name for rv in model.model.free_RVs}

        assert "psi_1_0_raw" in free_names
        # The signed matrix is exposed as a Deterministic, not a free RV.
        assert "psi_matrix" in model.model.named_vars
        assert "psi_matrix" not in free_names

    def test_no_full_psi_matrix_rv_and_no_diagonal_knobs(
        self, media_data, outcome_data
    ):
        """No full Normal psi matrix; no diagonal or unconfigured-direction RVs."""
        model = _combined_model(media_data, outcome_data)
        free_names = {rv.name for rv in model.model.free_RVs}

        assert "psi" not in free_names  # the old full-matrix RV
        # Diagonal self-effects must not exist.
        assert not any(n.startswith("psi_0_0") for n in free_names)
        assert not any(n.startswith("psi_1_1") for n in free_names)
        # The unconfigured reverse direction (prod_a -> prod_b) must not exist.
        assert not any(n.startswith("psi_0_1") for n in free_names)
        # Exactly one psi free RV in total.
        psi_rvs = [n for n in free_names if n.startswith("psi")]
        assert psi_rvs == ["psi_1_0_raw"]

    def test_psi_matrix_deterministic_has_no_duplicate_dims(
        self, media_data, outcome_data
    ):
        """psi_matrix must not carry the duplicate ("outcome", "outcome") dims."""
        model = _combined_model(media_data, outcome_data)
        dims = model.model.named_vars_to_dims.get("psi_matrix")
        if dims is not None:
            assert len(set(dims)) == len(dims)

    def test_no_cross_effects_configured_means_zero_psi_rvs(
        self, media_data, outcome_data
    ):
        model = _combined_model(media_data, outcome_data, with_cross_effect=False)
        all_names = set(model.model.named_vars)

        assert not any(n.startswith("psi") for n in all_names)


# =============================================================================
# Orientation: source -> target, identical to MultivariateMMM
# =============================================================================


class TestCrossEffectOrientation:
    @staticmethod
    def _mu_delta_for_psi_bump(model_obj):
        """Bump the psi_1_0_raw free value and return (delta_mu, delta_psi, Y)."""
        pymc_model = model_obj.model
        point = pymc_model.initial_point()
        (key,) = [k for k in point if k.startswith("psi_1_0_raw")]

        mu_value_graph = pymc_model.replace_rvs_by_values([pymc_model["mu"]])[0]
        fn = pymc_model.compile_fn(
            mu_value_graph,
            inputs=pymc_model.value_vars,
            on_unused_input="ignore",
        )
        mu0 = fn(point)

        bumped = dict(point)
        bumped[key] = point[key] + 1.0  # log-space bump of the HalfNormal
        mu1 = fn(bumped)

        # psi = -HalfNormal = -exp(log_value)
        psi0 = -np.exp(point[key])
        psi1 = -np.exp(bumped[key])
        return mu1 - mu0, psi1 - psi0, model_obj.outcome_data

    def test_combined_effect_lands_on_target_column_only(
        self, media_data, outcome_data
    ):
        """psi_1_0 must move mu[:, 0] (target prod_a) by delta_psi * Y[:, 1]."""
        model = _combined_model(media_data, outcome_data)
        delta_mu, delta_psi, outcomes = self._mu_delta_for_psi_bump(model)

        # Source column (prod_b, index 1) untouched.
        np.testing.assert_allclose(delta_mu[:, 1], 0.0, atol=1e-10)
        # Target column moves by delta_psi * Y[:, source].
        np.testing.assert_allclose(
            delta_mu[:, 0], delta_psi * outcomes["prod_b"], rtol=1e-6
        )

    def test_orientation_matches_multivariate(self, media_data, outcome_data):
        """CombinedMMM and MultivariateMMM share the source->target convention."""
        combined = _combined_model(media_data, outcome_data)
        mv = _mv_model(media_data, outcome_data)

        # Identical internal specs from the identical configuration.
        assert [
            (s.source_idx, s.target_idx, s.effect_type)
            for s in combined._cross_effect_specs
        ] == [
            (s.source_idx, s.target_idx, s.effect_type) for s in mv._cross_effect_specs
        ]

        d_combined, dpsi_c, _ = self._mu_delta_for_psi_bump(combined)
        d_mv, dpsi_m, _ = self._mu_delta_for_psi_bump(mv)

        # Same column responds in both models, scaled by the same Y[:, source].
        np.testing.assert_allclose(
            d_combined[:, 0] / dpsi_c, d_mv[:, 0] / dpsi_m, rtol=1e-6
        )
        np.testing.assert_allclose(d_mv[:, 1], 0.0, atol=1e-10)


# =============================================================================
# Builder parity
# =============================================================================


class TestBuilderParity:
    def test_combined_builder_has_add_cross_effect(self):
        effect = cannibalization_effect(source="prod_b", target="prod_a")
        cfg = (
            CombinedModelConfigBuilder()
            .with_awareness_mediator("awareness")
            .with_outcomes("prod_a", "prod_b")
            .add_cross_effect(effect)
            .build()
        )

        assert len(cfg.multivariate.cross_effects) == 1
        ce = cfg.multivariate.cross_effects[0]
        assert ce.source_outcome == "prod_b"
        assert ce.target_outcome == "prod_a"

    def test_add_cross_effect_equivalent_to_with_cannibalization(self):
        via_add = (
            CombinedModelConfigBuilder()
            .with_outcomes("prod_a", "prod_b")
            .add_cross_effect(cannibalization_effect("prod_b", "prod_a"))
            .build()
        )
        via_with = (
            CombinedModelConfigBuilder()
            .with_outcomes("prod_a", "prod_b")
            .with_cannibalization("prod_b", "prod_a")
            .build()
        )

        assert via_add.multivariate.cross_effects == via_with.multivariate.cross_effects


# =============================================================================
# Fitting: convergence checks + parameter learning must not crash
# =============================================================================


@pytest.fixture(scope="module")
def fitted_combined(media_data, outcome_data):
    """Tiny seeded fit WITHOUT compute_convergence_checks=False.

    cores=1 is required on macOS (MvNormal models break under chain
    multiprocessing); chains=2 sequential so r-hat is actually computed.
    """
    model = _combined_model(media_data, outcome_data)
    model.fit(draws=60, tune=60, chains=2, cores=1, random_seed=0)
    return model


class TestCombinedFitDiagnostics:
    def test_fit_runs_with_convergence_checks(self, fitted_combined):
        """Post-sampling ESS/r-hat checks no longer crash on duplicate dims."""
        assert fitted_combined._trace is not None
        assert "psi_1_0_raw" in fitted_combined._trace.posterior

    def test_compute_parameter_learning_returns_dataframe(self, fitted_combined):
        learning = fitted_combined.compute_parameter_learning(
            prior_samples=100, random_seed=0
        )

        assert isinstance(learning, pd.DataFrame)
        assert len(learning) > 0

    def test_cross_effects_summary_orientation_and_sign(self, fitted_combined):
        """Summary reports the configured prod_b -> prod_a effect, negative."""
        summary = fitted_combined.get_cross_effects_summary()

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["source"] == "prod_b"
        assert row["target"] == "prod_a"
        assert row["effect_type"] == "cannibalization"
        # Cannibalization is sign-constrained negative by construction.
        assert row["mean"] < 0
        assert row["hdi_97%"] <= 0
