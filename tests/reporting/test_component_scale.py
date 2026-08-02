"""Component Deterministics have one stated scale convention (#274).

The two model families register the same Deterministic names on different
scales — the core graph standardized, the extension graphs already multiplied
by ``y_std`` — and nothing said so. ``prior_component_facts`` applied the core
rule to both, so a nested model's Model Design Readout rendered a band scaled
by ``y_std**2``.

Measured on the 60-week nested model below (``y_std = 68.8``, KPI topping out
at 1,646): the seasonality band reached **3,940** before the fix — 2.4x the
entire KPI, which reads as "the prior is uninformative", the opposite of what a
pre-registration document is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import ModelConfig, SeasonalityConfig
from mmm_framework.model import TrendConfig, TrendType
from mmm_framework.model.component_scale import (
    COMPONENT_DETERMINISTICS,
    ComponentScale,
    component_scale,
    to_kpi_units,
)
from mmm_framework.reporting.helpers.prefit import (
    prior_component_facts,
    sample_prior,
)


# --------------------------------------------------------------------------- #
# the convention itself
# --------------------------------------------------------------------------- #
class TestConventionDeclaration:
    def test_core_declares_standardized(self):
        from mmm_framework.model.base import BayesianMMM

        assert BayesianMMM.COMPONENT_DETERMINISTIC_SCALE is ComponentScale.STANDARDIZED

    def test_extensions_declare_original(self):
        from mmm_framework.mmm_extensions.models.base import BaseExtendedMMM
        from mmm_framework.mmm_extensions.models.combined import CombinedMMM
        from mmm_framework.mmm_extensions.models.multivariate import MultivariateMMM
        from mmm_framework.mmm_extensions.models.nested import NestedMMM
        from mmm_framework.mmm_extensions.models.structural import (
            StructuralNestedMMM,
        )

        for cls in (
            BaseExtendedMMM,
            NestedMMM,
            MultivariateMMM,
            CombinedMMM,
            StructuralNestedMMM,
        ):
            assert cls.COMPONENT_DETERMINISTIC_SCALE is ComponentScale.ORIGINAL, cls

    def test_undeclared_model_is_treated_as_standardized(self):
        """The historical assumption, so duck-typed stubs keep working."""

        class Stub:
            y_std = 10.0

        assert component_scale(Stub()) is ComponentScale.STANDARDIZED
        np.testing.assert_allclose(to_kpi_units(np.ones(3), Stub()), 10.0)

    def test_nonsense_declaration_falls_back(self):
        class Stub:
            y_std = 10.0
            COMPONENT_DETERMINISTIC_SCALE = "not-a-scale"

        assert component_scale(Stub()) is ComponentScale.STANDARDIZED

    def test_original_scale_is_a_no_op(self):
        class Stub:
            y_std = 10.0
            COMPONENT_DETERMINISTIC_SCALE = ComponentScale.ORIGINAL

        np.testing.assert_array_equal(to_kpi_units(np.ones(3), Stub()), np.ones(3))

    def test_governed_names_are_the_ones_both_families_register(self):
        assert set(COMPONENT_DETERMINISTICS) == {
            "trend_component",
            "seasonality_component",
            "controls_total",
            "media_total",
            "channel_contributions",
        }


# --------------------------------------------------------------------------- #
# the readout
# --------------------------------------------------------------------------- #
@pytest.fixture
def nested_model():
    from mmm_framework.mmm_extensions.config import (
        MediatorConfig,
        MediatorType,
        NestedModelConfig,
    )
    from mmm_framework.mmm_extensions.models.nested import NestedMMM

    idx = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    rng = np.random.default_rng(0)
    media = np.abs(rng.normal(100, 20, (60, 2)))
    aware = 40 + 0.3 * media[:, 0] + rng.normal(0, 4, 60)
    y = 1000 + 4 * aware + 2 * media[:, 1] + rng.normal(0, 40, 60)

    mc = ModelConfig()
    mc.seasonality = SeasonalityConfig(yearly=2)
    return NestedMMM(
        media,
        y,
        ["TV", "Digital"],
        NestedModelConfig(
            mediators=(
                MediatorConfig(
                    name="Awareness", mediator_type=MediatorType.FULLY_LATENT
                ),
            )
        ),
        index=idx,
        model_config=mc,
        trend_config=TrendConfig(type=TrendType.LINEAR),
    )


class TestExtensionReadoutIsInKpiUnits:
    def test_bands_match_the_graphs_own_registered_values(self, nested_model):
        """The regression: pre-fix these came out y_std times too large."""
        m = nested_model
        idata = sample_prior(m, n_samples=120, random_seed=0)
        facts = prior_component_facts(m, idata)

        assert {"trend", "seasonality"} <= set(facts)
        for key, var in (
            ("trend", "trend_component"),
            ("seasonality", "seasonality_component"),
        ):
            raw = np.asarray(idata.prior[var].values, dtype=float).reshape(-1, m.n_obs)
            # National panel: one obs per period, so the helper's per-period sum
            # is the identity and the band is the raw quantile.
            np.testing.assert_allclose(
                facts[key]["bands"]["median"],
                np.percentile(raw, 50, axis=0),
                rtol=1e-9,
                atol=1e-9,
            )

    def test_bands_are_plausible_against_the_kpi(self, nested_model):
        """A band wider than the KPI itself is the symptom the fix removes."""
        m = nested_model
        idata = sample_prior(m, n_samples=120, random_seed=0)
        facts = prior_component_facts(m, idata)

        kpi_max = float(np.abs(np.asarray(m.y, dtype=float)).max())
        for key in ("trend", "seasonality"):
            upper = float(np.abs(facts[key]["bands"]["upper"]).max())
            assert upper < kpi_max, (
                f"{key} prior band ({upper:.1f}) exceeds the whole KPI "
                f"({kpi_max:.1f}) — the y_std**2 symptom"
            )
            # Non-vacuity: the pre-fix value WAS above the bar, so the assertion
            # above is a real gate rather than a trivially-satisfied one.
            assert upper * m.y_std > kpi_max


class TestChannelContributionsConsumers:
    """The bug is per-variable, not per-helper (#274 acceptance criterion 3).

    `channel_contributions` is governed by the same convention, and its two
    consumers hard-coded the core rule — one in the very same readout, one on
    the default post-fit ROI path.
    """

    def test_prior_estimand_facts_matches_the_graphs_own_contributions(
        self, nested_model
    ):
        from mmm_framework.reporting.helpers.prefit import prior_estimand_facts

        m = nested_model
        idata = sample_prior(m, n_samples=120, random_seed=0)
        facts = prior_estimand_facts(m, idata)
        assert facts.get("channels"), "no prior estimand rows"

        raw = np.asarray(idata.prior["channel_contributions"].values, dtype=float)
        raw = raw.reshape(-1, *raw.shape[-2:]).sum(axis=1)  # (S, channel)

        from mmm_framework.reporting.helpers.measurement import (
            resolve_channel_divisor,
        )

        by_name = {r["channel"]: r for r in facts["channels"]}
        for c, ch in enumerate(m.channel_names):
            if ch not in by_name:
                continue
            div = resolve_channel_divisor(m, ch)
            want = float(np.mean(raw[:, c] / div.total))
            assert by_name[ch]["mean"] == pytest.approx(want, rel=1e-9)

    @pytest.mark.slow
    def test_post_fit_roi_matches_the_graphs_own_contributions(self, nested_model):
        """Measured pre-fix: contribution 797,050 on a total KPI of 88,685."""
        from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

        m = nested_model
        m.fit(method="map", random_seed=0, progressbar=False)

        raw = np.asarray(
            m._trace.posterior["channel_contributions"].values, dtype=float
        )
        raw = raw.reshape(-1, *raw.shape[-2:]).sum(axis=1).mean(axis=0)

        df = compute_roi_with_uncertainty(m, hdi_prob=0.94)
        got = dict(zip(df["channel"], df["contribution_mean"]))
        for c, ch in enumerate(m.channel_names):
            assert got[ch] == pytest.approx(float(raw[c]), rel=1e-9)

        total_kpi = float(np.asarray(m.y, dtype=float).sum())
        for ch, contrib in got.items():
            assert abs(contrib) < total_kpi, (
                f"{ch} contribution {contrib:,.0f} exceeds the whole KPI "
                f"{total_kpi:,.0f} — the y_std inflation"
            )


class TestCoreReadoutIsUnchanged:
    def test_core_components_still_get_the_y_std_bridge(self):
        """The core path must be byte-identical: standardized x y_std."""
        from mmm_framework.config import InferenceMethod
        from mmm_framework.model import BayesianMMM
        from mmm_framework.synth import dgp

        panel = dgp.build("clean", seed=3, n_weeks=60).panel()
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=1,
            n_draws=50,
        )
        m = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
        idata = sample_prior(m, n_samples=60, random_seed=0)
        facts = prior_component_facts(m, idata)

        assert facts, "core model produced no component facts"
        time_idx = np.asarray(m.time_idx, dtype=int)
        n_periods = int(time_idx.max()) + 1
        for key, var in (
            ("trend", "trend_component"),
            ("seasonality", "seasonality_component"),
            ("controls", "controls_total"),
            ("media", "media_total"),
        ):
            if key not in facts:
                continue
            raw = np.asarray(idata.prior[var].values, dtype=float).reshape(
                -1, time_idx.size
            )
            scaled = raw * m.y_std
            per_period = np.zeros((scaled.shape[0], n_periods))
            np.add.at(
                per_period,
                (np.arange(scaled.shape[0])[:, None], time_idx[None, :]),
                scaled,
            )
            np.testing.assert_allclose(
                facts[key]["bands"]["median"],
                np.percentile(per_period, 50, axis=0),
                rtol=1e-9,
                atol=1e-9,
            )
