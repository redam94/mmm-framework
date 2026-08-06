"""Decomposition closure — the residual is disclosed, not absorbed (issue #220).

The defect these cover: several surfaces defined a baseline as ``observed -
modelled media``, so the model's residual disappeared into a bar labelled "base
demand" and the bridge closed because it was defined to close.

Two things are graded here that are easy to conflate:

* **that the bridge closes.** Cheap, and nearly vacuous on its own — a bridge
  built from a leftover line always closes.
* **that the closure does not read as a validation.** This is the one that
  matters. On the ``make_clean`` MAP fit below the disclosed residual is
  ``-67.3`` while the modelled baseline's actual error against planted truth is
  in the hundreds, so a reader who takes the small residual as evidence the
  baseline is right is off by more than an order of magnitude. The tests pin the
  epistemics (the interval, the collapse notice, the caveat sentence) alongside
  the arithmetic for that reason.

Numbers in the assertions were measured on the fits built here, not chosen.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig, TrendType
from mmm_framework.config import ModelSpecification
from mmm_framework.finance.closure import (
    MATERIAL_UNEXPLAINED_PCT,
    ClosureFacts,
    MediaReconciliation,
    decomposition_closure,
    fitted_total,
)
from mmm_framework.finance.lines import (
    ABSORBING,
    MODELLED,
    OBSERVED,
    RESIDUAL,
    BridgeLine,
    LineProvenance,
    absorbs_residual,
    bridge_gap,
    provenance_of,
)
from mmm_framework.synth import dgp

N_WEEKS = 104


def _fit(*, multiplicative: bool = False, method: str = "map", seed: int = 0):
    world = dgp.make_clean(seed=seed, n_weeks=N_WEEKS)
    b = ModelConfigBuilder()
    if method == "map":
        b = b.map_fit()
    else:
        b = b.bayesian_numpyro().with_chains(2).with_draws(400).with_tune(400)
    cfg = b.build()
    if multiplicative:
        cfg = cfg.model_copy(
            update={"specification": ModelSpecification.MULTIPLICATIVE}
        )
    model = BayesianMMM(world.panel(), cfg, TrendConfig(type=TrendType.LINEAR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            model.fit(random_seed=seed)
    return model


@pytest.fixture(scope="module")
def additive_map():
    return _fit()


@pytest.fixture(scope="module")
def multiplicative_map():
    return _fit(multiplicative=True)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


class TestLineProvenance:
    def test_absence_reads_as_absorbing(self):
        """The pessimistic default. Every line written before this module
        existed was a leftover, and promoting an unstated one to "modelled"
        would launder exactly the numbers this issue is about."""
        assert provenance_of(None) is ABSORBING
        assert provenance_of({}) is ABSORBING
        assert provenance_of("not a provenance") is ABSORBING
        assert BridgeLine("Base", 1.0).absorbs_residual

    def test_known_values_round_trip(self):
        for p in LineProvenance:
            assert provenance_of(p.value) is p
            assert provenance_of(p) is p
        assert provenance_of({"provenance": "residual"}) is RESIDUAL

    def test_only_absorbing_absorbs(self):
        assert absorbs_residual(BridgeLine("x", 1.0, ABSORBING))
        for p in (MODELLED, OBSERVED, RESIDUAL):
            assert not absorbs_residual(BridgeLine("x", 1.0, p))

    def test_an_interval_must_state_its_mass(self):
        """#277's rule, applied here rather than reintroduced."""
        with pytest.raises(ValueError, match="must state its mass"):
            BridgeLine("Base", 10.0, MODELLED, lower=9.0, upper=11.0)

    def test_half_an_interval_is_refused(self):
        with pytest.raises(ValueError, match="needs both bounds"):
            BridgeLine("Base", 10.0, MODELLED, lower=9.0, interval_mass=0.9)

    def test_describe_names_the_leftover(self):
        line = BridgeLine("Base", 10.0, ABSORBING)
        assert "carries the model residual" in line.describe()
        assert (
            BridgeLine("Base", 10.0, MODELLED, basis="components").to_dict()[
                "absorbs_residual"
            ]
            is False
        )


class TestBridgeGap:
    def test_closing_bridge(self):
        lines = [
            BridgeLine("Media", 10.0, MODELLED),
            BridgeLine("Base", 85.0, MODELLED),
            BridgeLine("Unexplained", 5.0, RESIDUAL),
        ]
        gap = bridge_gap(lines, 100.0)
        assert gap.closes and abs(gap.gap) < 1e-9
        assert gap.absorbing_lines == []

    def test_a_closing_bridge_is_not_a_clean_bill_of_health(self):
        """An absorbing line makes the bridge close by construction. The gap
        object says so rather than reporting a satisfied zero and stopping."""
        lines = [
            BridgeLine("Media", 10.0, MODELLED),
            BridgeLine("Base", 90.0, ABSORBING),
        ]
        gap = bridge_gap(lines, 100.0)
        assert gap.closes
        assert gap.absorbing_lines == ["Base"]
        assert "understates" in gap.describe()

    def test_open_bridge_reports_its_gap(self):
        gap = bridge_gap([BridgeLine("Media", 10.0, MODELLED)], 100.0)
        assert not gap.closes
        assert gap.gap == pytest.approx(90.0)
        assert gap.gap_pct == pytest.approx(0.9)
        assert "does not close" in gap.describe()

    def test_zero_target_does_not_demand_bit_equality(self):
        assert bridge_gap([BridgeLine("x", 0.0, MODELLED)], 0.0).closes


# ---------------------------------------------------------------------------
# The closure itself
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAdditiveClosure:
    def test_identity_holds(self, additive_map):
        """``sum(components) + unexplained == observed``, to 1e-6 of observed."""
        f = decomposition_closure(additive_map)
        assert f.closure_available and f.basis == "components"
        recovered = f.fitted_total + f.unexplained
        assert abs(recovered - f.observed_total) <= abs(f.observed_total) * 1e-6
        assert f.closes

    def test_lines_are_the_addends_not_the_target(self, additive_map):
        """The observed total is what the lines sum TO. Including it as a line
        would double it, and a consumer summing the list would be silently
        wrong."""
        f = decomposition_closure(additive_map)
        assert [ln.name for ln in f.lines] == [
            "Media",
            "Base (non-marketing)",
            "Unexplained",
        ]
        assert bridge_gap(f.lines, f.observed_total).closes

    def test_no_line_absorbs_the_residual(self, additive_map):
        f = decomposition_closure(additive_map)
        assert not any(ln.absorbs_residual for ln in f.lines)
        assert [ln.provenance for ln in f.lines] == [MODELLED, MODELLED, RESIDUAL]

    def test_the_residual_is_small_and_that_proves_nothing(self, additive_map):
        """Measured: -67.3 on 46801.1 observed, i.e. -0.144%. Below the
        materiality bar, and deliberately NOT asserted to be evidence of
        anything — the caveat sentence carries that."""
        f = decomposition_closure(additive_map)
        assert abs(f.unexplained_pct) < MATERIAL_UNEXPLAINED_PCT
        assert not f.is_material
        reading = f.residual_reading()
        assert "accounting property" in reading
        assert "not a validation" in reading

    def test_map_interval_collapses_and_says_so(self, additive_map):
        """A single-draw posterior has no spread. Reporting [x, x] as a 90%
        interval is the visual language of an extremely precise estimate, which
        is the opposite of what a MAP fit means (#249)."""
        f = decomposition_closure(additive_map)
        assert f.baseline_interval_basis == "collapsed"
        assert f.baseline_lower is None and f.baseline_upper is None
        assert f.interval_mass is None
        assert any("collapsed onto the point estimate" in w for w in f.warnings)

    def test_media_sources_agree_on_the_core_model(self, additive_map):
        f = decomposition_closure(additive_map)
        assert isinstance(f.reconciliation, MediaReconciliation)
        assert f.reconciliation.agrees
        assert f.reconciliation.ratio == pytest.approx(1.0, abs=0.05)

    def test_fitted_total_is_the_shared_definition(self, additive_map):
        """The CFO rollup routes through this, so the two cannot disagree about
        what "fitted" means."""
        from mmm_framework.reporting.helpers.cfo import _fitted_total

        total, basis, _media = fitted_total(additive_map)
        assert basis == "components"
        assert _fitted_total(additive_map) == pytest.approx(total)


@pytest.mark.slow
class TestNutsClosure:
    def test_the_baseline_interval_dwarfs_the_residual(self):
        """The point of pairing them. Measured on this fit the residual is
        ~84 and the 90% baseline interval is ~2700 wide — the residual is a
        rounding error against the uncertainty a reader should actually weigh,
        which is why a near-zero residual must never be rendered alone."""
        model = _fit(method="nuts")
        f = decomposition_closure(model)
        assert f.baseline_interval_basis == "media_only"
        assert f.interval_mass == 0.90
        assert f.baseline_lower < f.baseline_total < f.baseline_upper
        assert f.baseline_interval_width > 10 * abs(f.unexplained)
        assert "propagates media uncertainty only" in f.lines[1].note


@pytest.mark.slow
class TestMultiplicativeClosure:
    def test_identity_holds_on_the_log_scale_model(self, multiplicative_map):
        f = decomposition_closure(multiplicative_map)
        assert f.specification == "multiplicative"
        assert f.closure_available and f.basis == "components"
        recovered = f.fitted_total + f.unexplained
        assert abs(recovered - f.observed_total) <= abs(f.observed_total) * 1e-6

    def test_media_comes_from_lmdi_not_from_refused_draws(self, multiplicative_map):
        """``sample_channel_contributions`` refuses here by design — its
        additivity premise holds on the log scale. The closure takes the LMDI
        total instead of propagating the refusal as a missing number."""
        with pytest.raises(NotImplementedError):
            multiplicative_map.sample_channel_contributions()
        f = decomposition_closure(multiplicative_map)
        assert f.media_total is not None and f.media_total > 0
        assert f.lines[0].basis == "LMDI decomposition"
        assert f.reconciliation is None

    def test_the_jensen_gap_is_disclosed_not_folded_in(self, multiplicative_map):
        """The additive reconstruction of a log-scale model and the mean of its
        exponentiated predictions are different quantities. Measured: -281.7.
        Folding that into a component would be the same class of defect this
        issue is about."""
        f = decomposition_closure(multiplicative_map)
        assert f.jensen_gap is not None
        assert abs(f.jensen_gap) > 1.0
        assert any("Jensen gap" in w for w in f.warnings)


class TestUnavailableClosure:
    """A model that cannot state a fitted total must say so, not guess."""

    class NoFittedTotal:
        y_raw = np.full(10, 500.0)
        X_media_raw = np.full((10, 2), 100.0)
        channel_names = ["TV", "Search"]

        def sample_channel_contributions(self, X_media=None, **kw):
            X = self.X_media_raw if X_media is None else X_media
            return np.sqrt(X)[None, :, :] * np.ones((4, 1, 1))

    def test_reports_unavailable_and_labels_the_leftover(self):
        f = decomposition_closure(self.NoFittedTotal())
        assert not f.closure_available
        assert f.fitted_total is None and f.unexplained is None
        assert f.basis == "unavailable"
        assert not f.closes
        leftover = [ln for ln in f.lines if ln.absorbs_residual]
        assert len(leftover) == 1
        assert "carries the model residual" in leftover[0].note
        assert "absorbs it" in f.residual_reading()

    def test_no_observed_total_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError, match="observed total"):
            decomposition_closure(object())

    def test_payload_is_json_shaped(self):
        payload = decomposition_closure(self.NoFittedTotal()).to_dict()
        assert payload["closure_available"] is False
        assert isinstance(payload["lines"], list)
        assert payload["lines"][0]["provenance"] in {p.value for p in LineProvenance}
        assert isinstance(payload["residual_reading"], str)


class TestMaterialityThreshold:
    def _facts(self, pct: float) -> ClosureFacts:
        observed = 1000.0
        unexplained = observed * pct
        return ClosureFacts(
            observed_total=observed,
            fitted_total=observed - unexplained,
            unexplained=unexplained,
            unexplained_pct=pct,
            basis="components",
            closure_available=True,
            specification="additive",
        )

    def test_half_a_percent_is_the_bar(self):
        assert not self._facts(0.004).is_material
        assert self._facts(0.005).is_material
        assert self._facts(-0.015).is_material  # heavy-tailed world, measured

    def test_unknown_residual_is_not_material(self):
        f = decomposition_closure(TestUnavailableClosure.NoFittedTotal())
        assert not f.is_material


# ---------------------------------------------------------------------------
# The reconciliation guard — the trap this milestone is really about
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExtensionReconciliation:
    """A bridge can close perfectly around a media number that is 10x wrong.

    Measured on this NestedMMM: the reporting extractor's media total is
    ``2108.8`` while the model's own ``sample_channel_contributions`` gives
    ``22634.2``, against a planted truth of ``19591.7``. The closure closes to
    ``-0.005%`` either way. Closing is therefore not the check; disagreeing is.
    """

    @staticmethod
    def _nested():
        from mmm_framework.mmm_extensions.config import (
            MediatorConfig,
            MediatorType,
            NestedModelConfig,
        )
        from mmm_framework.mmm_extensions.models.nested import NestedMMM

        n = 60
        rng = np.random.default_rng(0)
        media = np.abs(rng.normal(100, 20, (n, 2)))
        idx = pd.date_range("2022-01-03", periods=n, freq="W-MON")
        r = np.random.default_rng(1)
        aware = 40 + 0.3 * media[:, 0] + r.normal(0, 4, n)
        sales = 1000 + 4 * aware + 2 * media[:, 1] + r.normal(0, 40, n)
        cfg = NestedModelConfig(
            mediators=(
                MediatorConfig(
                    name="Awareness", mediator_type=MediatorType.FULLY_LATENT
                ),
            )
        )
        model = NestedMMM(media, sales, ["TV", "Digital"], cfg, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                results = model.fit(method="map", random_seed=0)
        # The planted media-attributable total: TV moves sales only through
        # Awareness (0.3 per unit, worth 4 each), Digital moves it directly.
        planted_media = float((4 * 0.3 * media[:, 0]).sum() + (2 * media[:, 1]).sum())
        return model, results, planted_media

    @staticmethod
    def _extractor_media(model, results) -> float:
        from mmm_framework.reporting.extractors.extended import ExtendedMMMExtractor

        totals = ExtendedMMMExtractor(model, results).extract().component_totals or {}
        return sum(
            v
            for k, v in totals.items()
            if k.startswith("Via ") or k.endswith("(direct)")
        )

    def test_extension_reads_the_mu_deterministic(self):
        """Core BayesianMMM registers no ``mu``; the extension families do, and
        that is the branch this exercises."""
        model, _, _ = self._nested()
        f = decomposition_closure(model)
        assert f.basis == "mu"
        assert f.closure_available and f.closes

    def test_the_divergence_is_flagged_not_averaged(self):
        model, results, _ = self._nested()
        f = decomposition_closure(
            model, decomposition_media=self._extractor_media(model, results)
        )
        assert f.reconciliation is not None
        assert not f.reconciliation.agrees
        assert f.reconciliation.ratio > 5.0
        assert any("disagrees between sources" in w for w in f.warnings)

    def test_the_published_total_is_the_one_nearer_the_truth(self):
        """Grades both readings against the planted media effect, because
        "they disagree" is only half the finding — a guard that flagged the
        divergence and then published the worse number would pass every other
        test in this class.

        Measured against a planted 19591.7: the extractor total is 2108.8 (off
        by 17482.9) and the contribution total is 22634.2 (off by 3042.5). The
        contribution reading is wrong too, by ~15%; it is simply not wrong by
        an order of magnitude, and it is the one that gets published.
        """
        model, results, planted = self._nested()
        extractor_media = self._extractor_media(model, results)
        f = decomposition_closure(model, decomposition_media=extractor_media)

        assert planted == pytest.approx(19591.7, abs=1.0)
        published_err = abs(f.media_total - planted)
        extractor_err = abs(extractor_media - planted)
        assert published_err < extractor_err
        assert extractor_err > 5 * published_err
        # And the published number is not thereby endorsed.
        assert published_err / planted > 0.10

    def test_the_suspect_total_does_not_get_published(self):
        """An injected total arrives because the model could not produce it,
        which makes it the number under suspicion rather than the number to
        trust. It must not silently replace the published media line."""
        model, _, _ = self._nested()
        f = decomposition_closure(model, decomposition_media=2108.8)
        assert f.media_total != pytest.approx(2108.8)
        assert f.media_total > 10000.0
        assert f.lines[0].value == pytest.approx(f.media_total)

    def test_closing_is_not_the_check(self):
        """Both readings close. Stated as a test so a future change that grades
        only closure gets caught."""
        model, _, _ = self._nested()
        f = decomposition_closure(model, decomposition_media=2108.8)
        assert f.closes
        assert abs(f.unexplained_pct) < 0.001
        assert not f.reconciliation.agrees
