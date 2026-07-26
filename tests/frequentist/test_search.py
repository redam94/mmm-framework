"""The transform search, graded against planted truth.

#184's acceptance criterion is blunt and correct: *the search is only worth
shipping if it recovers a planted truth*. `synth.dgp.make_clean` is the positive
control — the model's exact generative family — and it plants per-channel
carryover and saturation:

    _ALPHA = {TV: 0.6, Search: 0.2, Social: 0.4, Display: 0.5}
    _LAM   = {TV: 1.6, Search: 1.8, Social: 1.7, Display: 1.5}

The parameterizations line up exactly, which is what makes recovery well posed:
``_geom_adstock`` is a normalized causal FIR kernel at ``l_max=8`` (identical to
``adstock_weights("geometric", 8, alpha, normalize=True)``), ``_logistic_sat`` is
``1 - exp(-lam * x)``, and the DGP normalizes by the channel max *before*
adstocking — the same order the model uses.

Those constants are module-private and are **not** exported through
``Scenario.notes`` or the JSON answer key, so the test imports them directly
rather than pretending the answer key is complete.

Note what is *not* asserted: exact per-channel recovery. Adstock and saturation
are famously equifinal — a long-carryover / weak-saturation fit and a
short-carryover / strong-saturation one are nearly indistinguishable, which
``transforms/adstock.py`` documents for the Bayesian path too. What the search
must do is beat an uninformed choice and land in the right region; claiming more
than that would be the same overconfidence the module's own caveat warns about.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.config.enums import AdstockType, SaturationType
from mmm_framework.frequentist.design import UnsupportedModelError, build_design_matrix
from mmm_framework.frequentist.search import (
    ADSTOCK_BOUNDS,
    HALF_SATURATION_FRACTION,
    SATURATION_BOUNDS,
    search_transforms,
)
from mmm_framework.model.trend_config import TrendConfig, TrendType
from mmm_framework.synth.dgp import _ALPHA, _LAM
from mmm_framework.validation.backtest import _slice_panel_prefix

from test_design_equivalence import CHANNELS, _configure, _panel

ALPHA = {c: {"alpha": 0.55} for c in CHANNELS}
LAM = {c: {"sat_lam": 2.7} for c in CHANNELS}


def _ready(n_periods=60):
    panel = _panel(n_periods=n_periods)
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    return panel


def _search(panel, **kw):
    kw.setdefault("model_config", ModelConfig())
    kw.setdefault("trend_config", TrendConfig(type=TrendType.LINEAR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return search_transforms(panel, **kw)


class TestBoundsCoverEveryFamily:
    def test_adstock_bounds_cover_every_declared_family(self):
        assert set(ADSTOCK_BOUNDS) == set(AdstockType)

    def test_saturation_bounds_cover_every_declared_family(self):
        assert set(SATURATION_BOUNDS) == set(SaturationType)

    def test_logistic_bound_is_derived_from_the_half_saturation_fraction(self):
        """Bounds live in DATA units: the elbow must sit inside observed spend.

        Media is normalized by the channel max, so a half-saturation fraction g
        maps to `lam = ln(2)/g` for `sat(x) = 1 - exp(-lam*x)`. Deriving the
        bound rather than writing it down is what keeps the saturation families
        agreeing about where the curve may bend.
        """
        lo, hi = SATURATION_BOUNDS[SaturationType.LOGISTIC]["sat_lam"]
        g_lo, g_hi = HALF_SATURATION_FRACTION
        assert lo == pytest.approx(np.log(2) / g_hi)
        assert hi == pytest.approx(np.log(2) / g_lo)
        # Comfortably inside the graph's exponent clip (lam * x = 20).
        assert hi <= 20.0


class TestSearchContract:
    def test_returns_every_evaluated_candidate_not_just_the_winner(self):
        panel = _ready()
        res = _search(
            panel,
            budget=6,
            horizon=8,
            max_origins=2,
            seed=1,
        )
        assert len(res.candidates) == 6
        assert res.candidates[0] is res.best
        assert all(
            a.score <= b.score
            for a, b in zip(res.candidates, res.candidates[1:], strict=False)
        )

    def test_records_the_criterion_and_the_caveat(self):
        """The 'predictive skill is not the causal estimand' objection must ride
        in the OUTPUT, not only in the docs."""
        panel = _ready()
        res = _search(
            panel,
            budget=4,
            horizon=8,
            max_origins=2,
            seed=1,
        )
        assert res.criterion == "rolling_origin_mape"
        assert "attribute worse" in res.caveat
        assert "rolling-origin MAPE" in res.caveat

    def test_is_reproducible_for_a_fixed_seed(self):
        panel = _ready()
        kw = dict(
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.LINEAR),
            budget=5,
            horizon=8,
            max_origins=2,
            seed=7,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = search_transforms(panel, **kw)
            b = search_transforms(panel, **kw)
        assert a.best.alpha == b.best.alpha
        assert a.best.lam == b.best.lam
        assert a.best.score == b.best.score

    def test_spline_trend_is_refused_because_it_cannot_extrapolate(self):
        panel = _ready()
        with pytest.raises(UnsupportedModelError, match="does not extrapolate"):
            search_transforms(
                panel,
                model_config=ModelConfig(),
                trend_config=TrendConfig(type=TrendType.SPLINE),
                budget=2,
                horizon=8,
                max_origins=1,
            )

    def test_unknown_objective_is_rejected(self):
        panel = _ready()
        with pytest.raises(ValueError, match="mape/smape/rmse/mae/bias"):
            search_transforms(
                panel,
                model_config=ModelConfig(),
                trend_config=TrendConfig(type=TrendType.LINEAR),
                objective="r_squared",
            )

    def test_too_short_a_panel_says_so(self):
        panel = _ready(12)
        with pytest.raises(ValueError, match="no rolling origin fits"):
            search_transforms(
                panel,
                model_config=ModelConfig(),
                trend_config=TrendConfig(type=TrendType.LINEAR),
                horizon=40,
                min_train_size=10,
            )


class TestOutOfTimeDesign:
    def test_causality_makes_the_training_rows_identical(self):
        """A design built with `evaluate_panel` must reproduce, on the training
        rows, exactly the design built on the prefix alone.

        Adstock is causal, so rows before the cutoff cannot see future spend. If
        this ever failed, the CV would be scoring a model that peeked.
        """
        panel = _ready()
        cut = 40
        prefix = _slice_panel_prefix(panel, cut)
        kw = dict(
            alpha=ALPHA,
            lam=LAM,
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.LINEAR),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = build_design_matrix(prefix, **kw)
            extended = build_design_matrix(prefix, evaluate_panel=panel, **kw)

        assert extended.X.shape[0] > alone.X.shape[0]
        np.testing.assert_allclose(
            extended.X[: alone.X.shape[0]], alone.X, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            extended.y[: alone.y.shape[0]], alone.y, rtol=0, atol=1e-12
        )

    def test_evaluate_panel_defaults_to_the_fit_panel(self):
        panel = _ready()
        kw = dict(
            alpha=ALPHA,
            lam=LAM,
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.LINEAR),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = build_design_matrix(panel, **kw)
            b = build_design_matrix(panel, evaluate_panel=panel, **kw)
        np.testing.assert_allclose(a.X, b.X, rtol=0, atol=0)

    def test_a_shorter_evaluate_panel_is_rejected(self):
        panel = _ready()
        prefix = _slice_panel_prefix(panel, 30)
        with pytest.raises(ValueError, match="must extend panel"):
            build_design_matrix(
                panel,
                alpha=ALPHA,
                lam=LAM,
                model_config=ModelConfig(),
                trend_config=TrendConfig(type=TrendType.LINEAR),
                evaluate_panel=prefix,
            )


def _clean_panel():
    """`make_clean` as a PanelDataset, through the MFF round-trip.

    `make_clean` is the positive control: geometric adstock with normalized
    weights at `l_max=8`, `1 - exp(-lam * x)` saturation, and normalization by
    the channel max *before* adstocking — the model's exact generative family,
    which is what makes recovery well posed rather than approximate.
    """
    import tempfile

    from mmm_framework.data_loader import MFFLoader
    from mmm_framework.synth import generate_mff
    from mmm_framework.synth.dgp import CHANNELS as DGP_CHANNELS

    df, _ = generate_mff("clean", seed=0)
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
            for c in DGP_CHANNELS
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        df.to_csv(fh.name, index=False)
        path = fh.name
    try:
        return MFFLoader(cfg).load(path).build_panel()
    finally:
        os.unlink(path)


def _uninformed(truth: dict, bounds: tuple[float, float], seed: int = 99) -> float:
    """Mean absolute error of a uniform draw from the same bounds."""
    rng = np.random.default_rng(seed)
    return float(
        np.mean(
            [
                [abs(rng.uniform(*bounds) - v) for _ in range(2000)]
                for v in truth.values()
            ]
        )
    )


@pytest.mark.slow
class TestRecoveryOnPlantedTruth:
    """#184's real acceptance criterion: does it recover a planted truth?

    Partly. The honest answer is asserted here rather than rounded up, because
    a search that reads as more trustworthy than it is would be exactly the
    defect this module's own caveat warns about.
    """

    @pytest.fixture(scope="class")
    def panel(self):
        return _clean_panel()

    def _run(self, panel, seed):
        return _search(panel, budget=256, horizon=13, max_origins=3, seed=seed)

    def test_carryover_is_recovered_better_than_chance(self, panel):
        """Alpha lands meaningfully closer than an uninformed draw.

        The near-optimal ENSEMBLE is asserted rather than the single winner:
        across seeds the winner occasionally lands no better than chance
        (0.257 against a 0.261 baseline), while the ensemble is stable.
        Reading the winner alone is the mistake the module warns about.
        """
        baseline = _uninformed(_ALPHA, ADSTOCK_BOUNDS[AdstockType.GEOMETRIC]["alpha"])
        for seed in (0, 1, 2):
            res = self._run(panel, seed)
            near = res.spread(0.10)
            err = float(
                np.mean(
                    [
                        [abs(c.alpha[ch]["alpha"] - _ALPHA[ch]) for ch in _ALPHA]
                        for c in near
                    ]
                )
            )
            assert err < baseline * 0.9, (
                f"seed {seed}: ensemble alpha error {err:.3f} is no better than "
                f"an uninformed draw ({baseline:.3f})"
            )

    def test_saturation_is_not_identified_by_predictive_error(self, panel):
        """The near-optimal set spans essentially the whole allowed window.

        The executable form of the module's caveat: candidates whose
        out-of-sample error is within 10% of the best disagree about TV's
        saturation across nearly the entire search range. Any surface rendering
        the winner's lambda as an estimate is overclaiming.

        Asserted as a FRACTION of the bound rather than an absolute width, so it
        stays meaningful when the bound changes. That matters: anchoring the
        bound to observed spend (HALF_SATURATION_FRACTION) shrank the window from
        80x to 3.3x and cut lambda error fivefold — but it did not make the
        parameter identifiable, and this test is what would notice if someone
        later mistook the narrower bound for identification.
        """
        lo, hi = SATURATION_BOUNDS[SaturationType.LOGISTIC]["sat_lam"]
        res = self._run(panel, 0)
        lams = [c.lam["TV"]["sat_lam"] for c in res.spread(0.10)]
        assert len(lams) >= 5, "too few near-optimal candidates to judge"
        covered = (max(lams) - min(lams)) / (hi - lo)
        assert covered > 0.75, (
            f"near-optimal lambda spans {min(lams):.2f}-{max(lams):.2f}, only "
            f"{covered:.0%} of the allowed window [{lo:.2f}, {hi:.2f}]. If "
            "saturation has become identifiable, the module docstring and the "
            "caveat both need revisiting."
        )

    def test_anchoring_the_bound_to_observed_spend_improves_recovery(self, panel):
        """Robyn's mechanism, measured here: bound the elbow, not the parameter.

        `HALF_SATURATION_FRACTION` puts the half-saturation point inside the
        observed spend range, which is what every production MMM does — Robyn
        via `inflexion = max(x) * gamma`, Meridian by scaling `ec` to median
        spend. It is CONTAINMENT, not identification (see the test above), but
        the containment is worth a great deal: it stops the search proposing
        curves no analyst would entertain.
        """
        lo, hi = SATURATION_BOUNDS[SaturationType.LOGISTIC]["sat_lam"]
        wide = {"sat_lam": (0.1, 8.0)}  # the absolute bound this replaced

        def err(bounds):
            original = SATURATION_BOUNDS[SaturationType.LOGISTIC]
            SATURATION_BOUNDS[SaturationType.LOGISTIC] = bounds
            try:
                res = self._run(panel, 0)
                near = res.spread(0.10)
                return float(
                    np.mean(
                        [
                            [abs(c.lam[ch]["sat_lam"] - _LAM[ch]) for ch in _LAM]
                            for c in near
                        ]
                    )
                )
            finally:
                SATURATION_BOUNDS[SaturationType.LOGISTIC] = original

        anchored = err({"sat_lam": (lo, hi)})
        absolute = err(wide)
        assert anchored < absolute / 2, (
            f"anchored bound gives lambda error {anchored:.3f} vs {absolute:.3f} "
            "for the absolute bound — the measured improvement was ~5x"
        )

    def test_the_criterion_cannot_separate_the_winner_from_the_truth(self, panel):
        """Scored head to head, the search's winner and the planted parameters
        are indistinguishable — in either direction.

        At budget 1000 the winner actually *beats* the truth out of sample
        (0.0328 against 0.0337) while sitting further from it in lambda. That is
        the whole argument for treating the selection as a nuisance to be
        propagated (#186's `refit_search`) rather than a fact to condition on.
        """
        import mmm_framework.frequentist.search as search_mod

        truth = (
            {ch: {"alpha": _ALPHA[ch]} for ch in _ALPHA},
            {ch: {"sat_lam": _LAM[ch]} for ch in _LAM},
        )
        original = search_mod._candidate_points
        search_mod._candidate_points = lambda mmm, budget, strategy, rng: [truth]
        try:
            at_truth = _search(panel, budget=1, horizon=13, max_origins=3, seed=0)
        finally:
            search_mod._candidate_points = original

        found = self._run(panel, 0)
        assert found.best.score <= at_truth.best.score * 1.10, (
            f"search winner {found.best.score:.5f} is much worse than the "
            f"planted truth {at_truth.best.score:.5f} — the search is "
            "under-powered rather than merely non-identifying"
        )


def test_importing_backtest_first_does_not_deadlock_the_package():
    """`validation.backtest` imports this package's numpy transforms.

    So a module-level `from ..validation.backtest import ...` in `search` closes
    the cycle: frequentist.__init__ -> search -> validation.backtest ->
    frequentist._transforms. It is imported lazily inside `search_transforms`
    instead. This fails in exactly the import order that triggered it —
    `test_lean_imports.py` catches it too, but not by name.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mmm_framework.validation.backtest\n"
            "import mmm_framework.frequentist\n"
            "assert mmm_framework.frequentist.search_transforms\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
