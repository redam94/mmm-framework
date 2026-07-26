"""Nothing may call a bootstrap confidence interval a posterior.

#188 is the integration risk of epic #180. The estimators were self-contained;
making them reachable was easy, and making the rest of the framework tell the
truth about them was not — every report section, estimand label and diagnostic
in this codebase was written assuming a posterior.

The failure modes here are **strings and verdicts**, not numbers, so the tests
are string and verdict tests. That is not a shortcut: a report that renders "90%
credible interval" over a bootstrap percentile interval is wrong in exactly one
observable way, and grepping the rendered HTML is the direct check.

The root cause worth remembering (measured in the spec, §8): a
``(chain=1, draw=B)`` bootstrap trace passes **every** convergence gate as
``True``. ``az.rhat`` on one chain is NaN → filtered to ``None``; a ``None``
metric raises no flag; ``az.ess`` returns ≈B because bootstrap replicates are
iid. So before the paradigm branch, a ridge fit was silently green everywhere
the verdict is consumed, and the Augur client deck — whose only stop sign fires
on ``approximate`` or ``is_converged is False`` — rendered with zero caveat.
"""

from __future__ import annotations

import re
import warnings

import pytest

from mmm_framework.config import ModelConfig
from mmm_framework.config.enums import InferenceMethod, SaturationType
from mmm_framework.diagnostics import provenance as prov
from mmm_framework.diagnostics.convergence import (
    annotate,
    convergence_flags,
    is_converged,
)
from mmm_framework.model import BayesianMMM
from mmm_framework.model.trend_config import TrendConfig, TrendType

from test_design_equivalence import _configure, _panel

TC = TrendConfig(type=TrendType.LINEAR)
SEARCH = {"budget": 6, "horizon": 10, "max_origins": 1}

#: Claims that must never appear in a rendered frequentist report. Deliberately
#: phrased as CLAIMS rather than bare words: the estimator's own caveat says
#: "not credible intervals", and the gated sections explain that R-hat describes
#: an MCMC sampler — both contain the words and both are the desired output.
FORBIDDEN = {
    "sized credible interval": r"\d+% credible",
    "credible-interval claim": r"credible interval[s]? (reflecting|excludes|capturing)",
    "posterior mean": r"posterior mean",
    "bayesian p-value": r"Bayes p|posterior[- ]predictive p-value",
    "green convergence": r"✅ Pass",
    "full MCMC claim": r"full MCMC posterior",
}


def _freq_model(method=InferenceMethod.FREQUENTIST_RIDGE, n_periods=104):
    panel = _panel(n_periods=n_periods)
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    config = ModelConfig(
        inference_method=method, bootstrap_samples=30, optim_maxiter=6
    )
    return BayesianMMM(panel, config, TC)


def _fit(model, **kw):
    kw.setdefault("search_kwargs", SEARCH)
    kw.setdefault("random_seed", 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(**kw)


@pytest.fixture(scope="module")
def fitted():
    model = _freq_model()
    return model, _fit(model)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


class TestDispatch:
    def test_fit_routes_to_the_frequentist_path(self, fitted):
        _, results = fitted
        assert results.diagnostics["inference_family"] == prov.FREQUENTIST
        assert results.diagnostics["estimator"] == "ridge"
        assert results.diagnostics["interval_kind"] == "bootstrap_percentile"

    def test_paradigm_beats_fit_method(self):
        """``method=`` selects among the BAYESIAN estimators and must not
        redirect a frequentist config into MAP."""
        model = _freq_model()
        results = _fit(model, method="map")
        assert results.diagnostics["inference_family"] == prov.FREQUENTIST

    def test_config_records_no_fit_method(self, fitted):
        """A stray "nuts" here is what makes downstream surfaces announce a
        full MCMC posterior for a ridge fit."""
        model, results = fitted
        assert model.model_config.fit_method is None
        assert results.diagnostics["fit_method"] is None

    def test_trace_is_one_chain_of_replicates(self, fitted):
        _, results = fitted
        assert results.trace.posterior.sizes["chain"] == 1
        assert results.trace.posterior.sizes["draw"] == 30

    def test_downstream_model_surface_still_works(self, fitted):
        """The container is a drop-in: predict and the marginal engine read it
        without knowing which paradigm produced it."""
        model, _ = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert model.predict().y_pred_mean.shape[0] == model.n_obs
            assert len(model.compute_marginal_contributions()) == 2

    def test_cvxpy_method_selects_the_constrained_estimator(self):
        cvxpy = pytest.importorskip(
            "cvxpy", reason="the constrained estimator is an optional extra"
        )
        assert cvxpy is not None
        results = _fit(_freq_model(InferenceMethod.FREQUENTIST_CVXPY))
        assert results.diagnostics["estimator"] == "constrained"
        assert results.diagnostics["inference_family"] == prov.FREQUENTIST

    def test_unsupported_model_names_the_feature(self):
        from mmm_framework.frequentist.design import UnsupportedModelError

        panel = _panel()
        _configure(panel, "geometric", SaturationType.LOGISTIC)
        model = BayesianMMM(
            panel,
            ModelConfig(inference_method=InferenceMethod.FREQUENTIST_RIDGE),
            TrendConfig(type=TrendType.GP),
        )
        # The refusal happens BEFORE the transform search, so the message names
        # the model feature that is not linear rather than the search step that
        # happened to trip over it first.
        with pytest.raises(UnsupportedModelError, match="Gaussian-process"):
            _fit(model)


# --------------------------------------------------------------------------- #
# the convergence gate — the root cause
# --------------------------------------------------------------------------- #


class TestConvergenceGate:
    def test_verdict_is_not_assessable(self, fitted):
        _, results = fitted
        assert results.converged is None

    def test_metrics_are_nulled_not_computed(self, fitted):
        _, results = fitted
        for key in ("rhat_max", "ess_bulk_min", "divergences"):
            assert results.diagnostics[key] is None, key

    def test_the_exact_trace_that_used_to_read_as_converged(self):
        """A bootstrap-shaped diagnostics dict, with the numbers the audit
        measured: NaN R-hat filtered to None, and ESS ≈ n_boot because iid
        replicates carry no autocorrelation."""
        diag = {"divergences": None, "rhat_max": None, "ess_bulk_min": 966.5}
        assert is_converged(diag) is True  # the defect, still present generally
        diag["inference_family"] = prov.FREQUENTIST
        assert is_converged(diag) is None
        assert convergence_flags(diag) == []

    def test_annotate_nulls_the_sampler_metrics(self):
        diag = annotate(
            {"inference_family": prov.FREQUENTIST, "ess_bulk_min": 966.5,
             "rhat_max": 1.0, "divergences": 0}
        )
        assert diag["converged"] is None
        assert diag["ess_bulk_min"] is None
        assert diag["rhat_max"] is None

    def test_bayesian_fits_are_untouched(self):
        diag = {"rhat_max": 1.001, "ess_bulk_min": 1200.0, "divergences": 0}
        assert is_converged(diag) is True
        assert is_converged({**diag, "rhat_max": 1.4}) is False

    def test_absence_of_the_key_reads_as_bayesian(self):
        """Every fit produced before this path existed has no such key."""
        assert prov.family_of({}) == prov.BAYESIAN
        assert prov.family_of(None) == prov.BAYESIAN
        assert not prov.is_frequentist({"approximate": True})


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #


class TestVocabulary:
    def test_interval_noun_switches_on_family(self):
        assert prov.interval_noun(prov.BAYESIAN) == "credible interval"
        assert prov.interval_noun(prov.FREQUENTIST) == "confidence interval"
        assert prov.interval_noun(prov.FREQUENTIST, plural=True).endswith("s")

    def test_interval_noun_accepts_a_diagnostics_dict(self):
        """Call sites hold one or the other; converting at every one of them is
        how a surface gets missed."""
        assert (
            prov.interval_noun({"inference_family": "frequentist"})
            == "confidence interval"
        )

    def test_interval_phrase_names_the_bootstrap(self):
        assert prov.interval_phrase(0.9, prov.FREQUENTIST) == (
            "90% bootstrap confidence interval"
        )
        assert prov.interval_phrase(0.9, prov.BAYESIAN) == "90% credible interval"

    def test_not_applicable_reasons_explain_rather_than_blank(self):
        for slug in ("convergence", "posterior_predictive", "prior", "learning"):
            reason = prov.not_applicable_reason(slug)
            assert len(reason) > 60, slug
            assert "posterior" in reason or "sampler" in reason or "prior" in reason

    def test_caveats_fall_back_when_diagnostics_were_trimmed(self):
        """A reloaded fit whose caveats were dropped must still say something
        true rather than nothing."""
        fallback = prov.frequentist_caveats({"inference_family": "frequentist"})
        assert any("confidence interval" in c for c in fallback)
        assert any("biased" in c for c in fallback)


# --------------------------------------------------------------------------- #
# provenance carried outward
# --------------------------------------------------------------------------- #


class TestProvenanceCarriesOutward:
    def test_merge_fit_provenance_branches_on_family(self, fitted):
        """The report recomputes R-hat/ESS from the raw trace and would
        otherwise let those decide the verdict."""
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        model, results = fitted
        extractor = BayesianMMMExtractor(model, results=results)
        merged = extractor._merge_fit_provenance(
            {"rhat_max": 1.0, "ess_bulk_min": 966.5, "divergences": 0}
        )
        assert merged["inference_family"] == prov.FREQUENTIST
        assert merged["converged"] is None
        assert merged["rhat_max"] is None and merged["ess_bulk_min"] is None
        assert merged["approximate"] is False

    def test_serializer_metadata_does_not_read_as_bayesian(self, fitted, tmp_path):
        """`approximate` is derived from fit_method, which is None here — so
        without its own stamp a reloaded ridge fit reads as Bayesian by
        omission."""
        from mmm_framework.serialization import MMMSerializer

        model, _ = fitted
        path = tmp_path / "freq_model"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MMMSerializer.save(model, str(path))
        import json

        meta = json.loads((path / "metadata.json").read_text())
        assert meta["inference_family"] == prov.FREQUENTIST
        assert meta["inference_method"] == "frequentist_ridge"
        assert meta["approximate"] is False
        assert meta["estimator"] == "ridge"
        assert meta["interval_semantics"] == "conditional_on_selection"

    def test_agent_registry_accepts_and_routes_the_method(self):
        from mmm_framework.agents.fitting import (
            _FREQUENTIST_METHODS,
            _INFERENCE_METHODS,
            unconsumed_spec_path,
        )

        spec = {"inference": {"method": "nuts"}}
        path = ["inference", "method"]
        assert _FREQUENTIST_METHODS <= _INFERENCE_METHODS
        assert unconsumed_spec_path(path, "frequentist_ridge", spec) is None
        assert unconsumed_spec_path(path, "frequentist_cvxpy", spec) is None
        # A near-miss must still be caught when the setting is WRITTEN, not at
        # fit time an hour later.
        assert unconsumed_spec_path(path, "ridge", spec) is not None

    def test_interactive_facts_carry_the_family(self, fitted):
        """The JS inherits its interval wording from this slot; without it the
        inference card defaults a missing fit_method to "NUTS"."""
        from mmm_framework.reporting.interactive import interactive_report_facts

        model, results = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            facts = interactive_report_facts(model, results, max_draws=20)
        meta = facts["meta"]
        assert meta["inference_family"] == prov.FREQUENTIST
        assert meta["fit_method"] is None
        assert meta["frequentist_caveats"]
        # Posterior-only blocks are blanked at the one place they leave the
        # facts layer, so a new section cannot pick them up by accident.
        assert not facts["ppc_stats"]
        assert facts["ppc_prior"] is None
        assert not facts["prior_posterior"]["rows"]


# --------------------------------------------------------------------------- #
# the acceptance test — a rendered report must not lie
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestRenderedReportsTellTheTruth:
    @staticmethod
    def _assert_clean(html: str, label: str) -> None:
        offenders = {
            name: len(re.findall(pattern, html, re.I))
            for name, pattern in FORBIDDEN.items()
        }
        offenders = {k: v for k, v in offenders.items() if v}
        assert not offenders, f"{label} report makes posterior claims: {offenders}"
        assert re.search("confidence interval", html, re.I), (
            f"{label} report names no confidence interval at all — the wording "
            "was gated off rather than corrected"
        )

    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.interactive import InteractiveReportGenerator

        tmp = tmp_path_factory.mktemp("freq_reports")
        model = _freq_model()
        results = _fit(model)
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for shell in ("classic", "augur"):
                cfg = ReportConfig(shell=shell) if shell == "augur" else ReportConfig()
                gen = MMMReportGenerator(model=model, results=results, config=cfg)
                out[shell] = gen.to_html(tmp / f"{shell}.html").read_text()
            out["interactive"] = InteractiveReportGenerator(
                model=model, results=results
            ).generate_report()
        return out

    @pytest.mark.parametrize("shell", ["classic", "augur", "interactive"])
    def test_no_posterior_claims(self, rendered, shell):
        self._assert_clean(rendered[shell], shell)

    def test_banner_is_present_in_every_shell(self, rendered):
        assert "Frequentist fit" in rendered["classic"]
        # The client deck speaks plain language, not statistics vocabulary.
        assert "fast statistical estimate" in rendered["augur"]
        assert "Frequentist fit" in rendered["interactive"]

    def test_diagnostics_section_explains_rather_than_blanks(self, rendered):
        html = rendered["classic"]
        assert "Not applicable to this fit" in html
        assert "describe an MCMC sampler" in html

    def test_selection_conditioning_is_stated(self, rendered):
        """The cheap interval's deficiency must be named where the number is
        read, not only in the docstring of the function that made it."""
        assert "OMITS selection uncertainty" in rendered["classic"]
