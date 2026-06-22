"""Foundation for non-MMM model families (CFA/LCA/…): the capability-gated garden
contract + compat, and the bare-`LatentVar` estimand realization. These exercise
the reusable plumbing without fitting a real model (the CFA end-to-end lives in
test_cfa_garden_model.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


class _DeclaredCFA:
    __garden_model_kind__ = "cfa"

    def fit(self, method=None, random_seed=None):  # noqa: D401, ARG002
        ...


class _DuckMMM:  # not a BayesianMMM subclass, no declared kind
    def fit(self, method=None):  # noqa: ARG002
        ...


class TestModelKindGating:
    def test_mmm_default_and_helpers(self):
        from mmm_framework.garden import contract as C
        from mmm_framework.garden.base import CustomMMM

        assert C.model_kind(CustomMMM) == "mmm"
        assert C.is_mmm_model(CustomMMM)
        assert C.model_kind(_DeclaredCFA) == "cfa"
        assert not C.is_mmm_model(_DeclaredCFA)
        # instance and class agree
        assert C.is_mmm_model(_DeclaredCFA()) is False

    def test_unknown_standalone_treated_as_mmm(self):
        from mmm_framework.garden import contract as C

        # No declaration + not a BayesianMMM subclass -> kind "unknown", but the
        # MMM gates still apply (historical default; only an explicit non-MMM
        # declaration opts out).
        assert C.model_kind(_DuckMMM) == "unknown"
        assert C.is_mmm_model(_DuckMMM)

    def test_validate_class_exempts_declared_non_mmm(self):
        from mmm_framework.garden.contract import validate_class

        # A declared non-MMM family only needs fit(); no sample_channel_contributions.
        assert validate_class(_DeclaredCFA) == []
        # A duck-typed unknown class still must implement the channel read surface.
        probs = validate_class(_DuckMMM)
        assert any("sample_channel_contributions" in p for p in probs)

    def test_validate_instance_skips_channel_checks_for_non_mmm(self):
        from mmm_framework.garden.contract import validate_instance

        # Non-MMM instance: no channels / _media_raw_max required, only base attrs.
        cfa = SimpleNamespace(
            __garden_model_kind__="cfa",
            y_mean=0.0,
            y_std=1.0,
            panel=object(),
            model_config=object(),
            has_geo=False,
            has_product=False,
        )
        assert validate_instance(cfa) == []
        # An MMM instance with empty channels IS flagged.
        mmm = SimpleNamespace(
            channel_names=[],
            _media_raw_max={},
            y_mean=0.0,
            y_std=1.0,
            panel=object(),
            model_config=object(),
            has_geo=False,
            has_product=False,
        )
        assert any("channel_names" in p for p in validate_instance(mmm))

    def test_validate_fitted_skips_beta_check_for_non_mmm(self):
        from mmm_framework.garden.contract import validate_fitted

        post = xr.Dataset({"factor_loadings": (("chain", "draw"), np.zeros((2, 10)))})
        cfa = SimpleNamespace(
            __garden_model_kind__="cfa",
            _trace=SimpleNamespace(posterior=post),
            channel_names=[],
        )
        # No beta_<channel> required; well-formed (chain/draw) trace passes.
        assert validate_fitted(cfa) == []


class TestManifestModelKind:
    def test_ast_detects_declared_kind(self):
        from mmm_framework.agents.garden_registry import static_model_kind

        src = (
            "from mmm_framework.garden import CustomMMM\n"
            "class Foo(CustomMMM):\n"
            "    __garden_model_kind__ = 'cfa'\n"
            "    def fit(self, method=None): ...\n"
        )
        assert static_model_kind(src, "Foo") == "cfa"
        assert static_model_kind(src, None) == "cfa"

    def test_ast_defaults_to_mmm(self):
        from mmm_framework.agents.garden_registry import static_model_kind

        src = "class Bar:\n    def fit(self): ...\n"
        assert static_model_kind(src, "Bar") == "mmm"


class TestBareLatentVarEstimand:
    def _model_with_posterior(self, post: xr.Dataset, n_obs: int):
        trace = SimpleNamespace(posterior=post)
        return SimpleNamespace(
            _trace=trace,
            trace=trace,
            channel_names=[],
            n_obs=n_obs,
            _get_time_mask=lambda tp: np.ones(n_obs, dtype=bool),
        )

    def test_scalar_latent_realizes_mean_hdi(self):
        from mmm_framework.estimands import registry
        from mmm_framework.estimands.evaluate import EstimandEvaluator

        rng = np.random.default_rng(0)
        post = xr.Dataset(
            {"cfi": (("chain", "draw"), rng.uniform(0.9, 0.99, (2, 200)))}
        )
        model = self._model_with_posterior(post, 10)
        res = EstimandEvaluator(model).evaluate([registry.fit_index("cfi")])
        r = res["cfi"]
        assert r.status == "ok"
        assert 0.9 <= r.mean <= 0.99
        assert r.hdi_low < r.hdi_high  # real spread

    def test_obs_indexed_latent_window_mean(self):
        from mmm_framework.estimands import registry
        from mmm_framework.estimands.evaluate import EstimandEvaluator

        rng = np.random.default_rng(1)
        post = xr.Dataset(
            {"level": (("chain", "draw", "obs"), rng.normal(3.0, 0.1, (2, 100, 8)))}
        )
        model = self._model_with_posterior(post, 8)
        res = EstimandEvaluator(model).evaluate([registry.latent_scalar("level")])
        assert res["level"].status == "ok"
        assert abs(res["level"].mean - 3.0) < 0.2

    def test_matrix_latent_is_unsupported(self):
        from mmm_framework.estimands import registry
        from mmm_framework.estimands.evaluate import EstimandEvaluator

        post = xr.Dataset(
            {
                "loadings": (
                    ("chain", "draw", "indicator", "factor"),
                    np.zeros((2, 5, 4, 2)),
                )
            }
        )
        model = self._model_with_posterior(post, 6)
        res = EstimandEvaluator(model).evaluate([registry.factor_loading("loadings")])
        r = res["loadings"]
        assert r.status == "unsupported"
        assert "array-valued" in (r.reason or "")

    def test_missing_capability_degrades(self):
        from mmm_framework.estimands import registry
        from mmm_framework.estimands.evaluate import EstimandEvaluator

        post = xr.Dataset({"cfi": (("chain", "draw"), np.ones((2, 10)))})
        model = self._model_with_posterior(post, 4)
        # Ask for a latent the posterior doesn't carry -> unsupported (capability gate).
        res = EstimandEvaluator(model).evaluate([registry.fit_index("rmsea")])
        assert res["rmsea"].status == "unsupported"


def test_defaults_for_empty_without_mmm_capabilities():
    from mmm_framework.estimands import registry

    # A model with only latent capabilities gets no MMM defaults.
    assert registry.defaults_for({"HAS_LATENT:cfi"}) == []
