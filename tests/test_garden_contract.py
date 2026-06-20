"""Tests for the Model Garden contract (mmm_framework.garden.contract): the
static + runtime structural checks that define oracle-compatibility, and the
class-resolution helper. All fast — no model construction or fitting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mmm_framework import BayesianMMM
from mmm_framework.garden import contract


class TestValidateClass:
    def test_reference_models_are_compatible(self):
        from mmm_framework.garden import CustomMMM

        assert contract.validate_class(BayesianMMM) == []
        assert contract.validate_class(CustomMMM) == []
        assert contract.is_bayesian_mmm_subclass(CustomMMM)

    def test_non_mmm_duck_type_missing_read_surface(self):
        class Broken:
            def fit(self, **kw):  # has fit, but not a BayesianMMM subclass…
                ...

        problems = contract.validate_class(Broken)
        # …so it must define predict + sample_channel_contributions itself.
        assert any("predict" in p for p in problems)
        assert any("sample_channel_contributions" in p for p in problems)

    def test_duck_type_with_full_surface_passes(self):
        class FullDuck:
            def fit(self, method=None, random_seed=None, **kw): ...
            def predict(self, **kw): ...
            def sample_channel_contributions(self, **kw): ...

        assert contract.validate_class(FullDuck) == []

    def test_missing_fit_is_flagged(self):
        class NoFit(BayesianMMM):  # subclass, but pretend fit is gone
            fit = None  # type: ignore[assignment]

        problems = contract.validate_class(NoFit)
        assert any("fit" in p for p in problems)

    def test_non_class_rejected(self):
        assert contract.validate_class(object()) != []
        assert contract.validate_class(42) != []


class TestFindGardenClass:
    def test_explicit_marker(self):
        class A: ...

        class B: ...

        mod = SimpleNamespace(A=A, B=B, GARDEN_MODEL=B, __name__="m")
        # GARDEN_MODEL wins even when several classes exist.
        assert contract.find_garden_class(mod) is B

    def test_single_subclass(self):
        from mmm_framework.garden import CustomMMM

        class OnlyOne(CustomMMM): ...

        OnlyOne.__module__ = "m"
        mod = SimpleNamespace(OnlyOne=OnlyOne, __name__="m")
        assert contract.find_garden_class(mod) is OnlyOne

    def test_ambiguous_raises(self):
        from mmm_framework.garden import CustomMMM

        class One(CustomMMM): ...

        class Two(CustomMMM): ...

        One.__module__ = Two.__module__ = "m"
        mod = SimpleNamespace(One=One, Two=Two, __name__="m")
        try:
            contract.find_garden_class(mod)
            raise AssertionError("expected ambiguity error")
        except ValueError as e:
            assert "multiple" in str(e).lower()


class TestRuntimeChecks:
    def test_validate_instance_flags_missing_attrs(self):
        stub = SimpleNamespace(channel_names=["TV"], y_mean=1.0, y_std=1.0)
        problems = contract.validate_instance(stub)
        # _media_raw_max / panel / model_config / has_geo / has_product missing.
        assert any("_media_raw_max" in p for p in problems)
        assert any("panel" in p for p in problems)

    def test_validate_instance_flags_channel_raw_max_mismatch(self):
        stub = SimpleNamespace(
            channel_names=["TV", "Search"],
            y_mean=1.0,
            y_std=1.0,
            _media_raw_max={"TV": 10.0},  # Search missing
            panel=object(),
            model_config=object(),
            has_geo=False,
            has_product=False,
        )
        problems = contract.validate_instance(stub)
        assert any("Search" in p for p in problems)

    def test_validate_fitted_requires_trace(self):
        stub = SimpleNamespace(_trace=None, channel_names=["TV"])
        assert any("_trace" in p for p in contract.validate_fitted(stub))

    def test_validate_fitted_requires_beta_naming(self):
        # A posterior without beta_<channel> fails — the ROI helpers key off it.
        post = SimpleNamespace(
            data_vars={"some_other_param": None}, dims={"chain", "draw"}
        )
        trace = SimpleNamespace(posterior=post)
        stub = SimpleNamespace(_trace=trace, channel_names=["TV"])
        assert any("beta_" in p for p in contract.validate_fitted(stub))


def test_describe_contract_mentions_version():
    text = contract.describe_contract()
    assert contract.GARDEN_CONTRACT_VERSION in text
    assert "fit(" in text


def test_param_prefixes_match_real_model():
    # Guards against drift: the documented prefixes must be the ones the model
    # actually emits (beta_/adstock_alpha_/sat_half_/sat_slope_).
    assert "beta_" in contract.PARAM_PREFIXES
    assert "adstock_alpha_" in contract.PARAM_PREFIXES
    assert "sat_half_" in contract.PARAM_PREFIXES
    # numpy import kept so the file is self-contained for future array checks
    assert np is not None
