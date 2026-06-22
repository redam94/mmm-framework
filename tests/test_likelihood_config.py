"""Tests for the declarative likelihood configuration (Phase 2, step 1).

Covers :class:`LikelihoodConfig` resolution/validation and its integration as a
default-normal field on :class:`ModelConfig` (which must stay backward
compatible: default normal/identity, ``extra:"forbid"`` preserved, round-trips).
The byte-identity of the *built* graph under the default likelihood is asserted
separately once the in-graph dispatch lands (test_model_likelihood_dispatch).
"""

from __future__ import annotations

import pytest

from mmm_framework.config import (
    LikelihoodConfig,
    LikelihoodFamily,
    LinkFunction,
    ModelConfig,
)


class TestLikelihoodConfigDefaults:
    def test_default_is_normal_identity(self):
        lc = LikelihoodConfig()
        assert lc.family is LikelihoodFamily.NORMAL
        assert lc.link is LinkFunction.IDENTITY
        assert lc.is_gaussian and lc.standardizes_y

    def test_canonical_link_resolution(self):
        assert LikelihoodConfig.binomial(n_trials=500).link is LinkFunction.LOGIT
        assert LikelihoodConfig(family="poisson").link is LinkFunction.LOG
        assert LikelihoodConfig(family="beta").link is LinkFunction.LOGIT
        assert LikelihoodConfig.student_t().link is LinkFunction.IDENTITY

    def test_gaussian_classification(self):
        assert LikelihoodFamily.NORMAL.is_gaussian
        assert LikelihoodFamily.STUDENT_T.is_gaussian
        assert not LikelihoodFamily.BINOMIAL.is_gaussian
        # lognormal is fit on standardized log(y): standardizes but not "gaussian"
        # for the additive dispatch (it needs the upstream log transform).
        assert LikelihoodFamily.LOGNORMAL.standardizes_y
        assert not LikelihoodFamily.LOGNORMAL.is_gaussian
        assert not LikelihoodFamily.BINOMIAL.standardizes_y
        assert not LikelihoodFamily.POISSON.standardizes_y


class TestLikelihoodConfigValidation:
    def test_binomial_n_trials_optional(self):
        # The family declares the scale/observation type; a custom model may
        # source n_trials from its own CONFIG_SCHEMA, so it is not required here.
        lc = LikelihoodConfig(family="binomial")
        assert lc.family is LikelihoodFamily.BINOMIAL
        assert "n_trials" not in lc.params

    def test_incoherent_link_rejected(self):
        with pytest.raises(ValueError, match="link"):
            LikelihoodConfig(family="normal", link="logit")
        with pytest.raises(ValueError):  # unknown field -> extra forbid
            LikelihoodConfig(family="binomial", n_trials_link_typo=None)

    def test_n_trials_column_name_allowed(self):
        lc = LikelihoodConfig.binomial(n_trials="survey_n")
        assert lc.params["n_trials"] == "survey_n"

    def test_n_trials_must_be_positive_int_or_str(self):
        for bad in (0, -3, 8.5, True):
            with pytest.raises(ValueError):
                LikelihoodConfig(family="binomial", params={"n_trials": bad})

    def test_extra_forbidden(self):
        with pytest.raises(ValueError):
            LikelihoodConfig(family="normal", nonsense=1)

    def test_round_trip(self):
        for lc in (
            LikelihoodConfig.normal(),
            LikelihoodConfig.student_t(nu=6),
            LikelihoodConfig.binomial(n_trials=1000),
            LikelihoodConfig(family="poisson"),
        ):
            assert LikelihoodConfig.model_validate(lc.model_dump()) == lc


class TestModelConfigIntegration:
    def test_modelconfig_defaults_to_normal(self):
        mc = ModelConfig()
        assert mc.likelihood.family is LikelihoodFamily.NORMAL
        assert mc.likelihood.link is LinkFunction.IDENTITY

    def test_modelconfig_round_trips_with_likelihood(self):
        mc = ModelConfig(likelihood=LikelihoodConfig.student_t(nu=5))
        restored = ModelConfig.model_validate(mc.model_dump())
        assert restored.likelihood.family is LikelihoodFamily.STUDENT_T
        assert restored.likelihood.params["nu"] == 5

    def test_modelconfig_still_forbids_extra(self):
        with pytest.raises(ValueError):
            ModelConfig(not_a_real_field=123)

    def test_default_modelconfig_dump_is_stable(self):
        # The likelihood field must serialize without surprising the existing
        # spec/diff/lock machinery (it diffs leaf values).
        dump = ModelConfig().model_dump()
        assert dump["likelihood"]["family"] == "normal"
        assert dump["likelihood"]["link"] == "identity"
        assert dump["likelihood"]["params"] == {}
