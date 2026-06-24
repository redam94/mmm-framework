"""Per-model config schema + likelihood threading through build_model (Phase 2,
steps 2/4). A bespoke model declares a ``CONFIG_SCHEMA``; the agent/spec layer
validates ``spec["model_params"]`` against it (defaults + validators applied) and
hands it to the constructor, and ``spec["likelihood"]`` flows into
``model_config.likelihood``. Uses the explicit ``model_cls=`` path (no garden
loader needed) + an in-memory MFF CSV.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field

from mmm_framework.config import LikelihoodFamily
from mmm_framework.garden import CustomMMM


def _write_synth_mff(tmp_path, n_weeks=30):
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

    df = generate_synthetic_mff(n_weeks=n_weeks)
    path = str(tmp_path / "mff.csv")
    df.to_csv(path, index=False)
    return path


class _Params(BaseModel):
    """A bespoke per-model schema with defaults + a validator."""

    number_of_trials: int = Field(default=500, gt=0)
    awareness_retention: float = 0.75

    model_config = {"extra": "forbid"}


class _ParamModel(CustomMMM):
    """Minimal bespoke model that declares a CONFIG_SCHEMA and records what it
    received (it doesn't override _build_model — it just needs to construct)."""

    CONFIG_SCHEMA = _Params


def _base_spec():
    return {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}],
        "control_variables": [],
    }


class TestModelParamsThreading:
    def test_defaults_applied_when_absent(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        mmm = build_model(_base_spec(), path, model_cls=_ParamModel)
        assert isinstance(mmm.model_params, _Params)
        assert mmm.model_params.number_of_trials == 500  # default tracked
        assert mmm.model_params.awareness_retention == 0.75

    def test_spec_overrides_and_validates(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        spec = {**_base_spec(), "model_params": {"number_of_trials": 2000}}
        mmm = build_model(spec, path, model_cls=_ParamModel)
        assert mmm.model_params.number_of_trials == 2000
        assert mmm.model_params.awareness_retention == 0.75  # default kept

    def test_invalid_model_params_clear_error(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        spec = {**_base_spec(), "model_params": {"number_of_trials": -5}}
        with pytest.raises(ValueError, match="Invalid model_params for _ParamModel"):
            build_model(spec, path, model_cls=_ParamModel)

    def test_base_model_ignores_model_params(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        # No CONFIG_SCHEMA on the base model -> passed through untouched.
        spec = {**_base_spec(), "model_params": {"anything": 1}}
        mmm = build_model(spec, path)
        assert mmm.model_params == {"anything": 1}


class TestModelParamsSerialization:
    def test_model_params_in_collected_metadata(self, tmp_path):
        # A model with a CONFIG_SCHEMA serializes its validated model_params into
        # metadata (a plain dict the reloaded model re-validates). No fit needed:
        # _collect_metadata reads config/names/params off the constructed model.
        from mmm_framework.agents.fitting import build_model
        from mmm_framework.serialization import MMMSerializer

        path = _write_synth_mff(tmp_path)
        spec = {**_base_spec(), "model_params": {"number_of_trials": 1234}}
        mmm = build_model(spec, path, model_cls=_ParamModel)

        meta = MMMSerializer._collect_metadata(mmm)
        assert meta["model_params"] == {
            "number_of_trials": 1234,
            "awareness_retention": 0.75,  # default tracked through serialization
        }
        # Likelihood rides inside model_config (nested) — already round-tripped.
        assert "likelihood" in meta or "model_params_schema_version" in meta

    def test_base_model_omits_model_params_metadata(self, tmp_path):
        from mmm_framework.agents.fitting import build_model
        from mmm_framework.serialization import MMMSerializer

        path = _write_synth_mff(tmp_path)
        mmm = build_model(_base_spec(), path)  # base model, model_params is None
        meta = MMMSerializer._collect_metadata(mmm)
        assert "model_params" not in meta


class TestLikelihoodThreading:
    def test_spec_likelihood_flows_into_model_config(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        spec = {
            **_base_spec(),
            "likelihood": {"family": "student_t", "params": {"nu": 6}},
        }
        mmm = build_model(spec, path)
        assert mmm.model_config.likelihood.family is LikelihoodFamily.STUDENT_T
        assert mmm.model_config.likelihood.params["nu"] == 6

    def test_default_likelihood_is_normal(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        mmm = build_model(_base_spec(), path)
        assert mmm.model_config.likelihood.family is LikelihoodFamily.NORMAL

    def test_invalid_likelihood_clear_error(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        # Incoherent link for the family -> clear, build-context error.
        spec = {**_base_spec(), "likelihood": {"family": "normal", "link": "logit"}}
        with pytest.raises(ValueError, match="Invalid spec.likelihood"):
            build_model(spec, path)
