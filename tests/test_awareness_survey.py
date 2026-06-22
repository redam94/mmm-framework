"""The synthetic brand-awareness survey generator (`make_awareness_survey`) used
by the awareness model's Atelier demo notebook: a binomial aware-count KPI driven
by a media goodwill stock that decays at a known retention ρ."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mmm_framework.synth import make_awareness_survey


class TestStructure:
    def test_mff_shape_and_answer_key(self):
        df, ans = make_awareness_survey(
            n_weeks=80, n_trials=400, retention=0.75, seed=3
        )
        assert list(df.columns) == [
            "Period",
            "Geography",
            "Product",
            "Campaign",
            "Outlet",
            "Creative",
            "VariableName",
            "VariableValue",
        ]
        vars_ = set(df["VariableName"])
        assert "Awareness" in vars_ and set(ans["channels"]) <= vars_
        aw = df[df["VariableName"] == "Awareness"]["VariableValue"]
        assert (aw >= 0).all() and (aw <= ans["n_trials"]).all()  # binomial counts
        assert ans["true_retention"] == 0.75
        assert ans["n_trials"] == 400
        assert abs(ans["true_half_life_weeks"] - np.log(0.5) / np.log(0.75)) < 1e-6

    def test_aware_rate_varies_and_is_bounded(self):
        df, ans = make_awareness_survey(seed=1)
        aw = df[df["VariableName"] == "Awareness"]["VariableValue"].values
        rate = aw / ans["n_trials"]
        assert 0.05 < rate.mean() < 0.95  # not pinned at 0/100%
        assert rate.std() > 0.02  # the media goodwill actually moves it


@pytest.mark.slow
def test_awareness_model_recovers_retention():
    """End-to-end: the awareness model, fit as a BINOMIAL survey-count KPI with
    number_of_trials, recovers the planted retention ρ from the generated data."""
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../examples/garden_models")
        ),
    )
    from awareness_structural_mmm import AwarenessStructuralMMM

    from mmm_framework.agents.fitting import build_model

    df, ans = make_awareness_survey(n_weeks=104, n_trials=500, retention=0.8, seed=7)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "awareness.csv"
        df.to_csv(path, index=False)
        spec = {
            "kpi": "Awareness",
            "media_channels": [{"name": c} for c in ans["channels"]],
            "trend": {"type": "none"},
            "seasonality": {"yearly": 0, "monthly": 0, "weekly": 0},
            "likelihood": {"family": "binomial"},
            "model_params": {"number_of_trials": ans["n_trials"]},
            "inference": {"method": "map"},
        }
        mmm = build_model(spec, str(path), model_cls=AwarenessStructuralMMM)
        mmm.fit(method="map", random_seed=7)
    rho = float(mmm._trace.posterior["awareness_retention"].values.mean())
    # MAP point estimate on a demo dataset -> a lenient band around the truth.
    assert abs(rho - ans["true_retention"]) < 0.15
