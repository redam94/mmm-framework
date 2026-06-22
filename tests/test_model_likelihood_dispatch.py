"""The built-in additive model's likelihood dispatch (Phase 2, step 3).

Asserts: the default (normal) path is byte-identical to the legacy hard-coded
``pm.Normal`` (same RV set, same compiled logp at the initial point), Student-T
swaps only the observation node, and non-Gaussian families are refused at build
time with a clear, actionable error. No sampling required — graph construction
only, so these are fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    LikelihoodConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM


def _panel():
    periods = pd.date_range("2021-01-04", periods=30, freq="W-MON")
    rng = np.random.default_rng(3)
    t = np.arange(len(periods))
    tv = np.abs(rng.normal(100, 25, len(periods)))
    digital = np.abs(rng.normal(80, 20, len(periods)))
    y = pd.Series(
        1000 + 8 * t + tv + 0.5 * digital + rng.normal(0, 15, len(periods)),
        name="Sales",
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    return PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=None,
        ),
        index=periods,
        config=config,
    )


def _y_obs_op_name(model) -> str:
    rv = next(v for v in model.observed_RVs if v.name == "y_obs")
    return type(rv.owner.op).__name__


def test_default_builds_normal_and_standardizes():
    mmm = BayesianMMM(_panel(), ModelConfig())
    assert "Normal" in _y_obs_op_name(mmm.model)
    # Gaussian family -> y is z-scored (mean ~0, std ~1), y_std != 1.
    assert mmm._standardizes_y
    assert abs(float(np.mean(mmm.y))) < 1e-6
    assert mmm.y_std > 1.0


def test_default_is_byte_identical_to_explicit_normal():
    """Adding the likelihood field must not perturb the default graph: the
    default config and an explicit LikelihoodConfig.normal() compile to the same
    logp at the same initial point, over the same RV set."""
    a = BayesianMMM(_panel(), ModelConfig())
    b = BayesianMMM(_panel(), ModelConfig(likelihood=LikelihoodConfig.normal()))
    ma, mb = a.model, b.model
    assert {v.name for v in ma.free_RVs} == {v.name for v in mb.free_RVs}
    assert {v.name for v in ma.observed_RVs} == {v.name for v in mb.observed_RVs}
    ip = ma.initial_point()
    la = ma.compile_logp()(ip)
    lb = mb.compile_logp()(ip)
    assert np.isclose(la, lb, rtol=0, atol=0)


def test_student_t_swaps_only_observation():
    base = BayesianMMM(_panel(), ModelConfig())
    st = BayesianMMM(_panel(), ModelConfig(likelihood=LikelihoodConfig.student_t(nu=5)))
    assert "StudentT" in _y_obs_op_name(st.model)
    # Same latent structure as Normal (only the observation node differs).
    assert {v.name for v in base.model.free_RVs} == {v.name for v in st.model.free_RVs}
    assert st._standardizes_y  # student_t is Gaussian-scale


def test_binomial_on_builtin_refused_with_actionable_error():
    mmm = BayesianMMM(
        _panel(),
        ModelConfig(likelihood=LikelihoodConfig.binomial(n_trials=1000)),
    )
    # Binomial does not standardize y, and the additive model refuses to build it.
    assert not mmm._standardizes_y
    with pytest.raises(NotImplementedError, match="own observation block"):
        _ = mmm.model
