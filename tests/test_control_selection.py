"""Control variable-selection wiring (horseshoe / spike-slab / lasso).

Graph-structure tests (no MCMC) verifying:
* the feature is OFF by default and leaves ``beta_controls`` a plain Normal RV
  (the historical, bit-identical path),
* when ON, confounders stay un-shrunk while the remaining controls get the
  selection prior, and the ``beta_controls`` node (name/shape) is preserved.

The empirical bit-identical-vs-HEAD check and the actual-shrinkage check are run
out-of-band (see the walkthrough notebook / robustness tooling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    CausalControlRole,
    ControlSelectionConfig,
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType


def _model(roles: dict, method: str) -> BayesianMMM:
    rng = np.random.default_rng(0)
    periods = pd.date_range("2022-01-03", periods=40, freq="W-MON")
    n = len(periods)
    controls = list(roles)
    coords = PanelCoordinates(
        periods=periods, geographies=None, products=None,
        channels=["TV", "Digital"], controls=controls,
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
            for c in ["TV", "Digital"]
        ],
        controls=[
            ControlVariableConfig(
                name=c, dimensions=[DimensionType.PERIOD], causal_role=roles[c]
            )
            for c in controls
        ],
    )
    panel = PanelDataset(
        y=pd.Series(1000 + rng.normal(0, 40, n), name="Sales"),
        X_media=pd.DataFrame({
            "TV": np.abs(rng.normal(100, 30, n)),
            "Digital": np.abs(rng.normal(80, 20, n)),
        }),
        X_controls=pd.DataFrame({c: rng.normal(0, 1, n) for c in controls}),
        coords=coords, index=periods, config=config,
    )
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        control_selection=ControlSelectionConfig(method=method, expected_nonzero=2),
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.NONE))


def test_off_by_default_is_plain_normal_beta_controls():
    m = _model({"Price": None, "Comp": None, "Weather": None}, "none")
    free = {v.name for v in m.model.free_RVs}
    assert "beta_controls" in free  # a plain free RV, as before
    assert not any("select" in n or "confounder" in n for n in free)


@pytest.mark.parametrize("method", ["horseshoe", "spike_slab", "lasso"])
def test_selection_on_keeps_contract_and_exempts_confounders(method):
    roles = {"Demand": CausalControlRole.CONFOUNDER, "Price": None, "Weather": None}
    m = _model(roles, method)
    free = {v.name for v in m.model.free_RVs}
    det = {d.name for d in m.model.deterministics}
    # contract: beta_controls still exists (now assembled as a Deterministic)
    assert "beta_controls" in det
    # confounder is un-shrunk in its own Normal, not handed to the selector
    assert "beta_controls_confounder" in free
    # the selection prior is applied to the remaining (selectable) controls
    assert any("beta_controls_select" in n for n in free)


def test_selection_inactive_without_enough_selectable_controls():
    # all confounders -> nothing to select -> fall back to the plain Normal path
    roles = {"A": CausalControlRole.CONFOUNDER, "B": CausalControlRole.CONFOUNDER}
    m = _model(roles, "horseshoe")
    free = {v.name for v in m.model.free_RVs}
    assert "beta_controls" in free
    assert not any("select" in n for n in free)
