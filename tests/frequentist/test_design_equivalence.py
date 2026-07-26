"""The design matrix must reproduce the graph's mean exactly.

This is the load-bearing test of epic #180. The frequentist path solves
``X @ theta`` where the Bayesian path evaluates a PyTensor graph; if the two ever
disagree, a ridge fit and a NUTS fit stop being comparable and every benchmark
between them measures the drift rather than the estimators.

The check is direct rather than statistical: build the graph, substitute a fixed
constant for every free random variable, evaluate the component Deterministics
(whose sum *is* ``mu`` — compare ``model/base.py``'s COMBINE AND LIKELIHOOD block),
and compare against the design matrix at the matching parameter vector.

Three reparameterizations have to be undone to line the two up, and each is a
deliberate design decision documented in ``technical-docs/frequentist-estimation.md``:

* geo / product enter the graph as ``sigma * offset[idx]``; the design carries
  penalized dummies whose coefficient is that composite;
* the spline trend's coefficient is ``spline_scale * cumsum(raw)``, and the graph
  demeans the trend output where the design demeans the basis columns;
* ``trend_m`` (piecewise) is a second constant column, exactly collinear with the
  intercept, so the design folds it into the intercept.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
from pytensor.graph.replace import graph_replace

from mmm_framework.config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
    SaturationConfig,
)
from mmm_framework.config.enums import SaturationType
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.frequentist.design import UnsupportedModelError, build_design_matrix
from mmm_framework.model import BayesianMMM
from mmm_framework.model.trend_config import TrendConfig, TrendType

CHANNELS = ["TV", "Digital"]
TOL = 1e-12

_ADSTOCK = {
    "geometric": (AdstockConfig.geometric, {"alpha": 0.55}),
    "delayed": (AdstockConfig.delayed, {"alpha": 0.55, "theta": 1.5}),
    "weibull": (AdstockConfig.weibull, {"shape": 2.0, "scale": 3.0}),
}
_SATURATION = {
    SaturationType.LOGISTIC: (SaturationConfig.logistic, {"sat_lam": 2.7}),
    SaturationType.HILL: (SaturationConfig.hill, {"sat_half": 0.4, "sat_slope": 1.5}),
    SaturationType.ROOT: (SaturationConfig.root, {"sat_exponent": 0.6}),
    SaturationType.TANH: (SaturationConfig.tanh, {"sat_half": 0.4}),
    SaturationType.MICHAELIS_MENTEN: (
        SaturationConfig.michaelis_menten,
        {"sat_half": 0.4},
    ),
}


def _panel(geos: list[str] | None = None, n_periods: int = 104) -> PanelDataset:
    """A panel with a flighted channel, so zero-spend rows are exercised."""
    periods = pd.date_range("2020-01-06", periods=n_periods, freq="W-MON")
    glist = geos or [None]
    idx = [p for _ in glist for p in periods]
    rows = [g for g in glist for _ in periods]
    n = len(idx)
    rng = np.random.default_rng(42)

    tv = np.abs(rng.standard_normal(n) * 50 + 100)
    tv[rng.random(n) < 0.25] = 0.0  # flighted: exercises the saturation guards

    dims = [DimensionType.PERIOD] + ([DimensionType.GEOGRAPHY] if geos else [])
    mff = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=dims),
        media_channels=[MediaChannelConfig(name=c, dimensions=dims) for c in CHANNELS],
        controls=[ControlVariableConfig(name="Price", dimensions=dims)],
    )
    index: pd.Index
    if geos:
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(idx), rows],
            names=[mff.columns.period, mff.columns.geography],
        )
    else:
        index = pd.DatetimeIndex(idx)

    return PanelDataset(
        y=pd.Series(1000 + rng.standard_normal(n) * 100, name="Sales"),
        X_media=pd.DataFrame(
            {"TV": tv, "Digital": np.abs(rng.standard_normal(n) * 30 + 80)}
        ),
        X_controls=pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5}),
        coords=PanelCoordinates(
            periods=periods,
            geographies=geos,
            products=None,
            channels=CHANNELS,
            controls=["Price"],
        ),
        index=index,
        config=mff,
    )


def _configure(panel: PanelDataset, adstock: str, sat: SaturationType) -> MFFConfig:
    acfg = _ADSTOCK[adstock][0]()
    scfg = _SATURATION[sat][0]()
    base = panel.config
    dims = base.media_channels[0].dimensions
    mff = MFFConfig(
        kpi=base.kpi,
        media_channels=[
            MediaChannelConfig(name=c, dimensions=dims, adstock=acfg, saturation=scfg)
            for c in CHANNELS
        ],
        controls=base.controls,
    )
    panel.config = mff
    return mff


def _fixed_point(model, seed: int = 7) -> dict[str, np.ndarray | float]:
    """A deterministic value for every free RV in the graph."""
    rs = np.random.default_rng(seed)
    fixed = {
        "adstock_alpha": 0.55,
        "adstock_theta": 1.5,
        "adstock_shape": 2.0,
        "adstock_scale": 3.0,
        "sat_lam": 2.7,
        "sat_half": 0.4,
        "sat_slope": 1.5,
        "sat_exponent": 0.6,
    }
    point: dict[str, np.ndarray | float] = {}
    for rv in model.free_RVs:
        name = rv.name
        shape = tuple(int(s) for s in rv.shape.eval())
        match = next((v for k, v in fixed.items() if name.startswith(k)), None)
        if match is not None:
            point[name] = match
        elif name.startswith("beta_") and not name.startswith("beta_controls"):
            point[name] = 0.83
        elif name.startswith("roi_"):
            point[name] = 1.4
        elif shape:
            point[name] = rs.standard_normal(shape) * 0.3
        else:
            point[name] = float(rs.standard_normal() * 0.3)
    return point


def _graph_mu(model, point) -> np.ndarray:
    """``mu``, reconstructed from the component Deterministics that compose it."""
    free = {rv.name: rv for rv in model.free_RVs}
    repl = {
        free[k]: pt.constant(np.asarray(v, dtype="float64"), name=k)
        for k, v in point.items()
    }
    total = 0.0
    for comp in (
        "intercept_component",
        "trend_component",
        "seasonality_component",
        "geo_component",
        "product_component",
        "media_total",
        "controls_total",
    ):
        total = total + graph_replace([model[comp]], repl, strict=False)[0].eval()
    return np.asarray(total)


def _theta(design, mmm, point) -> np.ndarray:
    """The parameter vector matching the fixed point, per column."""
    theta = np.zeros(design.n_params)
    for i, col in enumerate(design.columns):
        param, pos = design.param_map[i]
        if param == "intercept":
            # trend_m is folded into the intercept (exactly collinear).
            theta[i] = float(point["intercept"]) + float(point.get("trend_m", 0.0))
        elif param in ("geo_effect", "product_effect"):
            level = param.split("_")[0]
            theta[i] = float(
                np.asarray(point[f"{level}_sigma"])
                * np.atleast_1d(point[f"{level}_offset"])[pos]
            )
        elif param == "spline_coef":
            composite = float(point["spline_scale"]) * np.cumsum(
                np.atleast_1d(point["spline_coef_raw"])
            )
            theta[i] = composite[pos]
        elif param.startswith("beta_") and param != "beta_controls":
            ch = param[len("beta_") :]
            scale = design.roi_scale.get(ch, 1.0)
            if f"roi_{ch}" in point:
                # Design column is c_c * sat, so the coefficient IS the ROI.
                theta[i] = float(point[f"roi_{ch}"])
            else:
                theta[i] = float(point[param]) / (scale if scale else 1.0)
        else:
            value = point[param]
            theta[i] = (
                float(np.atleast_1d(value)[pos]) if np.ndim(value) else float(value)
            )
    return theta


def _assert_equivalent(panel, model_config, trend_config, adstock, sat) -> int:
    mff = _configure(panel, adstock, sat)
    mmm = BayesianMMM(panel, model_config, trend_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = mmm.model
    point = _fixed_point(graph)

    design = build_design_matrix(
        panel,
        alpha={c: dict(_ADSTOCK[adstock][1]) for c in CHANNELS},
        lam={c: dict(_SATURATION[sat][1]) for c in CHANNELS},
        model_config=model_config,
        trend_config=trend_config,
    )
    assert mff is panel.config
    got = design.X @ _theta(design, mmm, point)
    err = np.abs(_graph_mu(graph, point) - got).max()
    assert err < TOL, f"design/graph drift {err:.3e} (tolerance {TOL:.0e})"
    return design.n_params


@pytest.mark.parametrize("adstock", sorted(_ADSTOCK))
@pytest.mark.parametrize("sat", sorted(_SATURATION, key=lambda s: s.value))
def test_national_every_transform_pair(adstock, sat):
    """All three adstock families against all five saturation families."""
    _assert_equivalent(
        _panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR), adstock, sat
    )


@pytest.mark.parametrize(
    "trend",
    [TrendType.NONE, TrendType.LINEAR, TrendType.PIECEWISE, TrendType.SPLINE],
)
def test_every_linear_trend_family(trend):
    _assert_equivalent(
        _panel(),
        ModelConfig(),
        TrendConfig(type=trend),
        "geometric",
        SaturationType.LOGISTIC,
    )


def test_geo_panel_uses_per_cell_carryover():
    """Adstocking the flat stacked series would bleed across geographies."""
    _assert_equivalent(
        _panel(geos=["North", "South", "West"]),
        ModelConfig(),
        TrendConfig(type=TrendType.LINEAR),
        "geometric",
        SaturationType.LOGISTIC,
    )


def test_roi_parameterized_media_prior_mode():
    """`media_prior_mode="roi"` is the agent default; the column is rescaled."""
    _assert_equivalent(
        _panel(),
        ModelConfig(media_prior_mode="roi"),
        TrendConfig(type=TrendType.LINEAR),
        "geometric",
        SaturationType.LOGISTIC,
    )


def test_roi_mode_solves_for_roi_not_beta():
    """In ROI space the solved coefficient is the channel's ROI, so a single
    penalty means the same thing regardless of the channel's spend."""
    panel = _panel()
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    design = build_design_matrix(
        panel,
        alpha={c: {"alpha": 0.55} for c in CHANNELS},
        lam={c: {"sat_lam": 2.7} for c in CHANNELS},
        model_config=ModelConfig(media_prior_mode="roi"),
        trend_config=TrendConfig(type=TrendType.LINEAR),
    )
    assert set(design.roi_scale) == set(CHANNELS)
    assert all(v != 1.0 for v in design.roi_scale.values())


def test_penalty_mask_leaves_structure_alone():
    """Intercept, trend and seasonality are structural; media/controls shrink.

    The geo case matters most: the graph has no per-geo intercept, so penalized
    dummies against an unpenalized intercept is what identifies the split.
    """
    panel = _panel(geos=["North", "South"])
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    design = build_design_matrix(
        panel,
        alpha={c: {"alpha": 0.55} for c in CHANNELS},
        lam={c: {"sat_lam": 2.7} for c in CHANNELS},
        model_config=ModelConfig(),
        trend_config=TrendConfig(type=TrendType.LINEAR),
    )
    for block in ("intercept", "trend", "seasonality"):
        assert not design.penalize[design.blocks[block]].any(), block
    for block in ("media", "controls", "geo"):
        assert design.penalize[design.blocks[block]].all(), block


def test_gaussian_process_trend_is_refused_by_name():
    """Refusals name the feature rather than dropping the term silently."""
    panel = _panel()
    _configure(panel, "geometric", SaturationType.LOGISTIC)
    with pytest.raises(UnsupportedModelError) as exc:
        build_design_matrix(
            panel,
            alpha={c: {"alpha": 0.55} for c in CHANNELS},
            lam={c: {"sat_lam": 2.7} for c in CHANNELS},
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.GP),
        )
    assert "Gaussian-process trend" in str(exc.value)
    assert exc.value.feature == "Gaussian-process trend"


def test_missing_saturation_parameter_raises():
    panel = _panel()
    _configure(panel, "geometric", SaturationType.HILL)
    with pytest.raises(KeyError, match="sat_slope"):
        build_design_matrix(
            panel,
            alpha={c: {"alpha": 0.55} for c in CHANNELS},
            lam={c: {"sat_half": 0.4} for c in CHANNELS},  # sat_slope omitted
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.LINEAR),
        )
