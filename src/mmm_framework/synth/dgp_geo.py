"""Multi-geography and geography x product synthetic worlds for the MMM.

Companion to :mod:`mmm_framework.synth.dgp` (national worlds). Every world here is a
**balanced panel**: ``n_periods x n_geos (x n_products)`` observations stacked
period-major, exactly the layout the MFF loader produces. Ground truth follows
the same doctrine as ``dgp``: per-channel truth is the counterfactual zero-out
evaluated on the noiseless structural mean — the model's own estimand — and is
recorded **per geography** (and per cell for geo x product worlds), so panel
fits can be graded at the level stakeholders actually read: regional ROI.

The model's hierarchy (``model/base.py``) gives each geography/product an
*additive intercept offset* (``geo_sigma * geo_offset``) while every response
parameter — beta, saturation, adstock — is **global**. Two worlds bracket that
hypothesis space:

* :func:`make_geo_clean` — geo differences are level shifts only: exactly the
  model's family (the panel positive control).
* :func:`make_geo_heterogeneous` — channel *effectiveness* differs by geo and
  budgets chase performance, the way real regional allocation works. One
  pooled beta cannot represent this; per-geo readouts fail silently while the
  national total stays plausible.
* :func:`make_geo_product` — a geo x product positive control (level shifts
  per geo and per product, shared response, product-tilted channel mixes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import dgp
from .dgp import CHANNELS, _geom_adstock, _logistic_sat

GEOS = ["North", "South", "East", "West"]
N_WEEKS_GEO = 130  # 2.5 years weekly per geography
START = dgp.START

# Per-geo baseline level offsets (KPI units) — the model's geo hypothesis.
_GEO_OFFSET = {"North": 60.0, "South": -30.0, "East": 20.0, "West": -50.0}
# Per-geo media budget scale (market size): spend varies across geos.
_GEO_SHARE = {"North": 1.30, "South": 0.70, "East": 1.00, "West": 0.55}

# Channel response (shared across geos in the clean world; the per-geo spend
# scale already makes contributions differ by geo through the curve).
_AMP = {"TV": 65.0, "Search": 45.0, "Social": 40.0, "Display": 28.0}
_BASE_SPEND = dict(dgp._BASE_SPEND)
_ALPHA = dict(dgp._ALPHA)
_LAM = dict(dgp._LAM)
_FLIGHT_PERIOD = dict(dgp._FLIGHT_PERIOD)


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


@dataclass
class GeoScenario:
    """A ready-to-fit panel world plus per-geography causal ground truth."""

    name: str
    violates: str
    description: str
    weeks: pd.DatetimeIndex
    geos: list[str]
    products: list[str] | None
    spend: pd.DataFrame  # MultiIndex (Period, Geography[, Product]) x channels
    y: pd.Series  # same index
    controls: pd.DataFrame  # same index
    true_contribution: pd.Series  # national, per channel
    true_contribution_by_geo: pd.DataFrame  # rows: geo (or "geo|product" cell)
    true_roas_by_geo: pd.DataFrame
    representable: bool = True
    notes: dict = field(default_factory=dict)

    @property
    def channels(self) -> list[str]:
        return list(self.spend.columns)

    @property
    def cells(self) -> list[str]:
        return list(self.true_contribution_by_geo.index)

    @property
    def true_roas(self) -> pd.Series:
        spend_tot = self.spend.sum()
        return pd.Series(
            {c: self.true_contribution[c] / spend_tot[c] for c in self.channels},
            name="true_roas",
        )

    # -- model inputs -------------------------------------------------------

    def panel(self):
        """Build the geo(/product) :class:`PanelDataset` for ``BayesianMMM``."""
        from mmm_framework.config import (
            ControlVariableConfig,
            DimensionType,
            KPIConfig,
            MediaChannelConfig,
            MFFConfig,
        )
        from mmm_framework.data_loader import PanelCoordinates, PanelDataset

        dims = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]
        if self.products:
            dims = dims + [DimensionType.PRODUCT]
        controls = list(self.controls.columns)
        coords = PanelCoordinates(
            periods=self.weeks,
            geographies=list(self.geos),
            products=list(self.products) if self.products else None,
            channels=self.channels,
            controls=controls,
        )
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=dims),
            media_channels=[
                MediaChannelConfig(name=c, dimensions=dims) for c in self.channels
            ],
            controls=[ControlVariableConfig(name=c, dimensions=dims) for c in controls],
        )
        return PanelDataset(
            y=self.y.copy(),
            X_media=self.spend.copy(),
            X_controls=self.controls.copy(),
            coords=coords,
            index=self.spend.index,
            config=config,
        )

    def national_scenario(self) -> dgp.Scenario:
        """Aggregate the panel to one national series (the pre-geo-data view).

        Spend and KPI sum across geographies; the price control averages. The
        causal truth is unchanged (zeroing a channel everywhere is the sum of
        the per-geo zero-outs), but the *aggregated* response is no longer in
        the model's family: a sum of per-geo saturation curves is not a
        saturation curve of the summed spend (Jensen's inequality), so the
        national fit carries a structural aggregation error the panel fit
        does not.
        """
        lvl = "Period"
        spend_nat = self.spend.groupby(level=lvl, sort=True).sum()
        y_nat = self.y.groupby(level=lvl, sort=True).sum()
        ctrl_nat = self.controls.groupby(level=lvl, sort=True).mean()
        spend_nat.index = self.weeks
        ctrl_nat.index = pd.RangeIndex(len(ctrl_nat))
        return dgp.Scenario(
            name=f"{self.name}_national_aggregate",
            violates="aggregation: sum of per-geo saturations is not a "
            "saturation of summed spend",
            description=f"{self.name} summed across {len(self.geos)} geographies.",
            weeks=self.weeks,
            spend=spend_nat.reset_index(drop=True),
            y=pd.Series(y_nat.to_numpy(), index=self.weeks, name="Sales"),
            controls=ctrl_nat,
            true_contribution=self.true_contribution.copy(),
            true_roas=self.true_roas.copy(),
            representable=False,
            notes={"source": self.name, "n_geos": len(self.geos)},
        )

    def geo_scenario(self, geo: str) -> dgp.Scenario:
        """One geography as a standalone national-style scenario.

        Used by the per-geo refit pivot: a single geo's series with that geo's
        own causal truth. Exactly representable for the clean and heterogeneous
        worlds alike (a per-geo fit estimates its own beta).
        """
        mask = self.spend.index.get_level_values("Geography") == geo
        spend_g = self.spend[mask].copy()
        spend_g.index = self.weeks
        y_g = pd.Series(self.y[mask].to_numpy(), index=self.weeks, name="Sales")
        ctrl_g = self.controls[mask].reset_index(drop=True)
        truth_g = self.true_contribution_by_geo.loc[geo]
        return dgp.Scenario(
            name=f"{self.name}_{geo}",
            violates=self.violates,
            description=f"{self.name}: the {geo} geography in isolation.",
            weeks=self.weeks,
            spend=spend_g.reset_index(drop=True),
            y=y_g,
            controls=ctrl_g,
            true_contribution=pd.Series(
                {c: float(truth_g[c]) for c in self.channels}, name="true_contribution"
            ),
            true_roas=pd.Series(
                {c: float(truth_g[c]) / float(spend_g[c].sum()) for c in self.channels},
                name="true_roas",
            ),
            representable=True,
            notes={"source": self.name, "geo": geo},
        )


# ---------------------------------------------------------------------------
# shared ingredients
# ---------------------------------------------------------------------------


def _geo_levels(
    rng: np.random.Generator, geos: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Baseline offset + budget share per geo (defaults for the canonical four,
    seeded draws for custom geography names)."""
    offsets, shares = {}, {}
    for g in geos:
        if g in _GEO_OFFSET:
            offsets[g] = _GEO_OFFSET[g]
            shares[g] = _GEO_SHARE[g]
        else:
            offsets[g] = float(rng.uniform(-60.0, 60.0))
            shares[g] = float(rng.uniform(0.55, 1.30))
    return offsets, shares


def _het_multipliers(
    rng: np.random.Generator, geos: list[str]
) -> dict[str, dict[str, float]]:
    """Per-geo channel-effectiveness multipliers (canonical or seeded draws)."""
    return {
        g: (
            dict(_HET_MULT[g])
            if g in _HET_MULT
            else {c: float(rng.uniform(0.3, 1.8)) for c in CHANNELS}
        )
        for g in geos
    }


def _pulsed_spend(
    rng: np.random.Generator, n: int, scale: float
) -> dict[str, np.ndarray]:
    """Per-channel pulsed spend for one cell (independent phases per cell)."""
    t = np.arange(n)
    out = {}
    for c in CHANNELS:
        phase = rng.random() * 2 * np.pi
        burst = np.clip(np.sin(2 * np.pi * t / _FLIGHT_PERIOD[c] + phase), 0, None)
        idio = rng.lognormal(0.0, 0.25, n)
        level = (0.08 + 1.6 * burst) * idio
        out[c] = np.clip(_BASE_SPEND[c] * scale * level, 0.5, None)
    return out


def _shared_baseline(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Global trend + seasonality shared by every geography (model family)."""
    t = np.arange(n)
    season = 16.0 * np.sin(2 * np.pi * t / 52.0) + 10.0 * np.cos(2 * np.pi * t / 52.0)
    trend = 30.0 * (t / n)
    return 150.0 + trend + season, t


def _stack_index(
    weeks: pd.DatetimeIndex, geos: list[str], products: list[str] | None
) -> pd.MultiIndex:
    """Period-major MultiIndex matching the MFF loader's observation order."""
    if products:
        return pd.MultiIndex.from_product(
            [weeks, geos, products], names=["Period", "Geography", "Product"]
        )
    return pd.MultiIndex.from_product([weeks, geos], names=["Period", "Geography"])


def _media_contribution(
    spend: np.ndarray, maxes: dict[str, float], mult: dict[str, float] | None = None
) -> np.ndarray:
    """One cell's media response time-series (shared family, optional multiplier)."""
    out = np.zeros(spend.shape[0])
    for i, c in enumerate(CHANNELS):
        xn = spend[:, i] / maxes[c]
        m = 1.0 if mult is None else mult[c]
        out = out + m * _AMP[c] * _logistic_sat(_geom_adstock(xn, _ALPHA[c]), _LAM[c])
    return out


def _assemble(
    name: str,
    violates: str,
    description: str,
    weeks: pd.DatetimeIndex,
    geos: list[str],
    products: list[str] | None,
    cell_spend: dict[str, pd.DataFrame],  # cell label -> (n x channels) spend
    cell_mult: dict[str, dict[str, float]] | None,  # cell -> channel -> multiplier
    cell_offset: dict[str, float],  # cell -> baseline level offset
    seed: int,
    *,
    representable: bool,
    noise_sd: float = 12.0,
    notes: dict | None = None,
) -> GeoScenario:
    """Assemble a balanced panel world + per-cell counterfactual ground truth."""
    rng = np.random.default_rng(seed + 1000)
    n = len(weeks)
    cells = list(cell_spend)
    baseline_t, t = _shared_baseline(rng, n)

    # Per-geo price control (shared seasonal core + per-cell wiggle).
    price = {
        cell: 12.0 + 0.5 * np.cos(2 * np.pi * t / 52.0) + rng.normal(0, 0.25, n)
        for cell in cells
    }
    price_effect = {cell: -6.0 * (price[cell] - price[cell].mean()) for cell in cells}

    # Global per-channel normalization max (the model normalizes globally too).
    maxes = {
        c: max(float(cell_spend[cell][c].max()) for cell in cells) for c in CHANNELS
    }

    # Structural mean, noise, truth — per cell.
    truth_rows, roas_rows = {}, {}
    y_cells, mu_cells = {}, {}
    for cell in cells:
        sp = cell_spend[cell].to_numpy(float)
        mult = None if cell_mult is None else cell_mult[cell]
        media = _media_contribution(sp, maxes, mult)
        mu = baseline_t + cell_offset[cell] + price_effect[cell] + media
        noise = rng.normal(0, noise_sd, n)
        y_cells[cell] = np.clip(mu + noise, 1.0, None)
        mu_cells[cell] = mu
        row, roas = {}, {}
        for i, c in enumerate(CHANNELS):
            sp0 = sp.copy()
            sp0[:, i] = 0.0
            contrib = float((media - _media_contribution(sp0, maxes, mult)).sum())
            row[c] = contrib
            roas[c] = contrib / float(cell_spend[cell][c].sum())
        truth_rows[cell] = row
        roas_rows[cell] = roas

    # Stack period-major: row order (week, geo[, product]).
    index = _stack_index(weeks, geos, products)
    n_cells = len(cells)
    spend_stacked = np.zeros((n * n_cells, len(CHANNELS)))
    y_stacked = np.zeros(n * n_cells)
    price_stacked = np.zeros(n * n_cells)
    for j, cell in enumerate(cells):
        # Period-major stacking: cell j occupies positions j, j+n_cells, ...
        idx = np.arange(n) * n_cells + j
        spend_stacked[idx] = cell_spend[cell].to_numpy(float)
        y_stacked[idx] = y_cells[cell]
        price_stacked[idx] = price[cell]

    spend_df = pd.DataFrame(spend_stacked, index=index, columns=CHANNELS)
    y_s = pd.Series(y_stacked, index=index, name="Sales")
    controls_df = pd.DataFrame({"Price": price_stacked}, index=index)

    truth_by_cell = pd.DataFrame(truth_rows).T.loc[cells]
    roas_by_cell = pd.DataFrame(roas_rows).T.loc[cells]
    national = truth_by_cell.sum(axis=0)
    national.name = "true_contribution"

    return GeoScenario(
        name=name,
        violates=violates,
        description=description,
        weeks=weeks,
        geos=geos,
        products=products,
        spend=spend_df,
        y=y_s,
        controls=controls_df,
        true_contribution=national,
        true_contribution_by_geo=truth_by_cell,
        true_roas_by_geo=roas_by_cell,
        representable=representable,
        notes=(notes or {}) | {"maxes": maxes, "geo_offsets": dict(cell_offset)},
    )


# ===========================================================================
# scenario factories
# ===========================================================================


def make_geo_clean(
    seed: int = 20,
    *,
    geos: list[str] | None = None,
    n_weeks: int | None = None,
) -> GeoScenario:
    """PANEL POSITIVE CONTROL: geo differences are level shifts only.

    Four geographies share every response parameter (beta, adstock,
    saturation, trend, seasonality); they differ in baseline level (the
    model's additive geo offset) and in budget scale (market size). This is
    exactly the hierarchy ``BayesianMMM`` fits, so recovery should be clean
    nationally AND per geography.
    """
    rng = np.random.default_rng(seed)
    geos = list(geos) if geos else list(GEOS)
    n = int(n_weeks) if n_weeks else N_WEEKS_GEO
    offsets, shares = _geo_levels(rng, geos)
    weeks = pd.date_range(START, periods=n, freq="W-MON")
    cell_spend = {
        g: pd.DataFrame(_pulsed_spend(rng, n, shares[g]), columns=CHANNELS)
        for g in geos
    }
    return _assemble(
        "geo_clean",
        "",
        f"{len(geos)}-geo panel from the model's exact family: shared response, "
        "additive per-geo level offsets, geo-scaled exogenous budgets.",
        weeks,
        geos,
        None,
        cell_spend,
        None,
        offsets,
        seed,
        representable=True,
        notes={"role": "panel positive control", "geo_share": dict(shares)},
    )


# Per-geo effectiveness multipliers: same creative, very different markets.
# TV is a powerhouse in the North and near-dead in the West; Search inverts.
_HET_MULT = {
    "North": {"TV": 1.8, "Search": 0.6, "Social": 1.3, "Display": 1.0},
    "South": {"TV": 0.9, "Search": 1.5, "Social": 0.6, "Display": 1.4},
    "East": {"TV": 1.0, "Search": 1.2, "Social": 1.2, "Display": 0.5},
    "West": {"TV": 0.3, "Search": 1.7, "Social": 0.9, "Display": 1.1},
}


def make_geo_heterogeneous(
    seed: int = 21,
    *,
    geos: list[str] | None = None,
    n_weeks: int | None = None,
) -> GeoScenario:
    """Per-geo effectiveness + performance-chasing budgets: the panel trap.

    The same channel works differently in different markets (multipliers
    0.3-1.8 on the shared response), and the brand allocates the way real
    brands do: geos where a channel performs get a larger share of its
    budget. The model's hierarchy has one global beta per channel and only
    *intercept* offsets per geo, so per-geo attribution is structurally out
    of reach: the pooled beta lands near a spend-weighted average, every
    geo's readout inherits it, and regional ROI rankings scramble — while
    national totals and every sampler diagnostic stay plausible.
    """
    rng = np.random.default_rng(seed)
    geos = list(geos) if geos else list(GEOS)
    n = int(n_weeks) if n_weeks else N_WEEKS_GEO
    offsets, shares = _geo_levels(rng, geos)
    mults = _het_multipliers(rng, geos)
    weeks = pd.date_range(START, periods=n, freq="W-MON")
    cell_spend = {}
    for g in geos:
        base = _pulsed_spend(rng, n, shares[g])
        # Performance chasing: budget share scales with last year's ROI story.
        chased = {
            c: np.clip(base[c] * (0.45 + 0.75 * mults[g][c]), 0.5, None)
            for c in CHANNELS
        }
        cell_spend[g] = pd.DataFrame(chased, columns=CHANNELS)
    return _assemble(
        "geo_heterogeneous",
        "global media coefficients (per-geo effectiveness homogeneity)",
        "Channel effectiveness varies 0.3-1.8x by geography and budgets chase "
        "performance; the pooled model has one beta per channel and intercept-"
        "only geo offsets.",
        weeks,
        geos,
        None,
        cell_spend,
        mults,
        offsets,
        seed,
        representable=False,
        notes={"effect_multipliers": mults, "geo_share": dict(shares)},
    )


# geo x product: three geographies, two products with tilted channel mixes.
_GP_GEOS = ["North", "South", "West"]
_GP_PRODUCTS = ["Core", "Premium"]
_GP_GEO_OFFSET = {"North": 45.0, "South": -20.0, "West": 5.0}
_GP_PROD_OFFSET = {"Core": 35.0, "Premium": -25.0}
_GP_GEO_SHARE = {"North": 1.25, "South": 0.8, "West": 0.95}
# Channel mix tilt per product (budget allocation, not response).
_GP_MIX = {
    "Core": {"TV": 1.4, "Search": 0.7, "Social": 0.9, "Display": 1.2},
    "Premium": {"TV": 0.6, "Search": 1.5, "Social": 1.3, "Display": 0.7},
}


def make_geo_product(seed: int = 22, *, n_weeks: int | None = None) -> GeoScenario:
    """GEO x PRODUCT POSITIVE CONTROL: six cells, additive offsets, shared response.

    Three geographies x two products. Each cell gets its own budget calendar
    (product lines flight independently and tilt their channel mixes), and the
    baseline level is geo offset + product offset — exactly the model's
    additive geo + product hierarchy. Truth is recorded per cell, so per-cell
    contribution readouts can be graded directly.
    """
    rng = np.random.default_rng(seed)
    n = int(n_weeks) if n_weeks else 104  # 2 years x 6 cells = 624 obs
    weeks = pd.date_range(START, periods=n, freq="W-MON")
    cell_spend = {}
    cell_offset = {}
    for g in _GP_GEOS:
        for p in _GP_PRODUCTS:
            cell = f"{g}|{p}"
            base = _pulsed_spend(rng, n, _GP_GEO_SHARE[g] * 0.6)
            cell_spend[cell] = pd.DataFrame(
                {c: np.clip(base[c] * _GP_MIX[p][c], 0.5, None) for c in CHANNELS},
                columns=CHANNELS,
            )
            cell_offset[cell] = _GP_GEO_OFFSET[g] + _GP_PROD_OFFSET[p]
    sc = _assemble(
        "geo_product",
        "",
        "3 geographies x 2 products from the model's family: additive geo + "
        "product level offsets, shared response, product-tilted channel mixes.",
        weeks,
        _GP_GEOS,
        _GP_PRODUCTS,
        cell_spend,
        None,
        cell_offset,
        seed,
        representable=True,
        notes={"role": "geo x product positive control", "mix": _GP_MIX},
    )
    return sc


SCENARIOS = {
    "geo_clean": make_geo_clean,
    "geo_heterogeneous": make_geo_heterogeneous,
    "geo_product": make_geo_product,
}


def build(
    name: str,
    seed: int | None = None,
    *,
    geos: list[str] | None = None,
    n_weeks: int | None = None,
) -> GeoScenario:
    """Build a geo scenario by name (factory default seed when ``None``)."""
    fn = SCENARIOS[name]
    kwargs: dict = {}
    if n_weeks is not None:
        kwargs["n_weeks"] = int(n_weeks)
    if geos is not None:
        kwargs["geos"] = list(geos)  # geo_product has fixed cells: rejects this
    return fn(**kwargs) if seed is None else fn(seed, **kwargs)


__all__ = ["GeoScenario", "SCENARIOS", "build", "GEOS", "CHANNELS"]
