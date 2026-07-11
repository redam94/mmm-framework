"""Aurora Coffee Co. — the shared synthetic world for the showcase notebooks.

A single, deliberately-structured dataset anchors every notebook so the story
carries across chapters. Aurora is a fast-growing direct-to-consumer coffee
brand with two product lines (**Original** ground/whole-bean and **Cold Brew**
ready-to-drink) and four paid media channels (**TV, Search, Social, Display**).

The data is generated with the exact pathologies a *causal* MMM exists to handle,
so each capability has something real to find:

* **Unobserved demand confounding.** A latent ``demand`` signal drives *both*
  spend (the team bids harder on Search/Social when demand is high) *and* sales.
  Naively, the demand-chasing channels look far more effective than they are. A
  noisy observable proxy (``category_demand_index``) is the control that closes
  the back-door — the DAG tells you to include it.
* **Mediation.** TV and Display barely sell anything directly; they build
  **awareness**, and awareness drives sales. Their *total* effect is real but
  mostly *indirect* — invisible to a model that can't decompose it.
* **Cannibalization.** Cold Brew steals from Original in summer — a negative
  cross-effect between two correlated outcomes.

Every media effect is generated from explicit terms, so the *true* incremental
contribution and ROAS of each channel are known and stored on the returned
:class:`AuroraData` — letting the notebooks check what the model recovers.

Usage::

    from aurora import generate_aurora, CHANNELS, PALETTE
    aurora = generate_aurora()
    panel = aurora.base_panel()          # a ready-to-fit PanelDataset
    X, outcomes = aurora.extension_inputs()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from mmm_framework.data_loader import PanelDataset

# ---------------------------------------------------------------------------
# Brand identity (shared palette so the notebooks look like one deck)
# ---------------------------------------------------------------------------

CHANNELS = ["TV", "Search", "Social", "Display"]
PRODUCTS = ["Original", "Cold Brew"]

PALETTE = {
    "ink": "#2b2118",          # near-black espresso
    "accent": "#b5651d",       # roasted amber (primary)
    "crema": "#c8a26a",        # latte
    "leaf": "#3f7d5e",         # green (good / confounder)
    "berry": "#a63a50",        # red (bad control / mediator)
    "sky": "#3b6ea5",          # blue (precision control)
    "amber": "#d98a2b",        # orange (collider / caution)
    "muted": "#8a8079",
}
CHANNEL_COLORS = {
    "TV": "#b5651d",
    "Search": "#3b6ea5",
    "Social": "#a63a50",
    "Display": "#3f7d5e",
}

# True (generative) carryover and the role each channel plays in the story.
_ADSTOCK = {"TV": 0.7, "Search": 0.2, "Social": 0.35, "Display": 0.6}
_SATURATION_K = {"TV": 55.0, "Search": 40.0, "Social": 35.0, "Display": 30.0}
# How strongly each channel's spend chases demand (targeting -> endogeneity).
_DEMAND_CHASING = {"TV": 0.10, "Search": 0.85, "Social": 0.55, "Display": 0.15}
# Direct revenue coefficients. TV/Display are deliberately small: they work via
# awareness. Search/Social are direct-response but truly weak — they only *look*
# strong because their spend chases demand (the confounding to be undone).
_BETA_DIRECT = {"TV": 3.0, "Search": 49.0, "Social": 32.0, "Display": 4.0}
_GAMMA_AWARENESS = 5.4    # revenue per point of media-driven awareness lift
_GAMMA_DEMAND = 60.0      # how hard latent demand pushes revenue (the confounding)


# ---------------------------------------------------------------------------
# Small numpy transforms (mirror the framework's, kept dependency-free here)
# ---------------------------------------------------------------------------


def _geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    """Causal geometric carryover: y[t] = sum_k alpha^k x[t-k]."""
    weights = alpha ** np.arange(l_max)
    weights = weights / weights.sum()
    padded = np.concatenate([np.zeros(l_max - 1), x])
    return np.array([np.dot(padded[t : t + l_max][::-1], weights) for t in range(len(x))])


def _hill(x: np.ndarray, half: float) -> np.ndarray:
    """Diminishing-returns saturation in [0, 1): x / (x + half)."""
    return x / (x + half)


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


@dataclass
class AuroraData:
    """The generated Aurora world plus the ground truth used to grade models."""

    weeks: pd.DatetimeIndex
    spend: pd.DataFrame                  # columns = CHANNELS, weekly $000s
    category_demand_index: np.ndarray    # observable proxy for latent demand
    price: np.ndarray                    # avg unit price (control)
    demand: np.ndarray                   # LATENT confounder (hidden in real life)
    awareness: np.ndarray                # mediator, 0-100 (true)
    awareness_survey: np.ndarray         # mediator as partially-observed survey (NaN gaps)
    sales_original: np.ndarray
    sales_coldbrew: np.ndarray
    sales_total: np.ndarray
    true_contribution: pd.Series         # per-channel total incremental sales (units 000s)
    true_roas: pd.Series                 # per-channel total-effect ROAS
    true_mediated_share: pd.Series       # per-channel share of effect flowing via awareness
    notes: dict = field(default_factory=dict)

    # -- tidy frames ------------------------------------------------------

    def frame(self) -> pd.DataFrame:
        """One wide weekly DataFrame (spend + drivers + outcomes)."""
        df = self.spend.copy()
        df.insert(0, "week", self.weeks)
        df["category_demand_index"] = self.category_demand_index
        df["price"] = self.price
        df["awareness"] = self.awareness
        df["sales_original"] = self.sales_original
        df["sales_coldbrew"] = self.sales_coldbrew
        df["sales_total"] = self.sales_total
        return df.set_index("week")

    # -- framework adapters (lazy imports so importing aurora stays light) -

    def base_panel(self, control_demand: bool = True) -> "PanelDataset":
        """A ready-to-fit single-KPI :class:`PanelDataset` (KPI = total sales).

        With ``control_demand`` the observable ``category_demand_index`` proxy is
        included as a control (closing the demand back-door); set it ``False`` to
        reproduce the confounded, demand-blind model for contrast.
        """
        from mmm_framework.config import (
            ControlVariableConfig,
            DimensionType,
            KPIConfig,
            MediaChannelConfig,
            MFFConfig,
        )
        from mmm_framework.data_loader import PanelCoordinates, PanelDataset

        controls = ["Price"] + (["CategoryDemand"] if control_demand else [])
        coords = PanelCoordinates(
            periods=self.weeks,
            geographies=None,
            products=None,
            channels=list(CHANNELS),
            controls=controls,
        )
        ctrl_data = {"Price": self.price}
        if control_demand:
            ctrl_data["CategoryDemand"] = self.category_demand_index
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
                for c in CHANNELS
            ],
            controls=[
                ControlVariableConfig(name=c, dimensions=[DimensionType.PERIOD])
                for c in controls
            ],
        )
        return PanelDataset(
            y=pd.Series(self.sales_total, name="Sales"),
            X_media=self.spend.copy(),
            X_controls=pd.DataFrame(ctrl_data),
            coords=coords,
            index=self.weeks,
            config=config,
        )

    def media_matrix(self) -> np.ndarray:
        """``(n_weeks, n_channels)`` spend array for the extension models."""
        return self.spend[CHANNELS].to_numpy(dtype=float)

    def extension_inputs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """``(X_media, outcome_data)`` for MultivariateMMM / CombinedMMM."""
        outcomes = {
            "sales_original": self.sales_original,
            "sales_coldbrew": self.sales_coldbrew,
        }
        return self.media_matrix(), outcomes


def generate_aurora(seed: int = 7, n_weeks: int = 104) -> AuroraData:
    """Generate the Aurora world (≈2 years of weekly data)."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    # Seasonality: coffee peaks in winter; a separate summer signal for Cold Brew.
    yearly = np.sin(2 * np.pi * (t / 52.0))
    winter = np.cos(2 * np.pi * (t / 52.0))            # +1 in Jan, -1 in Jul
    summer = -winter                                    # +1 in Jul
    trend = t / n_weeks                                 # brand grows over two years

    # Latent demand (the confounder): seasonal + growth + persistent noise.
    demand_noise = rng.normal(0, 0.18, n_weeks)
    for k in range(1, n_weeks):                         # AR(1) persistence
        demand_noise[k] += 0.5 * demand_noise[k - 1]
    demand = 1.0 + 0.35 * winter + 0.45 * trend + demand_noise
    demand_c = demand - demand.mean()
    category_demand_index = 100 * demand + rng.normal(0, 4, n_weeks)  # observable proxy

    # Spend: a planned base + demand-chasing (targeting) + promo noise.
    base_spend = {"TV": 55.0, "Search": 30.0, "Social": 28.0, "Display": 26.0}
    spend = {}
    for c in CHANNELS:
        plan = base_spend[c] * (1.0 + 0.15 * np.sin(2 * np.pi * (t / 26.0 + rng.random())))
        chase = _DEMAND_CHASING[c] * base_spend[c] * demand_c
        promo = np.abs(rng.normal(0, 0.08 * base_spend[c], n_weeks))
        spend[c] = np.clip(plan + chase + promo, 1.0, None)
    spend = pd.DataFrame(spend, columns=CHANNELS)

    # Adstock + saturation per channel (the "media response" the model must learn).
    sat = {}
    for c in CHANNELS:
        ad = _geometric_adstock(spend[c].to_numpy(), _ADSTOCK[c])
        sat[c] = _hill(ad, _SATURATION_K[c])

    # Awareness mediator: built by the brand channels (TV, Display).
    aware_tv = 45.0 * sat["TV"]
    aware_display = 22.0 * sat["Display"]
    awareness = np.clip(
        35.0 + aware_tv + aware_display + 4.0 * winter + rng.normal(0, 2.0, n_weeks),
        0, 100,
    )
    awareness_lift = aware_tv + aware_display            # the media-driven part
    survey = awareness.copy()                            # partially observed (monthly)
    mask = np.ones(n_weeks, bool)
    mask[::4] = False
    survey[mask] = np.nan

    # Price control (promo weeks dip price), and its (negative) revenue effect.
    price = 12.0 + 0.5 * winter - 0.8 * (rng.random(n_weeks) < 0.15)
    price_effect = -28.0 * (price - price.mean())

    # Channel sales contributions (the GROUND TRUTH).
    contrib = {
        "Search": _BETA_DIRECT["Search"] * sat["Search"],
        "Social": _BETA_DIRECT["Social"] * sat["Social"],
        # TV/Display: small direct + the awareness they create * the awareness->sales rate
        "TV": _BETA_DIRECT["TV"] * sat["TV"] + _GAMMA_AWARENESS * aware_tv,
        "Display": _BETA_DIRECT["Display"] * sat["Display"] + _GAMMA_AWARENESS * aware_display,
    }
    media_sales = sum(contrib.values())

    baseline = 560.0 + 120.0 * trend + 55.0 * winter + _GAMMA_DEMAND * demand_c + price_effect
    common_noise = rng.normal(0, 22.0, n_weeks)
    sales_total = np.clip(baseline + media_sales + common_noise, 1.0, None)

    # Split into two products (revenue $000s): Cold Brew skews summer and
    # cannibalizes Original.
    coldbrew_share = np.clip(0.30 + 0.18 * summer, 0.12, 0.55)
    cannibal = 40.0 * np.clip(summer, 0, None)          # cold-brew pressure on original
    cb_noise = rng.normal(0, 14.0, n_weeks)
    sales_coldbrew = np.clip(
        coldbrew_share * sales_total + 0.6 * cannibal + cb_noise, 1.0, None
    )
    sales_original = np.clip(sales_total - sales_coldbrew, 1.0, None)

    # Ground-truth totals over the window.
    true_contribution = pd.Series(
        {c: float(np.sum(contrib[c])) for c in CHANNELS}, name="true_contribution"
    )
    true_roas = pd.Series(
        {c: float(np.sum(contrib[c]) / np.sum(spend[c])) for c in CHANNELS},
        name="true_roas",
    )
    mediated = {
        "TV": float(_GAMMA_AWARENESS * np.sum(aware_tv) / np.sum(contrib["TV"])),
        "Display": float(
            _GAMMA_AWARENESS * np.sum(aware_display) / np.sum(contrib["Display"])
        ),
        "Search": 0.0,
        "Social": 0.0,
    }
    true_mediated_share = pd.Series(mediated, name="true_mediated_share")

    return AuroraData(
        weeks=weeks,
        spend=spend,
        category_demand_index=category_demand_index,
        price=price,
        demand=demand,
        awareness=awareness,
        awareness_survey=survey,
        sales_original=sales_original,
        sales_coldbrew=sales_coldbrew,
        sales_total=sales_total,
        true_contribution=true_contribution,
        true_roas=true_roas,
        true_mediated_share=true_mediated_share,
        notes={
            "awareness_lift": awareness_lift,
            "story": "Aurora Coffee Co. — demand confounds spend↔sales; TV/Display "
            "work via awareness; Cold Brew cannibalizes Original in summer.",
        },
    )


__all__ = [
    "AuroraData",
    "generate_aurora",
    "CHANNELS",
    "PRODUCTS",
    "PALETTE",
    "CHANNEL_COLORS",
]
