"""Shared world / fit / palette for the **Experimental Measurement Lifecycle**
notebook series (``lifecycle_00`` .. ``lifecycle_06``).

One synthetic brand — *Northwind Outfitters* — carried through the framework's
own adaptive measurement loop:

    T0 fit  ->  T1 prioritize (EIG/EVOI)  ->  T2 design (economics + Pareto)
            ->  T3 calibrate  ->  T4 allocate  ->  T5 re-evaluate (decay)  -> T1 ...

Every notebook imports this module so the series looks and behaves like *one
deck*: same channels, same palette, one real posterior. The fit is done **once**
and cached to disk (``MMMSerializer`` + a cloudpickled panel); later notebooks
(and later ``nbconvert`` processes) reload it in a few seconds instead of
re-sampling. Delete the cache dir to force a clean re-fit.

The data is the ``synth`` ``"clean"`` scenario: the model's *exact* generative
family, so the true per-channel ROAS is known (a sealed answer key the analyst
never sees) and recovery is honest. See ``mmm_framework.synth.mff.generate_mff``.

Convention (matches ``nbs/aurora.py`` / ``nbs/charts_src.py``): a plain module
imported by the baked notebooks; bake with ``PYTHONPATH=..`` so both this module
and the installed ``mmm_framework`` resolve.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import json
import os
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Brand + palette
# ---------------------------------------------------------------------------
BRAND = "Northwind Outfitters"
TAGLINE = "a national outdoor-apparel brand"
KPI = "Sales"
CONTROL = "Price"
SEED = 42

# The four channels of the "clean" world, in a stable order.
CHANNELS = ["TV", "Search", "Social", "Display"]

# One palette shared by every plot in every notebook.
PALETTE = {
    "TV": "#3b6fb6",
    "Search": "#d98c3f",
    "Social": "#5a9e6f",
    "Display": "#b15a7a",
}
COLORS = [PALETTE[c] for c in CHANNELS]

# Editorial ink / accents (kept consistent across the series).
INK = "#2b2f36"
MUTED = "#9aa7b0"
GOOD = "#4a8d57"
BAD = "#c0504d"
GRID = "#e7ebee"

# The channel the loop carries end-to-end: the T1 EIG/EVOI winner (highest
# EVOI -> the uncertainty that most swings the budget decision). nb02 ASSERTS
# that the live priority grid still ranks this channel #1, so if the world ever
# shifts the bake fails loudly instead of silently drifting.
FOCUS_CHANNEL = "Display"

# The (simulated) readout precision used when we fold the experiment into the
# model in T3 — a tight, well-powered geo holdout on the focus channel.
READOUT_SE = 0.05

# ---------------------------------------------------------------------------
# Cache location (regenerable; delete to force a clean re-fit)
# ---------------------------------------------------------------------------
CACHE = Path(
    os.environ.get(
        "MMM_LIFECYCLE_CACHE", Path(tempfile.gettempdir()) / "mmm_lifecycle_cache"
    )
)


def dollars(x: float, unit: str = "") -> str:
    """A readable money string for the dataset's native ($000s) spend units."""
    return f"${x:,.0f}{unit}"


def channel_colors(channels=None) -> list[str]:
    channels = channels or CHANNELS
    return [PALETTE.get(c, MUTED) for c in channels]


def style(fig, *, height: int = 440, title: str | None = None, **kwargs):
    """Apply the series' uniform plotly styling to a figure and return it."""
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=30, t=70 if title else 30, b=80),
        font=dict(color=INK),
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# The world + config
# ---------------------------------------------------------------------------
def _quiet_pymc():
    """Silence pymc/arviz's post-fit advisory chatter (the "recommend >=4 chains"
    and "rhat > 1.01" lines) so an in-notebook refit leaves clean output. These
    live on child loggers that set their own level at import, so the parent-level
    silence in the notebook setup cell doesn't reach them — set them here, right
    before we sample."""
    import logging as _logging

    for _n in (
        "pymc",
        "pymc.sampling",
        "pymc.sampling.mcmc",
        "pymc.stats",
        "pymc.stats.convergence",
        "arviz",
    ):
        _logging.getLogger(_n).setLevel(_logging.ERROR)


def _build_world():
    from mmm_framework.synth.mff import generate_mff

    mff_df, truth = generate_mff("clean", seed=SEED)
    return mff_df, truth


def _build_config(channels):
    from mmm_framework import (
        ControlVariableConfigBuilder,
        KPIConfigBuilder,
        MediaChannelConfigBuilder,
        MFFConfigBuilder,
    )

    cfg = MFFConfigBuilder().with_kpi_builder(KPIConfigBuilder(KPI).national())
    for c in channels:
        cfg = cfg.add_media_builder(
            MediaChannelConfigBuilder(c)
            .national()
            .with_logistic_saturation()
            .with_geometric_adstock()
        )
    cfg = cfg.add_control_builder(
        ControlVariableConfigBuilder(CONTROL).national().allow_negative()
    )
    return cfg.weekly().build()


def structural_truth():
    """The hidden structural answer key (adstock / saturation / beta) for the
    'clean' world — not in the ``generate_mff`` truth dict, but module constants
    on the DGP. Used only to grade recovery."""
    from mmm_framework.synth import dgp

    return dict(
        adstock_alpha=dict(dgp._ALPHA),
        saturation_lambda=dict(dgp._LAM),
        amplitude_beta=dict(dgp._AMP),
    )


# ---------------------------------------------------------------------------
# Baseline fit (T0) — fit once, cache, reload
# ---------------------------------------------------------------------------
def _baseline_dir() -> Path:
    return CACHE / "baseline_model"


def _cache_complete() -> bool:
    return (
        _baseline_dir().exists()
        and (CACHE / "panel.pkl").exists()
        and (CACHE / "truth.json").exists()
        and (CACHE / "mff_df.parquet").exists()
    )


def fit_baseline(verbose: bool = True):
    """Fit-or-load the baseline national MMM (T0).

    Returns a dict: ``model, panel, truth, mff_df, mff_config``. The fit is a
    fast-but-real numpyro NUTS posterior (2 chains x 300 draws, ~10s) — enough
    draws for the response-curve / EIG / EVOI machinery downstream.
    """
    from mmm_framework.serialization import MMMSerializer

    if _cache_complete():
        panel = cloudpickle.load(open(CACHE / "panel.pkl", "rb"))
        truth = json.load(open(CACHE / "truth.json"))
        mff_df = pd.read_parquet(CACHE / "mff_df.parquet")
        mff_config = cloudpickle.load(open(CACHE / "mff_config.pkl", "rb"))
        model = MMMSerializer.load(
            str(_baseline_dir()), panel=panel, rebuild_model=True
        )
        if verbose:
            print(f"[{BRAND}] loaded cached baseline fit from {CACHE}")
        return dict(
            model=model, panel=panel, truth=truth, mff_df=mff_df, mff_config=mff_config
        )

    # Cold: build the world, fit, and cache.
    from mmm_framework import (
        BayesianMMM,
        ModelConfigBuilder,
        TrendConfigBuilder,
        load_mff,
    )

    if verbose:
        print(f"[{BRAND}] no cache — fitting the baseline MMM (~10s) ...")
    mff_df, truth = _build_world()
    mff_config = _build_config(truth["channels"])
    panel = load_mff(mff_df, mff_config)

    model_config = (
        ModelConfigBuilder()
        .bayesian_numpyro()
        .with_chains(2)
        .with_draws(300)
        .with_tune(300)
        .with_target_accept(0.9)
        .additive()
        .build()
    )
    trend_config = TrendConfigBuilder().linear().build()
    model = BayesianMMM(panel=panel, model_config=model_config, trend_config=trend_config)
    _quiet_pymc()
    model.fit(random_seed=SEED, progressbar=False)

    CACHE.mkdir(parents=True, exist_ok=True)
    MMMSerializer.save(model, str(_baseline_dir()), save_trace=True, compress=True)
    mff_df.to_parquet(CACHE / "mff_df.parquet")
    json.dump(truth, open(CACHE / "truth.json", "w"), indent=2)
    cloudpickle.dump(panel, open(CACHE / "panel.pkl", "wb"))
    cloudpickle.dump(mff_config, open(CACHE / "mff_config.pkl", "wb"))
    if verbose:
        print(f"[{BRAND}] baseline fit cached to {CACHE}")
    return dict(
        model=model, panel=panel, truth=truth, mff_df=mff_df, mff_config=mff_config
    )


def dataset_period(mff_df) -> tuple[str, str]:
    """(first, last) Period label of the panel — the full experiment window."""
    periods = sorted(pd.Series(mff_df["Period"].unique()).astype(str))
    return periods[0], periods[-1]


def national_csv() -> str:
    """Materialize the spine data as an MFF-long CSV (``planning.design`` reads
    CSV, not parquet). Returns the path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "mff.csv"
    if not p.exists():
        b = fit_baseline(verbose=False)
        b["mff_df"].to_csv(p, index=False)
    return str(p)


# The same brand at DMA grain — used ONLY for the geo-holdout DESIGN showcase in
# T2 (pure-pandas planning.design; no fit). Its per-geo truth is illustrative and
# is NOT cross-referenced with the national spine's incrementality answer key.
GEO_DMAS = [
    "Seattle", "Denver", "Boston", "Atlanta", "Chicago",
    "Phoenix", "Portland", "Dallas", "Miami", "Minneapolis",
]


def geo_csv(seed: int = 7):
    """The brand's 10-market DMA panel as an MFF-long CSV, for the geo-holdout
    design demo. Returns ``(path, truth)``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "geo_brand.csv"
    tp = CACHE / "geo_truth.json"
    if p.exists() and tp.exists():
        return str(p), json.load(open(tp))
    from mmm_framework.synth.mff import generate_mff

    df, truth = generate_mff("clean", geographies=GEO_DMAS, seed=seed)
    df.to_csv(p, index=False)
    json.dump(truth, open(tp, "w"), indent=2)
    return str(p), truth


# ---------------------------------------------------------------------------
# Calibrated fit (T3) — baseline + one in-graph experiment likelihood, refit
# ---------------------------------------------------------------------------
def build_readout(truth, channel: str = FOCUS_CHANNEL, se: float = READOUT_SE, period=None):
    """The (simulated) experiment readout as an in-graph ``ExperimentMeasurement``
    on the channel's ROAS estimand. The measured value is the *sealed truth* —
    i.e. a well-run experiment reveals the real return."""
    from mmm_framework.calibration import ExperimentEstimand, ExperimentMeasurement

    if period is None:
        b = fit_baseline(verbose=False)
        period = dataset_period(b["mff_df"])
    return ExperimentMeasurement(
        channel=channel,
        test_period=period,
        value=float(truth["true_roas"][channel]),
        se=se,
        estimand=ExperimentEstimand.ROAS,
        distribution="normal",
    )


def fit_calibrated(
    channel: str = FOCUS_CHANNEL,
    se: float = READOUT_SE,
    draws: int = 300,
    tune: int = 300,
    verbose: bool = True,
):
    """Fold one experiment readout into the model (T3) and refit.

    Loads a fresh copy of the baseline, attaches an in-graph ROAS likelihood on
    ``channel`` (Route B — the general likelihood route), and re-samples. The
    returned model has a live PyMC graph (freshly fit), so response curves /
    reallocation (T4) work directly on it. Refit is fast (~a few seconds).
    """
    b = fit_baseline(verbose=False)
    model, truth = b["model"], b["truth"]
    period = dataset_period(b["mff_df"])
    exp = build_readout(truth, channel=channel, se=se, period=period)
    model.add_experiment_calibration([exp])
    _quiet_pymc()
    model.fit(draws=draws, tune=tune, chains=2, random_seed=SEED + 1, progressbar=False)
    if verbose:
        print(f"[{BRAND}] calibrated {channel} on its ROAS readout (value={exp.value:.3f}, se={se})")
    return dict(
        model=model, panel=b["panel"], truth=truth, exp=exp, channel=channel,
        mff_df=b["mff_df"],
    )


__all__ = [
    "BRAND", "TAGLINE", "KPI", "CONTROL", "SEED", "CHANNELS", "PALETTE", "COLORS",
    "INK", "MUTED", "GOOD", "BAD", "GRID", "FOCUS_CHANNEL", "READOUT_SE", "CACHE",
    "dollars", "channel_colors", "style", "structural_truth",
    "fit_baseline", "dataset_period", "national_csv", "geo_csv", "GEO_DMAS",
    "build_readout", "fit_calibrated",
]
