"""Why the saturation parameter is not identified by predictive error.

    uv run --with plotly --with kaleido python nbs/builders/build_saturation_identifiability.py

Writes ``nbs/artifacts/saturation_identifiability.png``.

The finding this figure carries (measured in ``tests/frequentist/test_search.py``,
recorded in ``technical-docs/frequentist-estimation.md`` §4a): when the
frequentist transform search picks per-channel adstock and saturation by
rolling-origin out-of-sample error, **carryover is recovered but saturation is
not**. On ``synth.dgp.make_clean`` — a world whose truth the model represents
exactly — candidates whose out-of-sample error is within 10% of the best disagree
about TV's ``sat_lam`` across nearly the whole search range, while the planted
value is 1.6.

Two panels, showing the same search under two bound regimes:

* **Left — bounding lambda absolutely.** What the search entertains when the
  bound is written in the parameter's own units. Candidates that forecast
  indistinguishably span curves from nearly linear to almost fully saturated by a
  fifth of max spend: the difference between "spend more" and "stop spending",
  read off models the data cannot rank.
* **Right — bounding the elbow to observed spend.** Robyn hard-codes
  ``inflexion = max(x) * gamma`` with ``gamma`` in ``[0.3, 1.0]``; Meridian
  scales its ``ec`` prior to median spend. Media here is already normalized by
  the channel max, so that fraction *is* the half-saturation point, and for
  ``1 - exp(-lam*x)`` it maps to ``lam = ln(2)/gamma``.

Measured over four seeds, that one change cuts mean absolute ``lam`` error from
2.08 to 0.42. **It is containment, not identification** — inside the narrower
bound the near-optimal set still covers essentially the whole window, so the
criterion still cannot order ``lam``. It simply can no longer propose a curve no
analyst would entertain.

Palette: ``#2b6a9e`` / ``#7d9c2a``, snapped from the project's slate/sage brand
hues to the nearest steps that clear the categorical checks (the brand values
themselves read as gray at mark size). Verified with the dataviz validator:
chroma floor, CVD separation (protan dE 25.0), normal-vision floor (dE 25.8) and
3:1 contrast against the light surface all pass.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "nbs" / "artifacts" / "saturation_identifiability.png"

# Validated categorical pair (see module docstring).
TRUTH = "#2b6a9e"
WINNER = "#7d9c2a"
SURFACE = "#fcfcfb"
INK = "#1c1c1a"
INK_MUTED = "#6b6b66"
ENSEMBLE = "rgba(120,120,115,0.32)"  # recessive: many marks, no identity
GRID = "rgba(120,120,115,0.16)"

BUDGET = 256
WITHIN = 0.10
CHANNEL = "TV"

#: The absolute bound the data-anchored one replaced (see the module docstring).
ABSOLUTE_BOUNDS = (0.1, 8.0)


def _panel():
    """`make_clean` as a PanelDataset, through the MFF round trip."""
    from mmm_framework.config import (
        ControlVariableConfig,
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import MFFLoader
    from mmm_framework.synth import generate_mff
    from mmm_framework.synth.dgp import CHANNELS

    df, _ = generate_mff("clean", seed=0)
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
            for c in CHANNELS
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        df.to_csv(fh.name, index=False)
        path = fh.name
    try:
        return MFFLoader(cfg).load(path).build_panel()
    finally:
        Path(path).unlink()


def _search(panel):
    from mmm_framework.config import ModelConfig
    from mmm_framework.frequentist.search import search_transforms
    from mmm_framework.model.trend_config import TrendConfig, TrendType

    return search_transforms(
        panel,
        model_config=ModelConfig(),
        trend_config=TrendConfig(type=TrendType.LINEAR),
        budget=BUDGET,
        horizon=13,
        max_origins=3,
        seed=0,
    )


def _score_the_truth(panel):
    """The planted parameters' own out-of-sample score, for the reference line."""
    import mmm_framework.frequentist.search as search_mod
    from mmm_framework.config import ModelConfig
    from mmm_framework.model.trend_config import TrendConfig, TrendType
    from mmm_framework.synth.dgp import _ALPHA, _LAM

    truth = (
        {ch: {"alpha": _ALPHA[ch]} for ch in _ALPHA},
        {ch: {"sat_lam": _LAM[ch]} for ch in _LAM},
    )
    original = search_mod._candidate_points
    search_mod._candidate_points = lambda mmm, budget, strategy, rng: [truth]
    try:
        res = search_mod.search_transforms(
            panel,
            model_config=ModelConfig(),
            trend_config=TrendConfig(type=TrendType.LINEAR),
            budget=1,
            horizon=13,
            max_origins=3,
            seed=0,
        )
    finally:
        search_mod._candidate_points = original
    return res.best.score


def _run_at(panel, bounds):
    """Search once under a given logistic bound, returning (result, lam error)."""
    import numpy as _np

    import mmm_framework.frequentist.search as search_mod
    from mmm_framework.config.enums import SaturationType as _S
    from mmm_framework.synth.dgp import _LAM

    original = search_mod.SATURATION_BOUNDS[_S.LOGISTIC]
    search_mod.SATURATION_BOUNDS[_S.LOGISTIC] = {"sat_lam": bounds}
    try:
        res = _search(panel)
    finally:
        search_mod.SATURATION_BOUNDS[_S.LOGISTIC] = original
    near = res.spread(WITHIN)
    err = float(
        _np.mean([[abs(c.lam[ch]["sat_lam"] - _LAM[ch]) for ch in _LAM] for c in near])
    )
    return res, near, err


def build() -> Path:
    import mmm_framework.frequentist.search as search_mod
    from mmm_framework.config.enums import SaturationType as _S
    from mmm_framework.synth.dgp import _LAM

    panel = _panel()
    lam_true = _LAM[CHANNEL]
    anchored_bounds = search_mod.SATURATION_BOUNDS[_S.LOGISTIC]["sat_lam"]

    regimes = [
        ("absolute", ABSOLUTE_BOUNDS, "λ bounded absolutely"),
        ("anchored", anchored_bounds, "elbow bounded to observed spend"),
    ]
    runs = {}
    for key, bounds, _ in regimes:
        runs[key] = _run_at(panel, bounds) + (bounds,)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(f"<b>{t}</b>" for _, _, t in regimes),
        horizontal_spacing=0.10,
        shared_yaxes=True,
    )

    x = np.linspace(0, 1, 200)
    for col, (key, _, _) in enumerate(regimes, start=1):
        res, near, err, bounds = runs[key]
        lams = np.sort([c.lam[CHANNEL]["sat_lam"] for c in near])
        for i, lam in enumerate(lams):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=1 - np.exp(-lam * x),
                    mode="lines",
                    line=dict(width=1.4, color=ENSEMBLE),
                    name="candidates the criterion cannot separate",
                    legendgroup="near",
                    showlegend=(i == 0 and col == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=1 - np.exp(-lam_true * x),
                mode="lines",
                line=dict(width=3.5, color=TRUTH),
                name=f"planted truth (λ={lam_true})",
                legendgroup="truth",
                showlegend=(col == 1),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_annotation(
            x=0.03,
            y=0.965,
            xref=f"x{col if col > 1 else ''} domain",
            yref=f"y{col if col > 1 else ''} domain",
            xanchor="left",
            align="left",
            showarrow=False,
            text=(
                f"bound  λ ∈ [{bounds[0]:.2f}, {bounds[1]:.2f}]"
                f"   ({bounds[1] / bounds[0]:.0f}× window)<br>"
                f"near-optimal span  {lams.min():.2f} – {lams.max():.2f}"
                f"   ({len(lams)} of {BUDGET})<br>"
                f"<b>mean |λ error|  {err:.2f}</b>"
            ),
            font=dict(size=12, color=INK_MUTED),
        )

    for col in (1, 2):
        fig.update_xaxes(
            title_text="spend (share of channel maximum)",
            row=1,
            col=col,
            gridcolor=GRID,
            zeroline=False,
            title_font=dict(size=13, color=INK_MUTED),
            tickfont=dict(size=12, color=INK_MUTED),
        )
    fig.update_yaxes(
        title_text="saturated response",
        range=[0, 1.02],
        row=1,
        col=1,
        gridcolor=GRID,
        zeroline=False,
        title_font=dict(size=13, color=INK_MUTED),
        tickfont=dict(size=12, color=INK_MUTED),
    )
    fig.update_yaxes(range=[0, 1.02], row=1, col=2, gridcolor=GRID, zeroline=False)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Bounding the elbow to observed spend contains the damage — "
                "it does not identify the parameter</b><br>"
                "<span style='font-size:13px'>"
                "Response curves the criterion cannot separate, on <i>make_clean</i> "
                "— a world the model represents exactly. Same search, same data; "
                "only the bound differs.</span>"
            ),
            font=dict(size=18, color=INK),
            x=0.012,
            xanchor="left",
            y=0.965,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK),
        width=1180,
        height=530,
        margin=dict(t=118, b=68, l=74, r=26),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            x=0,
            font=dict(size=12, color=INK_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=14, color=INK)
        ann.y = ann.y + 0.012

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT), scale=2)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for key, _, label in regimes:
        res, near, err, bounds = runs[key]
        lams = [c.lam[CHANNEL]["sat_lam"] for c in near]
        print(
            f"  {label:<34} bound [{bounds[0]:.2f},{bounds[1]:.2f}]  "
            f"span {min(lams):.2f}-{max(lams):.2f}  |λ err| {err:.3f}  "
            f"MAPE {res.best.score:.5f}"
        )
    return OUT


if __name__ == "__main__":
    build()
