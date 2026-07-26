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

Two panels, because the point needs both halves:

* **Left — the criterion is flat.** Out-of-sample MAPE against ``sat_lam``. If
  saturation were identified this would be a bowl with a minimum near the truth.
  It is not; a wide band of lambda scores indistinguishably.
* **Right — those are not small differences.** The response curves those same
  near-optimal candidates imply. "Equally good at forecasting" spans curves from
  nearly linear to almost fully saturated by a fifth of max spend — which is the
  difference between "spend more" and "stop spending", read off models the data
  cannot rank.

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


def build() -> Path:
    from mmm_framework.synth.dgp import _LAM

    panel = _panel()
    res = _search(panel)
    truth_score = _score_the_truth(panel)
    lam_true = _LAM[CHANNEL]

    lams = np.array([c.lam[CHANNEL]["sat_lam"] for c in res.candidates])
    scores = np.array([c.score for c in res.candidates])
    near = res.spread(WITHIN)
    near_lams = np.array([c.lam[CHANNEL]["sat_lam"] for c in near])
    cutoff = res.best.score * (1 + WITHIN)
    lo, hi = float(near_lams.min()), float(near_lams.max())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>The criterion is flat in λ</b>",
            "<b>Those are not small differences</b>",
        ),
        horizontal_spacing=0.11,
    )

    # -- left: out-of-sample error against lambda ---------------------------
    fig.add_trace(
        go.Scatter(
            x=lams,
            y=scores,
            mode="markers",
            marker=dict(size=7, color="rgba(120,120,115,0.22)", line=dict(width=0)),
            name=f"candidate ({len(lams)})",
            hovertemplate="λ %{x:.2f}<br>MAPE %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # The lower envelope is what makes "flat" legible: the best score reachable
    # at each λ. A bowl with a minimum near the truth would mean λ is identified.
    edges = np.linspace(lams.min(), lams.max(), 11)
    centres, floor = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        sel = (lams >= a) & (lams < b if b < edges[-1] else lams <= b)
        if sel.sum():
            centres.append((a + b) / 2)
            floor.append(scores[sel].min())
    fig.add_trace(
        go.Scatter(
            x=centres,
            y=floor,
            mode="lines",
            line=dict(width=2.5, color=INK_MUTED),
            name="best reachable at each λ",
            hovertemplate="λ ≈ %{x:.2f}<br>best MAPE %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=cutoff,
        line=dict(width=1.5, color=INK_MUTED, dash="dot"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[lam_true],
            y=[truth_score],
            mode="markers",
            marker=dict(
                size=15,
                color=TRUTH,
                symbol="diamond",
                line=dict(width=2, color=SURFACE),
            ),
            name=f"planted truth (λ={lam_true})",
            legendgroup="truth",
            showlegend=False,
            hovertemplate="planted λ %{x:.2f}<br>MAPE %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[res.best.lam[CHANNEL]["sat_lam"]],
            y=[res.best.score],
            mode="markers",
            marker=dict(
                size=15, color=WINNER, symbol="star", line=dict(width=2, color=SURFACE)
            ),
            name="search winner",
            legendgroup="winner",
            showlegend=False,
            hovertemplate="winner λ %{x:.2f}<br>MAPE %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=float(lams.max()),
        y=cutoff,
        text=f"<b>within 10% of best — λ spans {lo:.2f} to {hi:.2f}</b>",
        showarrow=False,
        yshift=11,
        xanchor="right",
        font=dict(size=12, color=INK),
        row=1,
        col=1,
    )

    # -- right: the response curves those lambdas imply ---------------------
    x = np.linspace(0, 1, 200)
    for i, lam in enumerate(np.sort(near_lams)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=1 - np.exp(-lam * x),
                mode="lines",
                line=dict(width=1.5, color=ENSEMBLE),
                name="near-optimal candidates",
                legendgroup="near",
                showlegend=(i == 0),
                hovertemplate="λ %{text}<extra></extra>",
                text=[f"{lam:.2f}"] * len(x),
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1 - np.exp(-lam_true * x),
            mode="lines",
            line=dict(width=3, color=TRUTH),
            name=f"planted truth (λ={lam_true})",
            legendgroup="truth",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1 - np.exp(-res.best.lam[CHANNEL]["sat_lam"] * x),
            mode="lines",
            line=dict(width=3, color=WINNER, dash="dash"),
            name=f"search winner (λ={res.best.lam[CHANNEL]['sat_lam']:.2f})",
            legendgroup="winner",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text=f"saturation λ for {CHANNEL}",
        row=1,
        col=1,
        gridcolor=GRID,
        zeroline=False,
        title_font=dict(size=13, color=INK_MUTED),
        tickfont=dict(size=12, color=INK_MUTED),
    )
    fig.update_yaxes(
        title_text="rolling-origin out-of-sample MAPE",
        row=1,
        col=1,
        gridcolor=GRID,
        zeroline=False,
        title_font=dict(size=13, color=INK_MUTED),
        tickfont=dict(size=12, color=INK_MUTED),
    )
    fig.update_xaxes(
        title_text="spend (share of channel maximum)",
        row=1,
        col=2,
        gridcolor=GRID,
        zeroline=False,
        title_font=dict(size=13, color=INK_MUTED),
        tickfont=dict(size=12, color=INK_MUTED),
    )
    fig.update_yaxes(
        title_text="saturated response",
        row=1,
        col=2,
        gridcolor=GRID,
        zeroline=False,
        title_font=dict(size=13, color=INK_MUTED),
        tickfont=dict(size=12, color=INK_MUTED),
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Predictive error does not identify saturation</b><br>"
                "<span style='font-size:13px'>"
                f"{BUDGET} candidates on <i>make_clean</i>, a world the model represents exactly. "
                "Carryover is recovered; saturation is not.</span>"
            ),
            font=dict(size=19, color=INK),
            x=0.012,
            xanchor="left",
            y=0.965,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK),
        width=1180,
        height=520,
        margin=dict(t=118, b=64, l=74, r=26),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            x=0,
            font=dict(size=12, color=INK_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=14, color=INK)
        ann.x = ann.x - 0.045
        ann.xanchor = "left"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT), scale=2)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  candidates      : {len(lams)}")
    print(f"  best MAPE       : {res.best.score:.5f}  (λ={res.best.lam[CHANNEL]['sat_lam']:.2f})")
    print(f"  planted truth   : {truth_score:.5f}  (λ={lam_true})")
    print(f"  within {int(WITHIN*100)}%      : {len(near)} candidates, λ = {lo:.2f} … {hi:.2f}")
    return OUT


if __name__ == "__main__":
    build()
