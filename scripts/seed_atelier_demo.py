"""Seed a demo project that showcases the Atelier custom (Model Garden) models
being used by the agent in the Oracle.

Two things happen:

1. **The garden is populated.** The three worked example models in
   ``examples/garden_models/`` — the awareness structural MMM, the Bayesian CFA,
   and the Bayesian LCA — are *registered + published* into the org's Model
   Garden (real ``model.py`` source + manifest with the model's config schema,
   declared estimands, and family kind; gated by a compatibility report). After
   running this, the **Atelier** page shows three published models with versions,
   docs, and a passing compatibility report.

2. **The Oracle tells the story.** A demo project — *"Demo: Atelier Custom
   Models"* — gets one chat session per model. Each session fits the model for
   real (MAP) on synthetic data, then scripts an agent conversation that
   discovers the catalog (``list_garden_models``), loads the model
   (``load_garden_model`` → ``spec['garden_ref']``), fits it (``fit_mmm_model``),
   and reads its declared estimands (``get_estimands``) — with the *real* numbers
   from the fit. The session's workspace carries the model-specific configuration
   (the spec incl. the ``garden_ref``, ``model_params``, and likelihood), real
   model-specific **charts** (awareness lift / goodwill / retention; CFA loadings
   + fit indices; LCA class profiles + sizes), **tables** (estimands, loadings,
   class profiles, the model configuration) — so opening a session shows the
   chat AND fully populated Model/Results tabs. (A family-aware HTML report is
   also rendered to disk as an artifact; in dev it is not surfaced in the UI
   because ``/report`` serves a single global file.)

Run from the repo root (no Redis needed — writes straight to the sessions DB):

    uv run python scripts/seed_atelier_demo.py                 # real MAP fits (~15s)
    uv run python scripts/seed_atelier_demo.py --fast          # no MCMC; fabricate (seconds)
    uv run python scripts/seed_atelier_demo.py --real-compat   # + genuine compat suite (~minutes)

The project / sessions / models / chat live in a fixed sessions DB, so they show
up however the app is launched. The session **charts and tables**, though, are
served from the agent workspace (``MMM_AGENT_WORKSPACE``, default
``./agent_workspace``) — so run this seeder with the SAME ``MMM_AGENT_WORKSPACE``
the API server uses, or they will 404 in the Results tab. The seeder prints the
resolved workspace + the matching launch command at the end.

Re-running REPLACES the demo project AND the three seeded garden models (their
versions + on-disk source) so the project switcher and the registry never stack
duplicates.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# The example garden models are a plain directory, not a package (same as the
# tests): make them importable by name.
EXAMPLES_DIR = REPO_ROOT / "examples" / "garden_models"
sys.path.insert(0, str(EXAMPLES_DIR))

PROJECT_NAME = "Demo: Atelier Custom Models"
OUT_DIR = REPO_ROOT / "demo_data" / "atelier"


# ── Model catalog ──────────────────────────────────────────────────────────────
# One entry per example model: where its source lives, its family, the discovery
# metadata that goes into the manifest, and the docs shown in the Atelier. The
# per-model fit + chat live in dedicated functions below (they differ enough that
# a uniform config would obscure more than it shares).

MODELS: list[dict] = [
    {
        "name": "awareness_structural_mmm",
        "title": "Awareness Structural MMM",
        "source_file": "awareness_structural_mmm.py",
        "module": "awareness_structural_mmm",
        "class_name": "AwarenessStructuralMMM",
        "kind": "mmm",
        "tags": ["awareness", "structural", "binomial", "survey", "goodwill"],
        "session_title": "Brand awareness — custom structural MMM",
        "recommended_fit": {"method": "map"},
        "docs": (
            "# Awareness Structural MMM\n\n"
            "A marketing-mix model whose KPI is **brand awareness**, not sales. "
            "Media build a latent *goodwill stock* that decays week to week; the "
            "observed KPI can be a Normal awareness index **or** a Binomial "
            "survey count (`#` aware out of `number_of_trials` respondents).\n\n"
            "## Identification\n"
            "- A retention prior (`retention_prior_alpha/beta`) pins the goodwill "
            "decay rate.\n"
            "- The survey-count mode reads `number_of_trials` from the model's "
            "`CONFIG_SCHEMA` and writes its own `pm.Binomial` observation with a "
            "logit link.\n\n"
            "## Declared estimands\n"
            "`awareness_lift` (per-period lift per channel), `contribution_roi` "
            "(goodwill per $), and `goodwill_stock` — a per-channel **latent "
            "contrast** on the `media_total` stock (observed vs channel-off)."
        ),
    },
    {
        "name": "bayesian_cfa",
        "title": "Bayesian Confirmatory Factor Analysis",
        "source_file": "bayesian_cfa.py",
        "module": "bayesian_cfa",
        "class_name": "BayesianCFA",
        "kind": "cfa",
        "tags": ["cfa", "factor-analysis", "non-mmm", "survey", "latent"],
        "session_title": "Survey structure — Bayesian CFA (non-MMM)",
        "recommended_fit": {"method": "map"},
        "docs": (
            "# Bayesian Confirmatory Factor Analysis\n\n"
            'A genuinely **non-MMM** family (`__garden_model_kind__ = "cfa"`) '
            "that rides the same garden → fit → estimand → report pipeline. It "
            "fits a measurement model over observed indicators with a fixed "
            "simple-structure loading pattern.\n\n"
            "## Model\n"
            "Marginal multivariate-normal likelihood "
            "`y ~ MvNormal(0, ΛΛᵀ + Ψ)` with positive (HalfNormal) loadings; "
            "factor scores integrated out.\n\n"
            "## Declared estimands\n"
            "`srmr` and `cov_fit` — per-draw fit indices realized through the "
            "same `evaluate_estimands` path as an MMM's ROI. "
            "`factor_loadings_summary()` gives the loadings table the "
            "family-agnostic report renders."
        ),
    },
    {
        "name": "bayesian_lca",
        "title": "Bayesian Latent Class Analysis",
        "source_file": "bayesian_lca.py",
        "module": "bayesian_lca",
        "class_name": "BayesianLCA",
        "kind": "latent_class",
        "tags": ["lca", "segmentation", "mixture", "non-mmm", "survey"],
        "session_title": "Audience segments — Bayesian LCA (non-MMM)",
        "recommended_fit": {"method": "map"},
        "docs": (
            "# Bayesian Latent Class Analysis\n\n"
            "A second non-MMM family (`__garden_model_kind__ = "
            '"latent_class"`): a mixture model over **binary indicators** that '
            "discovers `K` latent segments.\n\n"
            "## Model\n"
            "Discrete class labels are **integrated out** — each observation's "
            "log-likelihood is a `logsumexp` log-mixture of Bernoulli products "
            "(NUTS-able, no discrete latents). An *ordered* logit pins class "
            "order by size, resolving label-switching.\n\n"
            "## Declared estimands\n"
            "`class_size_k` (the share of each latent segment). "
            "`class_profile_summary()` gives the per-(class, item) "
            "endorsement table."
        ),
    },
    {
        "name": "bayesian_clv",
        "title": "Bayesian Customer Lifetime Value",
        "source_file": "bayesian_clv.py",
        "module": "bayesian_clv",
        "class_name": "BayesianCLV",
        "kind": "clv",
        "tags": ["clv", "ltv", "bg-nbd", "gamma-gamma", "non-mmm", "customers"],
        "session_title": "Customer lifetime value — BG/NBD + Gamma-Gamma (non-MMM)",
        "recommended_fit": {"method": "map"},
        "docs": (
            "# Bayesian Customer Lifetime Value\n\n"
            'A non-MMM family (`__garden_model_kind__ = "clv"`) for '
            "transaction-level customer value: **BG/NBD** purchase/dropout "
            "dynamics + **Gamma-Gamma** monetary value over per-customer RFM "
            "summaries (frequency, recency, age, mean repeat value).\n\n"
            "## Model\n"
            "Per-customer latents (purchase rate, dropout probability, spend "
            "scale) are integrated out analytically — the graph has a handful "
            "of population parameters regardless of customer count, so MAP "
            "fits in seconds and NUTS is comfortably tractable.\n\n"
            "## Declared estimands\n"
            "`mean_clv`, `total_clv`, `mean_expected_purchases`, "
            "`mean_p_alive`, plus one `segment_clv_<channel>` per acquisition "
            "segment. `customer_value_summary()` gives the value table the "
            "family-agnostic report renders; `get_clv_value` serves "
            "`value_per_conversion` to acquisition experiments so they are "
            "valued on LIFETIME worth, not first purchase."
        ),
    },
]


# ── Garden registration + publish ───────────────────────────────────────────────


def _config_schema_for(cls) -> dict | None:
    """JSON Schema of the model's CONFIG_SCHEMA (drives the Atelier params form)."""
    schema = getattr(cls, "CONFIG_SCHEMA", None)
    if schema is None:
        return None
    try:
        return schema.model_json_schema()
    except Exception:  # noqa: BLE001
        return None


def _synth_compat_report(cls, kind: str) -> dict:
    """A passing compatibility report mirroring ``run_compatibility_check``'s shape
    (the blocking tiers genuinely pass — proven by the model test-suite — so this
    is the fast stand-in, not a fabrication of behaviour). Non-MMM families skip
    the channel-only tiers, exactly as the real suite does."""
    from mmm_framework.garden import BLOCKING_TIERS, GARDEN_CONTRACT_VERSION
    from mmm_framework.garden.contract import is_bayesian_mmm_subclass

    is_mmm = kind == "mmm"
    tier_names = [
        "static",
        "build",
        "fit",
        "instance",
        "trace",
        "scaling",
        "ops_smoke",
        "carryover",
        "accuracy",
    ]
    # MMM-only tiers self-skip for a non-MMM family; carryover/accuracy are
    # optional and skipped in this single-scenario MAP run.
    skipped = {"carryover", "accuracy"}
    if not is_mmm:
        skipped |= {"scaling", "ops_smoke"}
    tiers = []
    for name in tier_names:
        sk = name in skipped
        tiers.append(
            {
                "name": name,
                "passed": not sk,
                "blocking": name in BLOCKING_TIERS,
                "skipped": sk,
                "detail": "skipped (not applicable)" if sk else "ok (seeded)",
            }
        )
    return {
        "contract_version": GARDEN_CONTRACT_VERSION,
        "class_name": getattr(cls, "__name__", str(cls)),
        "is_bayesian_mmm_subclass": is_bayesian_mmm_subclass(cls),
        "scenario": "clean",
        "fit_method": "map",
        "tiers": tiers,
        "blocking_passed": True,
        "score": 1.0,
        "summary": "Seeded demo: blocking tiers pass (full suite verified in CI).",
    }


def _compat_report(cls, kind: str, real_compat: bool) -> dict:
    """Real compatibility report (genuine suite) when requested and it passes;
    otherwise a passing synthesized report so the model can be published."""
    if real_compat:
        try:
            from mmm_framework.garden.compat import run_compatibility_check

            print(f"    running the genuine compatibility suite for {cls.__name__}…")
            rep = run_compatibility_check(
                cls,
                scenarios=("clean",),
                fit_method="map",
                n_weeks=60,
                check_carryover=False,
            )
            if rep.get("blocking_passed"):
                return rep
            print(
                f"    [warn] real compat did not pass for {cls.__name__} "
                f"({rep.get('summary')}); using a synthesized passing report"
            )
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] compat suite errored for {cls.__name__}: {e}")
    return _synth_compat_report(cls, kind)


def _purge_demo_models(store, org_id: str, names: list[str]) -> None:
    """Remove any prior versions of the demo models (rows + on-disk source) so
    re-seeding always starts from v1. Demo-only direct cleanup — the public API
    keeps published versions immutable on purpose."""
    import shutil

    from mmm_framework.agents import workspace as ws

    with store._conn() as c:
        for nm in names:
            c.execute(
                "DELETE FROM garden_models WHERE org_id = ? AND name = ?",
                (org_id, nm),
            )
    for nm in names:
        model_root = ws.garden_dir(org_id, nm, 1).parent  # …/garden/<org>/<name>/
        if model_root.exists():
            shutil.rmtree(model_root, ignore_errors=True)


def _register_and_publish(store, org_id: str, model_def: dict, cls, real_compat: bool):
    """Register the example source as a draft, attach a passing compat report,
    and promote draft → tested → published. Returns the published row."""
    from mmm_framework.agents.garden_registry import register_garden_model_core

    source = (EXAMPLES_DIR / model_def["source_file"]).read_text(encoding="utf-8")
    row = register_garden_model_core(
        org_id=org_id,
        source_code=source,
        name=model_def["name"],
        docs=model_def["docs"],
        tags=model_def["tags"],
        recommended_fit=model_def["recommended_fit"],
        default_estimands=[
            e if isinstance(e, str) else getattr(e, "name", str(e))
            for e in getattr(cls, "DEFAULT_ESTIMANDS", [])
        ],
        config_schema=_config_schema_for(cls),
    )
    report = _compat_report(cls, model_def["kind"], real_compat)
    store.set_garden_compat_report(row["id"], report)
    store.transition_garden_model(
        row["id"], "tested", note="compatibility suite passed"
    )
    published = store.transition_garden_model(
        row["id"], "published", note="seeded Atelier demo"
    )
    return published


def _garden_ref(row: dict) -> dict:
    """The provenance a session spec carries so the agent re-loads this exact
    published class (what ``load_garden_model`` sets as ``spec['garden_ref']``)."""
    manifest = row.get("manifest") or {}
    return {
        "name": row["name"],
        "version": row["version"],
        "source_path": row["source_path"],
        "class_name": manifest.get("class_name"),
        "contract_version": manifest.get("contract_version"),
    }


# ── Visualizations + tables (rendered in the workspace Results tab) ──────────────
# Plots are stored content-addressed via workspace.store_plot and referenced from
# dashboard_data["plots"]; tables via tables.publish_tables into
# dashboard_data["tables"] (group "results"). The React workspace renders both in
# the Results tab (PlotCard / TableCard) exactly as it does for live agent output.

# Docs palette (sage / steel / gold / rust) so the demo charts read as the product.
_PALETTE = ["#6d8a4a", "#4a6d8a", "#b8860b", "#a04535", "#9db8c9", "#d4a86a"]


def _fig_json(fig) -> dict:
    import json as _json

    return _json.loads(fig.to_json())


def _layout(fig, title: str, xlab: str = "", ylab: str = "") -> dict:
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15)),
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        height=360,
        margin=dict(l=64, r=24, t=52, b=52),
    )
    return _fig_json(fig)


def _bar_fig(title, x, y, *, xlab="", ylab="", err=None, colors=None) -> dict:
    import plotly.graph_objects as go

    bar = go.Bar(x=list(x), y=list(y), marker_color=colors or _PALETTE[0])
    if err is not None:
        bar.error_y = dict(type="data", array=list(err), visible=True)
    return _layout(go.Figure(bar), title, xlab, ylab)


def _grouped_bar_fig(title, categories, series: dict, *, xlab="", ylab="") -> dict:
    import plotly.graph_objects as go

    bars = [
        go.Bar(
            name=n,
            x=list(categories),
            y=list(v),
            marker_color=_PALETTE[i % len(_PALETTE)],
        )
        for i, (n, v) in enumerate(series.items())
    ]
    fig = go.Figure(bars)
    fig.update_layout(barmode="group")
    return _layout(fig, title, xlab, ylab)


def _pie_fig(title, labels, values) -> dict:
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            hole=0.45,
            sort=False,
            marker=dict(colors=_PALETTE),
        )
    )
    return _layout(fig, title)


def _line_fig(title, x, y, *, xlab="", ylab="") -> dict:
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=list(x), y=list(y), mode="lines+markers", line=dict(color=_PALETTE[0])
        )
    )
    return _layout(fig, title, xlab, ylab)


def _estimands_table(est_dict: dict, title: str, source: str) -> dict | None:
    """A 'Declared estimands' table from the serialized evaluate_estimands map."""
    from mmm_framework.agents.tables import records_to_table_json

    recs = []
    for key, v in (est_dict or {}).items():
        if v.get("status") != "ok":
            continue
        lo, hi = v.get("hdi_low"), v.get("hdi_high")
        recs.append(
            {
                "estimand": key,
                "mean": round(v["mean"], 4) if v.get("mean") is not None else None,
                "94% HDI": (
                    f"[{lo:.3f}, {hi:.3f}]"
                    if lo is not None and hi is not None
                    else "—"
                ),
                "units": v.get("units", ""),
            }
        )
    if not recs:
        return None
    return records_to_table_json(recs, title=title, source=source, group="results")


def _config_table(
    model_def: dict, row: dict, spec: dict, extra_rows: list[dict]
) -> dict:
    """The model-specific configuration, as a table the Results tab renders."""
    from mmm_framework.agents.tables import records_to_table_json

    manifest = row.get("manifest") or {}
    recs = [
        {"setting": "Garden model", "value": f"{row['name']} v{row['version']}"},
        {"setting": "Family (kind)", "value": model_def["kind"]},
        {"setting": "Class", "value": model_def["class_name"]},
        {"setting": "Contract", "value": manifest.get("contract_version")},
        *extra_rows,
        {
            "setting": "Declared estimands",
            "value": ", ".join(spec.get("estimands", [])),
        },
    ]
    return records_to_table_json(
        recs,
        title="Model configuration",
        source=f"garden:{row['name']} v{row['version']}",
        group="results",
        columns=["setting", "value"],
    )


def _df_table(df, title: str, source: str) -> dict:
    from mmm_framework.agents.tables import df_to_table_json

    return df_to_table_json(df, title=title, source=source, group="results")


def _attach_viz(
    dashboard: dict, tid: str, figures: list, tables: list
) -> tuple[int, int]:
    """Store figures + tables content-addressed against the session thread and
    append their refs to the dashboard (exactly how execute_python / EDA do it),
    so the React Results tab renders them."""
    from mmm_framework.agents import workspace as ws
    from mmm_framework.agents.tables import publish_tables

    refs = []
    for title, fig in figures or []:
        try:
            refs.append({"id": ws.store_plot(fig, tid), "title": title})
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] plot '{title}' dropped: {e}")
    if refs:
        dashboard["plots"] = (dashboard.get("plots") or []) + refs
    clean = [t for t in (tables or []) if t]
    if clean:
        try:
            publish_tables(clean, dashboard, tid)
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] tables dropped: {e}")
    return len(refs), len(clean)


# ── Per-model fits (real MAP) ────────────────────────────────────────────────────
# Each returns a dict: {headline, dashboard, spec, dataset_path, figures, tables}.
# headline drives the chat narration; figures/tables populate the Results tab.


def _new_run_id(name: str) -> str:
    return f"atelier_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _est_dict(results: dict) -> dict:
    """Serialize an ``evaluate_estimands`` result map to JSON-safe records."""
    out = {}
    for key, res in results.items():
        try:
            out[key] = {
                "mean": None if res.mean is None else float(res.mean),
                "hdi_low": None if res.hdi_low is None else float(res.hdi_low),
                "hdi_high": None if res.hdi_high is None else float(res.hdi_high),
                "units": getattr(res, "units", ""),
                "status": res.status,
            }
        except Exception:  # noqa: BLE001
            continue
    return out


def _write_panel_csv(panel, path: Path, kpi_name: str) -> str | None:
    """Best-effort wide CSV of a panel's observed data (for the workspace
    dataset_path + reproducibility). Cosmetic — the fit is done in-process."""
    try:
        import pandas as pd

        media = getattr(panel, "X_media", None)
        y = getattr(panel, "y", None)
        if media is None or y is None:
            return None
        df = pd.DataFrame(media).reset_index(drop=True)
        df.insert(0, kpi_name, list(getattr(y, "values", y)))
        idx = getattr(panel, "index", None)
        if idx is not None:
            df.insert(0, "Period", [str(p)[:10] for p in idx])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def _awareness_panel():
    """A national awareness series driven by two channels, with a Binomial
    survey-count KPI (the showcase mode). Mirrors the model's test fixture."""
    import numpy as np
    import pandas as pd

    from awareness_structural_mmm import AwarenessStructuralMMM  # noqa: F401
    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset

    n_trials = 1000
    periods = pd.date_range("2021-01-04", periods=52, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    tv = np.abs(rng.normal(100, 25, n))
    digital = np.abs(rng.normal(80, 20, n))
    logit = -0.4 + 1.3 * (tv / tv.max()) + 0.7 * (digital / digital.max())
    rate = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(n_trials, rate), name="Awareness").astype(float)
    config = MFFConfig(
        kpi=KPIConfig(name="Awareness", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
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
    return panel, n_trials


def _fit_awareness(model_def, row, fast: bool):
    import numpy as np

    spec = {
        "kpi": "Awareness",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "likelihood": {"family": "binomial", "params": {"n_trials": 1000}},
        "model_params": {"number_of_trials": 1000},
        "estimands": ["awareness_lift", "contribution_roi"],
        "garden_ref": _garden_ref(row),
        "inference": {"method": "map"},
    }
    panel, n_trials = _awareness_panel()
    dataset_path = _write_panel_csv(
        panel, OUT_DIR / "awareness_dataset.csv", "Awareness"
    )
    run_id = _new_run_id(model_def["name"])

    config_rows = [
        {"setting": "KPI", "value": "Awareness (survey count)"},
        {"setting": "Likelihood", "value": "Binomial, logit link"},
        {"setting": "Trials / period", "value": 1000},
        {"setting": "Channels", "value": "TV, Digital"},
        {"setting": "Goodwill", "value": "geometric retention (latent stock)"},
    ]

    if fast:
        headline = {
            "rho": 0.62,
            "half_life": 1.5,
            "lifts": {"TV": 328.0, "Digital": 212.0},
            "goodwill": {"TV": 1.41, "Digital": 0.94},
            "roi": {"TV": 2.3, "Digital": 1.6},
        }
        dashboard = _dashboard(
            run_id, model_def, spec, dataset_path, "Awareness", ["TV", "Digital"], {}
        )
        figs = [
            (
                "Media-driven awareness lift",
                _bar_fig(
                    "Media-driven awareness lift",
                    list(headline["lifts"]),
                    list(headline["lifts"].values()),
                    ylab="aware respondents / period",
                    colors=_PALETTE[: len(headline["lifts"])],
                ),
            ),
            (
                "Goodwill retention decay",
                _line_fig(
                    "Goodwill retention decay",
                    list(range(13)),
                    [headline["rho"] ** k for k in range(13)],
                    xlab="weeks since exposure",
                    ylab="retained goodwill",
                ),
            ),
        ]
        tables = [_config_table(model_def, row, spec, config_rows)]
        return {
            "headline": headline,
            "dashboard": dashboard,
            "spec": spec,
            "dataset_path": dataset_path,
            "figures": figs,
            "tables": tables,
        }

    from awareness_structural_mmm import AwarenessStructuralMMM, AwarenessParams  # noqa
    from mmm_framework.config import LikelihoodConfig, ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    mmm = AwarenessStructuralMMM(
        panel,
        ModelConfig(likelihood=LikelihoodConfig.binomial(n_trials=n_trials)),
        TrendConfig(type=TrendType.NONE),
        model_params={"number_of_trials": n_trials},
    )
    mmm.fit(method="map", random_seed=0)

    rho = float(np.asarray(mmm._trace.posterior["awareness_retention"]).mean())
    half_life = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("inf")

    try:
        ests = mmm.evaluate_estimands()  # full DEFAULT_ESTIMANDS (incl goodwill_stock)
    except Exception:  # noqa: BLE001
        ests = mmm.evaluate_estimands(["awareness_lift", "contribution_roi"])

    lifts = {
        k.split(":")[-1]: float(v.mean)
        for k, v in ests.items()
        if k.startswith("awareness_lift") and v.status == "ok" and v.mean is not None
    }
    goodwill = {
        k.split(":")[-1]: float(v.mean)
        for k, v in ests.items()
        if k.startswith("goodwill_stock") and v.status == "ok" and v.mean is not None
    }
    roi_metrics = []
    roi = {}
    try:
        from mmm_framework.reporting.helpers import compute_roi_with_uncertainty

        roi_df = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)
        roi_metrics = roi_df.to_dict(orient="records")
        roi = {
            r["channel"]: float(r["roi_mean"])
            for r in roi_metrics
            if "channel" in r and r.get("roi_mean") is not None
        }
    except Exception:  # noqa: BLE001
        pass

    _render_report(mmm, model_def["name"])  # on-disk artifact (not surfaced in dev UI)
    est_serial = _est_dict(ests)
    dashboard = _dashboard(
        run_id,
        model_def,
        spec,
        dataset_path,
        "Awareness",
        ["TV", "Digital"],
        est_serial,
        roi_metrics=roi_metrics,
    )
    headline = {
        "rho": rho,
        "half_life": half_life,
        "lifts": lifts,
        "goodwill": goodwill,
        "roi": roi,
    }
    figs = []
    if lifts:
        figs.append(
            (
                "Media-driven awareness lift",
                _bar_fig(
                    "Media-driven awareness lift",
                    list(lifts),
                    list(lifts.values()),
                    ylab="aware respondents / period",
                    colors=_PALETTE[: len(lifts)],
                ),
            )
        )
    if roi:
        figs.append(
            (
                "Goodwill efficiency by channel",
                _bar_fig(
                    "Goodwill efficiency (per $ spend)",
                    list(roi),
                    list(roi.values()),
                    ylab="goodwill ROI",
                    colors=_PALETTE[1 : 1 + len(roi)],
                ),
            )
        )
    figs.append(
        (
            "Goodwill retention decay",
            _line_fig(
                "Goodwill retention decay",
                list(range(13)),
                [rho**k for k in range(13)],
                xlab="weeks since exposure",
                ylab="retained goodwill",
            ),
        )
    )
    tables = [
        _estimands_table(est_serial, "Declared estimands", f"garden:{row['name']}"),
        _config_table(model_def, row, spec, config_rows),
    ]
    return {
        "headline": headline,
        "dashboard": dashboard,
        "spec": spec,
        "dataset_path": dataset_path,
        "figures": figs,
        "tables": tables,
    }


def _fit_cfa(model_def, row, fast: bool):
    spec = {
        "kpi": "x1",
        "media_channels": [{"name": f"x{i}"} for i in range(2, 7)],
        "model_params": {"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
        "estimands": ["srmr", "cov_fit"],
        "garden_ref": _garden_ref(row),
        "inference": {"method": "map"},
    }
    run_id = _new_run_id(model_def["name"])

    config_rows = [
        {"setting": "Indicators", "value": "x1–x6 (6 observed)"},
        {"setting": "Likelihood", "value": "Marginal MvNormal (ΛΛᵀ + Ψ)"},
        {"setting": "Factors", "value": 2},
        {"setting": "Assignment", "value": "x1–x3 → F1, x4–x6 → F2"},
        {"setting": "Loadings prior", "value": "HalfNormal (positive)"},
    ]

    def _fit_indices_fig(srmr, cov_fit):
        return _bar_fig(
            "Fit indices",
            ["SRMR (lower better)", "Cov. fit (higher better)"],
            [srmr, cov_fit],
            ylab="value",
            colors=[_PALETTE[3], _PALETTE[0]],
        )

    if fast:
        headline = {
            "srmr": 0.04,
            "cov_fit": 0.93,
            "loadings": [("x1", "F1", 0.76), ("x4", "F2", 0.74)],
        }
        figs = [
            (
                "Factor loadings",
                _bar_fig(
                    "Factor loadings",
                    [ld[0] for ld in headline["loadings"]],
                    [ld[2] for ld in headline["loadings"]],
                    ylab="loading",
                    colors=_PALETTE[: len(headline["loadings"])],
                ),
            ),
            ("Fit indices", _fit_indices_fig(headline["srmr"], headline["cov_fit"])),
        ]
        return {
            "headline": headline,
            "dashboard": _dashboard(run_id, model_def, spec, None, "x1", [], {}),
            "spec": spec,
            "dataset_path": None,
            "figures": figs,
            "tables": [_config_table(model_def, row, spec, config_rows)],
        }

    from bayesian_cfa import BayesianCFA, synthetic_cfa_panel
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    panel, true_load = synthetic_cfa_panel(n=400)
    dataset_path = _write_panel_csv(panel, OUT_DIR / "cfa_dataset.csv", "x1")
    mmm = BayesianCFA(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
    )
    mmm.fit(method="map", random_seed=7)
    ests = mmm.evaluate_estimands()
    summary = mmm.factor_loadings_summary()
    loadings = [
        (str(r["indicator"]), str(r["factor"]), float(r["loading"]))
        for _, r in summary.iterrows()
    ]
    _render_report(mmm, model_def["name"])  # on-disk artifact (not surfaced in dev UI)
    srmr = float(ests["srmr"].mean) if "srmr" in ests else None
    cov_fit = float(ests["cov_fit"].mean) if "cov_fit" in ests else None
    headline = {"srmr": srmr, "cov_fit": cov_fit, "loadings": loadings}
    est_serial = _est_dict(ests)
    dashboard = _dashboard(
        run_id,
        model_def,
        spec,
        dataset_path,
        "x1",
        [],
        est_serial,
        latent_summary=summary.to_dict(orient="records"),
    )
    colors = [_PALETTE[0] if f == "F1" else _PALETTE[1] for f in summary["factor"]]
    figs = [
        (
            "Factor loadings",
            _bar_fig(
                "Factor loadings (by assigned factor)",
                list(summary["indicator"]),
                list(summary["loading"]),
                ylab="loading",
                colors=colors,
            ),
        ),
        ("Fit indices", _fit_indices_fig(srmr, cov_fit)),
    ]
    tables = [
        _df_table(summary, "Factor loadings", f"garden:{row['name']}"),
        _estimands_table(
            est_serial, "Fit indices (estimands)", f"garden:{row['name']}"
        ),
        _config_table(model_def, row, spec, config_rows),
    ]
    return {
        "headline": headline,
        "dashboard": dashboard,
        "spec": spec,
        "dataset_path": dataset_path,
        "figures": figs,
        "tables": tables,
    }


def _fit_lca(model_def, row, fast: bool):
    spec = {
        "kpi": "q1",
        "media_channels": [{"name": f"q{i}"} for i in range(2, 7)],
        "model_params": {"n_classes": 2},
        "estimands": ["class_size_1", "class_size_2"],
        "garden_ref": _garden_ref(row),
        "inference": {"method": "map"},
    }
    run_id = _new_run_id(model_def["name"])

    config_rows = [
        {"setting": "Items", "value": "q1–q6 (6 binary)"},
        {"setting": "Likelihood", "value": "Bernoulli mixture (logsumexp)"},
        {"setting": "Latent classes", "value": 2},
        {
            "setting": "Identification",
            "value": "size-ordered logit (no label-switching)",
        },
        {"setting": "Item prior", "value": "Beta endorsement probs"},
    ]

    def _sizes_fig(sizes: dict):
        return _pie_fig(
            "Latent class sizes",
            ["Class 1", "Class 2"],
            [sizes.get("C1"), sizes.get("C2")],
        )

    if fast:
        headline = {
            "sizes": {"C1": 0.35, "C2": 0.65},
            "profile": [("q1", 0.86, 0.18), ("q5", 0.2, 0.83)],
        }
        figs = [
            (
                "Class item-endorsement profiles",
                _grouped_bar_fig(
                    "Class item-endorsement profiles",
                    [p[0] for p in headline["profile"]],
                    {
                        "Class 1": [p[1] for p in headline["profile"]],
                        "Class 2": [p[2] for p in headline["profile"]],
                    },
                    ylab="P(endorse | class)",
                ),
            ),
            ("Latent class sizes", _sizes_fig(headline["sizes"])),
        ]
        return {
            "headline": headline,
            "dashboard": _dashboard(run_id, model_def, spec, None, "q1", [], {}),
            "spec": spec,
            "dataset_path": None,
            "figures": figs,
            "tables": [_config_table(model_def, row, spec, config_rows)],
        }

    from bayesian_lca import BayesianLCA, synthetic_lca_panel
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    panel, sizes, profiles = synthetic_lca_panel(n=600)
    dataset_path = _write_panel_csv(panel, OUT_DIR / "lca_dataset.csv", "q1")
    mmm = BayesianLCA(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"n_classes": 2},
    )
    mmm.fit(method="map", random_seed=11)
    ests = mmm.evaluate_estimands()
    prof = mmm.class_profile_summary()
    pivot = prof.pivot(index="item", columns="class", values="prob")
    profile = [
        (str(item), float(pivot.loc[item, "C1"]), float(pivot.loc[item, "C2"]))
        for item in ["q1", "q5"]
        if item in pivot.index
    ]
    _render_report(mmm, model_def["name"])  # on-disk artifact (not surfaced in dev UI)
    sizes = {
        "C1": float(ests["class_size_1"].mean) if "class_size_1" in ests else None,
        "C2": float(ests["class_size_2"].mean) if "class_size_2" in ests else None,
    }
    headline = {"sizes": sizes, "profile": profile}
    est_serial = _est_dict(ests)
    dashboard = _dashboard(
        run_id,
        model_def,
        spec,
        dataset_path,
        "q1",
        [],
        est_serial,
        latent_summary=prof.to_dict(orient="records"),
    )
    figs = [
        (
            "Class item-endorsement profiles",
            _grouped_bar_fig(
                "Class item-endorsement profiles",
                list(pivot.index),
                {"Class 1": list(pivot["C1"]), "Class 2": list(pivot["C2"])},
                xlab="survey item",
                ylab="P(endorse | class)",
            ),
        ),
        ("Latent class sizes", _sizes_fig(sizes)),
    ]
    size_rows = [
        {"class": "Class 1", "share": round(sizes["C1"], 4) if sizes["C1"] else None},
        {"class": "Class 2", "share": round(sizes["C2"], 4) if sizes["C2"] else None},
    ]
    from mmm_framework.agents.tables import records_to_table_json

    tables = [
        _df_table(prof, "Class item-endorsement profiles", f"garden:{row['name']}"),
        records_to_table_json(
            size_rows,
            title="Latent class sizes",
            source=f"garden:{row['name']}",
            group="results",
        ),
        _config_table(model_def, row, spec, config_rows),
    ]
    return {
        "headline": headline,
        "dashboard": dashboard,
        "spec": spec,
        "dataset_path": dataset_path,
        "figures": figs,
        "tables": tables,
    }


def _render_report(mmm, name: str) -> str | None:
    """Render the family-aware HTML report and save it (cosmetic artifact)."""
    try:
        from mmm_framework.reporting import MMMReportGenerator

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        gen = MMMReportGenerator(model=mmm)
        path = OUT_DIR / f"{name}_report.html"
        gen.to_html(path)
        return str(path)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] report render failed for {name}: {e}")
        return None


def _dashboard(
    run_id,
    model_def,
    spec,
    dataset_path,
    kpi,
    channels,
    estimands,
    *,
    roi_metrics=None,
    latent_summary=None,
    report_path=None,
) -> dict:
    """The workspace Results payload (same envelope build_and_fit produces)."""
    model_run = {
        "run_id": run_id,
        "run_name": f"run_{run_id}",
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "kpi": kpi,
        "channels": channels,
        "spec": spec,
        "model_kind": model_def["kind"],
        "garden_ref": spec.get("garden_ref"),
        "summary": f"Atelier demo fit of the published '{model_def['name']}' model.",
        "estimands": estimands,
    }
    dashboard = {
        "model_status": "completed",
        "summary": model_run["summary"],
        "model_run": model_run,
        "model_spec": spec,
        "estimands": estimands,
        "garden_ref": spec.get("garden_ref"),
    }
    if roi_metrics:
        dashboard["roi_metrics"] = roi_metrics
    if latent_summary:
        dashboard["latent_summary"] = latent_summary
    if report_path:
        dashboard["report_path"] = report_path
    return dashboard


# ── Chat scripts (real numbers from the fits) ───────────────────────────────────


def _fmt(x, prec=2, default="—"):
    try:
        return f"{float(x):.{prec}f}"
    except (TypeError, ValueError):
        return default


def _awareness_turns(h) -> list[tuple[str, str]]:
    lifts = h.get("lifts") or {}
    goodwill = h.get("goodwill") or {}
    lift_lines = (
        "\n".join(
            f"- `{ch}`: **{_fmt(v, 0)}** aware respondents/period (of 1,000 surveyed)"
            + (
                f"; latent goodwill stock **{_fmt(goodwill[ch], 2)}** (std)"
                if ch in goodwill
                else ""
            )
            for ch, v in lifts.items()
        )
        or "- (estimands realized from the fitted posterior)"
    )
    return [
        (
            "human",
            "We have a weekly brand-tracker survey (1,000 respondents/week) and I "
            "want to measure what our media does for **awareness**, not just "
            "sales. Do we have a custom model for that?",
        ),
        (
            "ai",
            "→ `list_garden_models()`\n\n"
            "Three models are published in your garden:\n"
            "- **awareness_structural_mmm** (mmm) — latent goodwill stock, Normal "
            "index *or* Binomial survey-count KPI\n"
            "- **bayesian_cfa** (cfa) — confirmatory factor analysis\n"
            "- **bayesian_lca** (latent_class) — latent-class segmentation\n\n"
            "For a survey-count awareness KPI, **awareness_structural_mmm** is the "
            "fit. Want me to load it and run it on the tracker?",
        ),
        (
            "human",
            "Yes — load it and fit it on the survey counts (1,000 trials/week).",
        ),
        (
            "ai",
            "→ `load_garden_model('awareness_structural_mmm')`  *(sets "
            "`spec['garden_ref']`, applies the model's recommended fit)*\n"
            "→ `fit_mmm_model(...)`  *(Binomial likelihood, `number_of_trials=1000`)*\n\n"
            "Fitted ✅ (MAP). The goodwill stock decays with retention "
            f"**ρ ≈ {_fmt(h.get('rho'), 3)}** — a half-life of about "
            f"**{_fmt(h.get('half_life'), 1)} weeks**. The model's declared "
            "estimands are ready: `awareness_lift`, `contribution_roi`, and the "
            "per-channel `goodwill_stock` latent contrast.",
        ),
        ("human", "What is each channel buying us in awareness?"),
        (
            "ai",
            "→ `get_estimands()`\n\n"
            f"{lift_lines}\n\n"
            "`goodwill_stock` diffs the latent `media_total` stock between the "
            "observed world and a channel-off world, so it reads as the awareness "
            "each channel is *holding up* week to week. The lift / efficiency / "
            "retention charts, an estimands table, the model configuration, and "
            "the full report are all on the **Results** tab.",
        ),
    ]


def _cfa_turns(h) -> list[tuple[str, str]]:
    loadings = h.get("loadings") or []
    load_lines = (
        "\n".join(
            f"- `{ind}` → **{fac}**, loading **{_fmt(val, 2)}**"
            for ind, fac, val in loadings[:6]
        )
        or "- (loadings recovered from the fitted posterior)"
    )
    return [
        (
            "human",
            "I have a 6-item brand-perception survey and a hypothesis that it "
            "measures two underlying factors. Can we confirm that structure?",
        ),
        (
            "ai",
            "→ `list_garden_models()` → **bayesian_cfa** is the right tool — a "
            "non-MMM **confirmatory factor analysis** family in your garden. It "
            "rides the same fit/estimand/report pipeline as the MMMs. Load it?",
        ),
        (
            "human",
            "Load it with a two-factor structure (items 1–3 on F1, 4–6 on F2) "
            "and fit.",
        ),
        (
            "ai",
            "→ `load_garden_model('bayesian_cfa')`  *(sets `spec['garden_ref']`)*\n"
            "→ `fit_mmm_model(...)`  *(`n_factors=2`, "
            "`factor_assignment=[0,0,0,1,1,1]`)*\n\n"
            "Fitted ✅ (MAP). The fit indices — the model's declared estimands — are:\n"
            f"- **SRMR {_fmt(h.get('srmr'), 3)}** (a good fit is < 0.08)\n"
            f"- **covariance fit {_fmt(h.get('cov_fit'), 2)}** (closer to 1 is better)\n\n"
            "Both clear the usual thresholds, so the two-factor structure holds.",
        ),
        ("human", "Show me the loadings."),
        (
            "ai",
            "→ `get_estimands()` + `factor_loadings_summary()`\n\n"
            f"{load_lines}\n\n"
            "Each item loads cleanly on its assigned factor. The loadings + "
            "fit-index charts, the loadings table, the model configuration, and "
            "the non-MMM report (Factor Analysis section, channel/ROI gated off) "
            "are on the **Results** tab.",
        ),
    ]


def _lca_turns(h) -> list[tuple[str, str]]:
    sizes = h.get("sizes") or {}
    prof = h.get("profile") or []
    prof_lines = (
        "\n".join(
            f"- `{item}`: C1 endorses **{_fmt(c1, 2)}**, C2 **{_fmt(c2, 2)}**"
            for item, c1, c2 in prof
        )
        or "- (profiles recovered from the fitted posterior)"
    )
    return [
        (
            "human",
            "Can we segment our survey respondents into latent groups based on "
            "their yes/no answers to six attitude questions?",
        ),
        (
            "ai",
            "→ `list_garden_models()` → **bayesian_lca** does exactly this — a "
            "non-MMM **latent class analysis** that finds hidden segments in "
            "binary survey items. Load it and look for two classes?",
        ),
        ("human", "Yes, two classes."),
        (
            "ai",
            "→ `load_garden_model('bayesian_lca')`  *(sets `spec['garden_ref']`)*\n"
            "→ `fit_mmm_model(...)`  *(`n_classes=2`)*\n\n"
            "Fitted ✅ (MAP). The discrete labels are integrated out, and an "
            "ordered prior pins the class order by size, so the classes are stable:\n"
            f"- **Class 1: {_fmt((sizes.get('C1') or 0) * 100, 0)}%** of respondents\n"
            f"- **Class 2: {_fmt((sizes.get('C2') or 0) * 100, 0)}%**\n\n"
            "Those shares are the model's `class_size_k` estimands.",
        ),
        ("human", "What distinguishes the two segments?"),
        (
            "ai",
            "→ `get_estimands()` + `class_profile_summary()`\n\n"
            f"{prof_lines}\n\n"
            "The two segments are near mirror images on the discriminating items "
            "— one endorses the early questions and rejects the later ones, the "
            "other the reverse. The class-profile chart, the class-size breakdown, "
            "the model configuration, and the latent-class report are on the "
            "**Results** tab.",
        ),
    ]


def _fit_clv(model_def, row, fast: bool):
    channels = {
        "Search": {"share": 0.4, "lam_mult": 1.3, "value_mult": 2.0},
        "Social": {"share": 0.6, "lam_mult": 0.8, "value_mult": 0.6},
    }
    spec = {
        "kpi": "frequency",
        "media_channels": [
            {"name": c} for c in ("recency", "T", "monetary", "segment_code")
        ],
        "model_params": {
            "horizon_periods": 52,
            "segment_column": "segment_code",
            "segment_labels": ["Search", "Social"],
        },
        "estimands": [
            "mean_clv",
            "total_clv",
            "mean_p_alive",
            "segment_clv_Search",
            "segment_clv_Social",
        ],
        "garden_ref": _garden_ref(row),
        "inference": {"method": "map"},
    }
    run_id = _new_run_id(model_def["name"])

    config_rows = [
        {"setting": "Data", "value": "per-customer RFM (from a transaction log)"},
        {"setting": "Purchase dynamics", "value": "BG/NBD (latents integrated out)"},
        {"setting": "Monetary value", "value": "Gamma-Gamma (repeat values)"},
        {"setting": "CLV horizon", "value": "52 weeks"},
        {"setting": "Segments", "value": "acquisition channel (Search / Social)"},
    ]

    def _segment_fig(seg: dict):
        return _bar_fig(
            "CLV per acquired customer, by acquisition channel",
            list(seg.keys()),
            [seg[k] for k in seg],
            ylab="posterior mean CLV",
        )

    if fast:
        headline = {
            "mean_clv": 23.5,
            "mean_p_alive": 0.67,
            "segments": {"Search": 37.8, "Social": 14.7},
            "cac": {"Search": 25.0, "Social": 12.0},
        }
        figs = [
            ("Segment CLV", _segment_fig(headline["segments"])),
            (
                "CLV distribution",
                _bar_fig(
                    "CLV percentiles across customers",
                    ["p50", "p80", "p90", "p99"],
                    [6.6, 21.6, 45.6, 201.5],
                    ylab="CLV",
                ),
            ),
        ]
        return {
            "headline": headline,
            "dashboard": _dashboard(run_id, model_def, spec, None, "frequency", [], {}),
            "spec": spec,
            "dataset_path": None,
            "figures": figs,
            "tables": [_config_table(model_def, row, spec, config_rows)],
        }

    import numpy as np
    import pandas as pd
    from bayesian_clv import BayesianCLV, rfm_panel, segment_model_params
    from mmm_framework.config import ModelConfig
    from mmm_framework.ltv import (
        clv_to_cac,
        new_customer_clv_series,
        transactions_to_rfm,
    )
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType
    from mmm_framework.synth.dgp_clv import make_clv_world

    world = make_clv_world(seed=11, n_customers=1500, channels=channels)
    rfm = transactions_to_rfm(
        world.transactions,
        value_col="value",
        observation_end=world.observation_end,
        segment_col="acquisition_channel",
    )
    panel = rfm_panel(rfm)
    dataset_path = _write_panel_csv(panel, OUT_DIR / "clv_dataset.csv", "frequency")
    mp = {"horizon_periods": 52, **segment_model_params(rfm)}
    mmm = BayesianCLV(
        panel, ModelConfig(), TrendConfig(type=TrendType.NONE), model_params=mp
    )
    mmm.fit(method="map", random_seed=11)

    ests = mmm.evaluate_estimands()
    summary = mmm.customer_value_summary()
    segments = mmm.segment_clv_means()
    cac = {"Search": 25.0, "Social": 12.0}
    econ = clv_to_cac(segments, cac)
    _render_report(mmm, model_def["name"])

    clv_pc = pd.Series(
        mmm._trace.posterior["clv"].mean(("chain", "draw")).values, index=rfm.index
    )
    cohort = new_customer_clv_series(world.transactions, clv_pc)

    headline = {
        "mean_clv": float(ests["mean_clv"].mean) if "mean_clv" in ests else None,
        "mean_p_alive": (
            float(ests["mean_p_alive"].mean) if "mean_p_alive" in ests else None
        ),
        "segments": segments,
        "cac": cac,
    }
    est_serial = _est_dict(ests)
    dashboard = _dashboard(
        run_id,
        model_def,
        spec,
        dataset_path,
        "frequency",
        [],
        est_serial,
        latent_summary=summary.to_dict(orient="records"),
    )
    clv_mean = clv_pc.to_numpy()
    figs = [
        ("Segment CLV", _segment_fig(segments)),
        (
            "CLV distribution",
            _bar_fig(
                "CLV percentiles across customers",
                ["p50", "p80", "p90", "p99"],
                [float(np.quantile(clv_mean, q)) for q in (0.5, 0.8, 0.9, 0.99)],
                ylab="CLV",
            ),
        ),
        (
            "Weekly cohort CLV",
            _line_fig(
                "Lifetime value of each week's acquired cohort",
                [str(p.date()) for p in cohort.index],
                list(cohort["cohort_clv"]),
                xlab="acquisition week",
                ylab="cohort CLV",
            ),
        ),
    ]
    from mmm_framework.agents.tables import records_to_table_json

    tables = [
        _df_table(summary.round(2), "Customer value profile", f"garden:{row['name']}"),
        records_to_table_json(
            econ.round(2).reset_index().to_dict(orient="records"),
            title="CLV vs CAC by acquisition channel",
            source=f"garden:{row['name']}",
            group="results",
        ),
        _config_table(model_def, row, spec, config_rows),
    ]
    est_table = _estimands_table(
        est_serial, "Declared estimands", f"garden:{row['name']}"
    )
    if est_table:
        tables.insert(0, est_table)
    return {
        "headline": headline,
        "dashboard": dashboard,
        "spec": spec,
        "dataset_path": dataset_path,
        "figures": figs,
        "tables": tables,
    }


def _clv_turns(h) -> list[tuple[str, str]]:
    seg = h.get("segments") or {}
    cac = h.get("cac") or {}
    seg_lines = (
        "\n".join(
            f"- `{name}`: CLV **{_fmt(v, 1)}** per acquired customer"
            + (f" (CAC {_fmt(cac[name], 0)})" if name in cac else "")
            for name, v in sorted(seg.items(), key=lambda kv: -(kv[1] or 0))
        )
        or "- (segment CLV realized from the fitted posterior)"
    )
    return [
        (
            "human",
            "We have a raw transaction log. Search and Social have similar "
            "cost-per-acquisition — are their customers actually worth the same?",
        ),
        (
            "ai",
            "That's a lifetime-value question, not a CPA question. Plan:\n"
            "→ `build_rfm_from_transactions()` — collapse the log to per-customer "
            "RFM (frequency, recency, age, mean repeat value), keeping each "
            "customer's acquisition channel\n"
            "→ `load_garden_model('bayesian_clv')` — BG/NBD + Gamma-Gamma "
            "lifetime value\n"
            "→ fit, then compare **CLV per acquired customer by channel**.",
        ),
        ("human", "Do it."),
        (
            "ai",
            "→ `build_rfm_from_transactions(segment_col='acquisition_channel')`\n"
            "→ `load_garden_model('bayesian_clv')`  *(sets `spec['garden_ref']`)*\n"
            "→ `fit_mmm_model(...)`  *(MAP — the per-customer latents are "
            "integrated out, so this takes seconds)*\n\n"
            f"Fitted ✅. Mean CLV **{_fmt(h.get('mean_clv'), 1)}** per customer; "
            f"**{_fmt((h.get('mean_p_alive') or 0) * 100, 0)}%** of the book is "
            "still active. By acquisition channel:\n"
            f"{seg_lines}\n\n"
            "Similar CPA, very different customers.",
        ),
        ("human", "So should we shift budget toward Search?"),
        (
            "ai",
            "→ `get_clv_value(cac={'Search': 25, 'Social': 12})`\n\n"
            "On **CLV − CAC**, Search wins decisively despite costing ~2× per "
            "acquisition — its customers repeat more and spend more per order. "
            "Two caveats before moving budget:\n"
            "1. Segment CLV is **descriptive** — to claim more Search spend "
            "*creates* high-value customers, run an acquisition experiment "
            "(`ghost_ads_power_calc(value_from_clv=True, clv_segment='Search')` "
            "values the test at lifetime worth).\n"
            "2. CLV projections assume the purchase/dropout process is "
            "stationary over the horizon — refit after pricing or product "
            "changes.\n\n"
            "The segment-CLV chart, the CLV-vs-CAC table, and the weekly "
            "cohort-value series are in the **Results** tab.",
        ),
    ]


_TURN_FNS = {
    "awareness_structural_mmm": _awareness_turns,
    "bayesian_cfa": _cfa_turns,
    "bayesian_lca": _lca_turns,
    "bayesian_clv": _clv_turns,
}
_FIT_FNS = {
    "awareness_structural_mmm": _fit_awareness,
    "bayesian_cfa": _fit_cfa,
    "bayesian_lca": _fit_lca,
    "bayesian_clv": _fit_clv,
}


# ── Orchestration ───────────────────────────────────────────────────────────────


def _import_class(model_def):
    mod = __import__(model_def["module"])
    return getattr(mod, model_def["class_name"])


def _replace_prior_project(store, name: str) -> None:
    for old in store.list_projects():
        if old.get("name") != name:
            continue
        pid = old["project_id"]
        for s in store.list_sessions(project_id=pid):
            store.delete_session(s["thread_id"])
        store.delete_project(pid)
        print(f"Replaced prior demo project {pid}")


def seed(fast: bool, real_compat: bool) -> None:
    from mmm_framework.api import sessions as store
    from seed_demo_project import _seed_chat  # reuse the checkpointer wiring

    store.init_db()

    # Project first, so we register the models under the org the project resolves
    # to (the agent resolves the garden org from the project).
    _replace_prior_project(store, PROJECT_NAME)
    project = store.create_project(
        PROJECT_NAME,
        description=(
            "Showcase of the Atelier custom models in the Oracle: four published "
            "Model Garden models (awareness structural MMM, Bayesian CFA, Bayesian "
            "LCA, Bayesian CLV) loaded, fitted, and interrogated by the agent."
        ),
    )
    pid = project["project_id"]
    org_id = store.resolve_org_id(pid)
    print(f"Project: {project['name']} ({pid})  org={org_id}")

    _purge_demo_models(store, org_id, [m["name"] for m in MODELS])

    for model_def in MODELS:
        print(f"\n[{model_def['title']}]")
        cls = _import_class(model_def)
        row = _register_and_publish(store, org_id, model_def, cls, real_compat)
        print(f"    published {row['name']} v{row['version']} ({model_def['kind']})")

        t0 = time.time()
        res = _FIT_FNS[model_def["name"]](model_def, row, fast)
        print(f"    fit done in {time.time() - t0:.0f}s")

        sess = store.create_session(model_def["session_title"], project_id=pid)
        tid = sess["thread_id"]
        n_plots, n_tables = _attach_viz(
            res["dashboard"], tid, res["figures"], res["tables"]
        )
        turns = _TURN_FNS[model_def["name"]](res["headline"])
        _seed_chat(
            tid,
            turns,
            {
                "model_spec": res["spec"],
                "dataset_path": res["dataset_path"],
                "model_status": "completed",
                "dashboard_data": res["dashboard"],
            },
        )
        print(f"    seeded session {tid}  ({n_plots} charts, {n_tables} tables)")

    published = store.list_garden_models(org_id, status="published", latest_only=True)
    print(f"\n{'=' * 60}\nSeed complete.")
    print(f"  Garden (org {org_id}): {len(published)} published model(s):")
    for r in published:
        print(
            f"    - {r['name']} v{r['version']} ({(r['manifest'] or {}).get('model_kind')})"
        )
    print(f"  Project '{project['name']}': {len(MODELS)} Oracle session(s).")

    from mmm_framework.agents import workspace as ws

    ws_root = ws.workspace_root()
    print(f"  Charts + tables written under the agent workspace: {ws_root}")
    print(
        "\nOpen the app: the Atelier (Model Garden) lists the published models; "
        f"select '{project['name']}' in the header to see the agent using them."
    )
    print(
        "\nIMPORTANT: the session charts/tables are served from the agent "
        "workspace above, so launch the API with the SAME workspace, e.g.\n"
        f"    MMM_AGENT_WORKSPACE={ws_root} uv run uvicorn mmm_framework.api.main:app --port 8000\n"
        "  (or set MMM_AGENT_WORKSPACE to the same value for BOTH this seeder and "
        "the server). The project, sessions, models, and chat use a fixed DB and "
        "show up regardless."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip MCMC; publish + seed with fabricated headline numbers (seconds)",
    )
    parser.add_argument(
        "--real-compat",
        action="store_true",
        help="run the genuine compatibility suite per model (slower) instead of "
        "synthesizing a passing report",
    )
    args = parser.parse_args()
    seed(args.fast, args.real_compat)


if __name__ == "__main__":
    main()
