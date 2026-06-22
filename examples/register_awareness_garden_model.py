"""Register the Persistent-Awareness Structural TS model into the Model Garden.

After running this, the model shows up in the **Atelier** (`/atelier` in the
React app) under the dev org — browse its source in the Monaco editor, read the
"About" docs, and view its compatibility report.

    uv run python examples/register_awareness_garden_model.py            # register/update + test → "tested"
    uv run python examples/register_awareness_garden_model.py --skip-test  # register/update only (fast)
    uv run python examples/register_awareness_garden_model.py --new-version # force a fresh version

The model source lives in ``examples/garden_models/awareness_structural_mmm.py``;
this script reads that file verbatim and stores it in the org-scoped garden so
the agent ("oracle") can load and re-fit it on any project. It is **idempotent**:
re-running propagates source edits into the existing (non-published) version in
place and re-runs the compatibility suite. Registration writes to the SAME
sessions DB + workspace the API serves from, so the Atelier picks it up
immediately (no restart needed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_SOURCE = REPO_ROOT / "examples" / "garden_models" / "awareness_structural_mmm.py"
MODEL_NAME = "awareness_structural_ts"

DOCS = """\
# Persistent-Awareness Structural Time-Series MMM

A bespoke MMM whose KPI is a **latent brand-awareness stock** rather than
contemporaneous sales. It answers: *how does media build awareness, and how long
does that awareness persist after the spend stops?*

## The structure

Awareness is a state-space **local level** (Nerlove–Arrow advertising goodwill):

```
Aₜ = intercept + Lₜ + Σ_c Sₜ,c          (baseline + organic level + media goodwill)

Lₜ   = ρ·Lₜ₋₁   + εₜ                     organic structural level   (εₜ ~ N(0, σ_level))
Sₜ,c = ρ·Sₜ₋₁,c + βc·sat_c(spendₜ,c)     per-channel goodwill stock (saturated inflow)

yₜ ~ Normal(Aₜ + seasonality + controls, σ)
```

The headline parameter is **`awareness_retention` ρ** (a `Beta` on `(0,1)`): the
share of awareness retained each period. Its **half-life `ln 0.5 / ln ρ`**
answers "how long does brand memory last?". Both the organic level and every
channel's goodwill decay at the same ρ, so each channel's carryover IS a
geometric adstock with `alpha = ρ` — the standard half-life / carryover
reporting reads it directly.

The decay accumulation is computed as a single vectorized lower-triangular
Toeplitz matmul (`Sₜ = Σ_{τ≤t} ρ^(t-τ)·inflow_τ`), NOT a `pytensor.scan` — same
math, but it compiles instantly and fits in seconds, so an in-process
compatibility test never bogs the app down.

## How it differs from the base MMM

| | Base `BayesianMMM` | This model |
|---|---|---|
| Persistence lives in | each channel's adstock | the **awareness state** (shared ρ) |
| KPI | sales / units | **awareness** (survey %, brand tracker, consideration) |
| Carryover after spend stops | per-channel decay | a single brand-memory half-life |
| `channel_contributions` | per-period response | accumulated **goodwill stock** by channel |

## Tuning per brand

Bespoke params live in the model's `CONFIG_SCHEMA` (`AwarenessParams`), settable
per-fit via `spec["model_params"]` (no longer hidden class attributes):

- `retention_prior_alpha` / `retention_prior_beta` — the `Beta` prior on the
  awareness retention ρ. The default `Beta(6, 2)` (mean 0.75) suits sticky
  categories (insurance, autos); lower the mean for impulse/promotional brands.
- `level_innovation_sigma` — kept tight (0.15) so media, not the latent level,
  explains the swings and channel effects stay identified.
- `number_of_trials` — the survey sample size; set `spec["likelihood"] =
  {"family": "binomial"}` to fit a Binomial survey-count awareness KPI instead
  of the default Normal awareness index.

The model declares default estimands (`awareness_lift`, `contribution_roi`,
`goodwill_stock`), so `evaluate_estimands()` with no args returns those.

## Scope & assumptions

- **National single series only.** It raises on geo/product panels rather than
  silently sharing one stock across cells; extend `_build_model` with a per-cell
  decay matrix to support hierarchical panels.
- Recommended fit: **NUTS** for calibrated uncertainty; `method="map"` for a
  fast structural sanity check.
- Compatibility: passes all blocking oracle tiers (build / fit / scaling /
  read-ops). The advisory accuracy tier scores it against a *sales*-ROI answer
  key, so magnitudes differ from a sales model by design — directional agreement
  is the meaningful signal there.
"""

DATASET_SCHEMA = {
    "kpi": "a persistent brand-awareness metric (aided/unaided awareness %, brand tracker, consideration)",
    "kpi_kind": "other",
    "level": "national",
    "geo_product": "not supported (single national stock)",
    "media_channels": "any number; spend per channel per period",
}

RECOMMENDED_FIT = {
    "method": "nuts",
    "draws": 1000,
    "tune": 1000,
    "target_accept": 0.9,
    "note": "Use method='map' for a fast structural sanity check before paying for NUTS.",
}

TAGS = [
    "awareness",
    "brand",
    "structural-time-series",
    "state-space",
    "persistence",
    "nerlove-arrow",
    "national",
]


def _cell(cid: str, ctype: str, source: str) -> dict:
    return {"id": cid, "type": ctype, "source": source, "outputs": None}


def _reset_atelier_notebook(sessions, org_id: str) -> None:
    """Overwrite the model's Atelier **draft** notebook with the curated
    DEMO_NOTEBOOK. The GET handler only AUTO-seeds the demo when no persisted doc
    exists, so a model opened before this demo existed keeps a stale notebook —
    re-running this script corrects it. (``tid`` mirrors
    ``api.main._notebook_tid(org, name, None)``.)"""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{MODEL_NAME}__draft")[:80]
    tid = f"__atelier_nb__{org_id}__{safe}"
    payload = {
        "cells": DEMO_NOTEBOOK,
        "dataset": None,
        "name": MODEL_NAME,
        "version": None,
        "org_id": org_id,
    }
    docs = [
        a for a in sessions.list_artifacts(tid) if a.get("kind") == "atelier_notebook"
    ]
    if docs:
        sessions.update_artifact_payload(docs[-1]["id"], payload)
    else:
        sessions.add_artifact(tid, "atelier_notebook", payload)
    print(
        f"  ✓ reset the Atelier notebook to the curated {len(DEMO_NOTEBOOK)}-cell demo"
    )


# A curated, runnable walkthrough seeded into the Atelier notebook the first time
# this model is opened (`manifest["demo_notebook"]`). It fits the model on a
# synthetic AWARENESS SURVEY and shows how to pass the model's bespoke specs
# (binomial likelihood + number_of_trials / retention priors via `model_params`).
DEMO_NOTEBOOK = [
    _cell(
        "c1",
        "markdown",
        "# Demo: Persistent-Awareness Structural MMM\n\n"
        "This model treats the KPI as a **latent brand-awareness stock** built by "
        "media, not contemporaneous sales. Its headline parameter is the "
        "**retention** ρ — the share of awareness carried over each week — whose "
        "**half-life `ln 0.5 / ln ρ`** answers *how long brand memory lasts after "
        "the spend stops*.\n\n"
        "Run the cells to fit it on a synthetic **awareness survey** (or upload your "
        "own MFF data). **Cell 3** shows how to pass this model's *bespoke "
        "specifications* — the binomial survey-count likelihood and "
        "`number_of_trials` / retention priors via `model_params`.",
    ),
    _cell(
        "c2",
        "code",
        "# Synthetic brand-awareness survey, or your uploaded `df`. KPI 'Awareness'\n"
        "# is a weekly COUNT of aware respondents out of `n_trials` surveyed;\n"
        "# awareness is a media goodwill stock that decays at a retention rho.\n"
        "n_trials = 500\n"
        "try:\n"
        "    data, src, kpi, channels = df, dataset_path, 'Awareness', None\n"
        "except NameError:\n"
        "    from mmm_framework.synth import make_awareness_survey\n"
        "    data, answer = make_awareness_survey(n_weeks=104, n_trials=n_trials,\n"
        "                                         retention=0.8, seed=7)\n"
        "    src = 'awareness.csv'; data.to_csv(src, index=False)\n"
        "    kpi, channels = 'Awareness', answer['channels']\n"
        "    print('Planted retention rho =', answer['true_retention'],\n"
        "          '-> half-life', round(answer['true_half_life_weeks'], 1), 'weeks')\n"
        "show_table(data.head(20), title='Awareness survey (MFF long-format)')\n",
    ),
    _cell(
        "c3",
        "code",
        "# Pass this model's BESPOKE specs from the spec dict:\n"
        "#   likelihood.family='binomial'   -> awareness is a survey COUNT (logit link)\n"
        "#   model_params.number_of_trials  -> the binomial denominator (survey size)\n"
        "#   model_params.retention_prior_* -> the Beta prior on the retention rho\n"
        "# (model_params maps to the model's CONFIG_SCHEMA — edit them and re-run.)\n"
        "all_vars = list(dict.fromkeys(data['VariableName'].tolist()))\n"
        "if not channels:\n"
        "    channels = [v for v in all_vars if v != kpi]\n"
        "spec = {\n"
        "    'kpi': kpi,\n"
        "    'media_channels': [{'name': c} for c in channels],\n"
        "    'trend': {'type': 'none'},  # the awareness state IS the trend\n"
        "    'seasonality': {'yearly': 0, 'monthly': 0, 'weekly': 0},\n"
        "    'likelihood': {'family': 'binomial'},\n"
        "    'model_params': {\n"
        "        'number_of_trials': n_trials,\n"
        "        'retention_prior_alpha': 6.0,\n"
        "        'retention_prior_beta': 2.0,\n"
        "    },\n"
        "    'inference': {'method': 'map'},\n"
        "}\n"
        "from mmm_framework.agents.fitting import build_model\n"
        "mmm = build_model(spec, src, model_cls=GardenModel)\n"
        "results = mmm.fit(method='map')\n"
        "print('Fitted', GardenModel.__name__, '(binomial awareness) on:', channels)\n",
    ),
    _cell(
        "c4",
        "code",
        "# Recovered retention + half-life, and observed vs fitted aware-rate.\n"
        "import numpy as np, plotly.graph_objects as go\n"
        "rho = float(mmm._trace.posterior['awareness_retention'].values.mean())\n"
        "half_life = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float('inf')\n"
        "print('Recovered retention rho =', round(rho, 3),\n"
        "      '-> half-life', round(half_life, 1), 'weeks')\n"
        "obs = (data[data['VariableName'] == kpi].sort_values('Period')['VariableValue']\n"
        "       .values / n_trials)\n"
        "post = mmm._trace.posterior\n"
        "fig = go.Figure()\n"
        "fig.add_scatter(y=obs, name='observed aware-rate', mode='lines')\n"
        "if 'awareness_rate' in post:\n"
        "    fig.add_scatter(y=post['awareness_rate'].mean(('chain', 'draw')).values,\n"
        "                    name='fitted aware-rate', mode='lines')\n"
        "fig.update_layout(title='Brand awareness: observed vs fitted',\n"
        "                  yaxis_title='aware-rate')\n"
        "fig.show()\n",
    ),
    _cell(
        "c5",
        "code",
        "# Declared estimands: mean awareness lift + per-channel goodwill-stock\n"
        "# latent contrasts (latent media_total under media-on vs each channel off).\n"
        "import pandas as pd\n"
        "res = mmm.evaluate_estimands()\n"
        "rows = [{'estimand': k, 'mean': r.mean, 'hdi_low': r.hdi_low,\n"
        "         'hdi_high': r.hdi_high, 'status': r.status, 'units': r.units}\n"
        "        for k, r in res.items()]\n"
        "show_table(pd.DataFrame(rows),\n"
        "           title='Awareness estimands (counterfactual quantities)')\n",
    ),
    _cell(
        "c6",
        "markdown",
        "## Try it\n\n"
        "- **Edit the bespoke specs in cell 3** and re-run: raise `number_of_trials` "
        "(a larger survey → a tighter binomial), or shift "
        "`retention_prior_alpha/beta` to encode a stickier or more impulsive "
        "category.\n"
        "- Swap `likelihood.family` to `'normal'` to treat the KPI as a continuous "
        "awareness *index* instead of a survey count.\n"
        "- Upload your own MFF survey data with the control above to fit on real "
        "awareness.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Register/update only; skip the compat suite + promotion to 'tested'.",
    )
    parser.add_argument(
        "--new-version",
        action="store_true",
        help="Register a new version even if this model already exists.",
    )
    args = parser.parse_args()

    from mmm_framework.agents.garden_registry import (
        register_garden_model_core,
        static_class_name,
        static_model_kind,
    )
    from mmm_framework.api import sessions
    from mmm_framework.garden.contract import GARDEN_CONTRACT_VERSION

    sessions.init_db()
    org_id = sessions.DEFAULT_ORG_ID
    source_code = MODEL_SOURCE.read_text(encoding="utf-8")

    class_name, err = static_class_name(source_code)
    if err:
        raise SystemExit(f"Cannot register: {err}")

    # Import the model class to read its CONFIG_SCHEMA (AwarenessParams) as a JSON
    # Schema — host registration is AST-only, so the demo supplies the schema for
    # the UI params form. These new manifest fields are advisory metadata.
    sys.path.insert(0, str(MODEL_SOURCE.parent))
    from awareness_structural_mmm import AwarenessStructuralMMM

    config_schema = AwarenessStructuralMMM.CONFIG_SCHEMA.model_json_schema()
    DEFAULT_ESTIMANDS = ["awareness_lift", "contribution_roi", "goodwill_stock"]
    CAPABILITIES = ["HAS_CHANNELS", "HAS_CONTRIBUTIONS", "HAS_LATENT:media_total"]

    manifest = {
        "contract_version": GARDEN_CONTRACT_VERSION,
        "class_name": class_name,
        "model_kind": static_model_kind(source_code, class_name),
        "dataset_schema": DATASET_SCHEMA,
        "recommended_fit": RECOMMENDED_FIT,
        "tags": TAGS,
        "config_schema": config_schema,
        "default_estimands": DEFAULT_ESTIMANDS,
        "capabilities": CAPABILITIES,
        "demo_notebook": DEMO_NOTEBOOK,
    }

    existing = sessions.list_garden_models(org_id, name=MODEL_NAME, latest_only=True)
    if existing and not args.new_version:
        row = existing[0]
        if row["status"] == "published":
            print(
                f"'{MODEL_NAME}' v{row['version']} is PUBLISHED (immutable). "
                "Re-run with --new-version to register a new version."
            )
            return
        # Idempotent in-place update: overwrite the stored source + manifest so a
        # re-test loads the latest code (the loader keys its cache on mtime).
        src_path = Path(row["source_path"])
        src_path.write_text(source_code, encoding="utf-8")
        (src_path.parent / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        row = sessions.upsert_garden_model(
            org_id=org_id,
            name=MODEL_NAME,
            version=row["version"],
            model_id=row["id"],
            docs=DOCS,
            manifest=manifest,
        )
        from mmm_framework.garden import clear_cache

        clear_cache()
        print(
            f"Updated existing '{MODEL_NAME}' v{row['version']} "
            f"(status='{row['status']}') source in place → {src_path}"
        )
    else:
        print(f"Registering '{MODEL_NAME}' into the Model Garden (org='{org_id}')…")
        row = register_garden_model_core(
            org_id=org_id,
            source_code=source_code,
            name=MODEL_NAME,
            docs=DOCS,
            tags=TAGS,
            dataset_schema=DATASET_SCHEMA,
            recommended_fit=RECOMMENDED_FIT,
            config_schema=config_schema,
            default_estimands=DEFAULT_ESTIMANDS,
            capabilities=CAPABILITIES,
            demo_notebook=DEMO_NOTEBOOK,
        )
        print(
            f"  ✓ registered v{row['version']} as DRAFT "
            f"(class {row['manifest'].get('class_name')}) → {row['source_path']}"
        )

    model_id = row["id"]
    status_now = row["status"]

    # Refresh the Atelier notebook to the curated walkthrough (corrects a stale
    # doc that would otherwise block the demo from seeding).
    _reset_atelier_notebook(sessions, org_id)

    if args.skip_test:
        print(
            "\nSkipped compatibility testing (--skip-test). Click 'Test' in the "
            "Atelier (or re-run without --skip-test)."
        )
        _print_done(org_id)
        return

    print("\nRunning the compatibility suite (vectorized MAP fit; ~10s)…")
    from mmm_framework.garden import run_compatibility_check
    from mmm_framework.garden.loader import load_garden_class_from_path

    cls = load_garden_class_from_path(row["source_path"], class_name)
    report = run_compatibility_check(cls, fit_method="map", seed=7, n_weeks=104)
    sessions.set_garden_compat_report(model_id, report)
    passed = report.get("blocking_passed")
    print(f"  blocking_passed={passed}  advisory_score={report.get('score')}")
    for tier in report["tiers"]:
        mark = "skip" if tier["skipped"] else ("pass" if tier["passed"] else "FAIL")
        print(f"    [{mark}] {tier['name']}")

    if passed and status_now == "draft":
        sessions.transition_garden_model(
            model_id, "tested", note="Auto-tested by register_awareness_garden_model.py"
        )
        print("  ✓ promoted DRAFT → TESTED (blocking tiers passed)")
    elif passed:
        print(f"  ✓ compat report refreshed (model stays '{status_now}')")
    else:
        print("  ⚠ blocking tiers did NOT pass — see the report above.")

    _print_done(org_id)


def _print_done(org_id: str) -> None:
    print(
        "\nDone. Start the app (see the run-app skill / CLAUDE.md) and open the "
        "Atelier at /atelier — the model is listed under org "
        f"'{org_id}'. Or load it in any project chat:\n"
        f"    list_garden_models()  →  load_garden_model('{MODEL_NAME}')  →  fit_mmm_model()"
    )


if __name__ == "__main__":
    main()
