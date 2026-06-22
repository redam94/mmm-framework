"""Register the Bayesian Latent Class Analysis model into the Model Garden.

A **non-MMM** garden model (customer segmentation over binary survey items) — the
Atelier demo for the non-MMM family path. After running, it shows up in the
Atelier (`/atelier`) under the dev org with a curated segmentation walkthrough
notebook.

    uv run python examples/register_lca_garden_model.py            # register/update + test → "tested"
    uv run python examples/register_lca_garden_model.py --skip-test  # register/update only (fast)
    uv run python examples/register_lca_garden_model.py --new-version # force a fresh version

The model source lives in ``examples/garden_models/bayesian_lca.py``; this script
reads it verbatim and stores it in the org-scoped garden. Idempotent: re-running
propagates source edits into the existing (non-published) version + re-runs the
compatibility suite. Mirrors ``register_awareness_garden_model.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_SOURCE = REPO_ROOT / "examples" / "garden_models" / "bayesian_lca.py"
MODEL_NAME = "latent_class_analysis"

DOCS = """\
# Bayesian Latent Class Analysis (LCA)

A **non-MMM** measurement model: a mixture that discovers **latent segments**
(classes) in a set of **binary indicators** (e.g. yes/no survey items). There are
no media channels, no spend, no single KPI — it answers *"are there distinct
segments here, how large is each, and how does each respond to each item?"*.

## The structure

Each respondent belongs to one of ``K`` unobserved classes; class ``k`` has its
own item-endorsement profile θₖⱼ = P(itemⱼ = 1 | classₖ). The discrete labels are
**integrated out** (so NUTS works): each respondent's log-likelihood is a
`logsumexp` log-mixture of products of Bernoullis. Mixing proportions π are a
softmax of an **ordered** logit (pins class order by size → resolves the
label-switching symmetry); θ are `Beta` priors.

## How it plugs into the framework (non-MMM)

It declares `__garden_model_kind__ = "latent_class"`, which exempts it from the
MMM-specific garden gates (channels, `beta_<channel>`, channel read-ops/compat
tiers). It overrides only `_prepare_data` (binarize the observed columns) and
`_build_model`, reusing fit / serialization / the estimand engine. Its quantities
of interest are **class sizes** (estimands `class_size_k`) and the per-class item
profiles (`class_profile_summary()`), surfaced through the same `evaluate_estimands`
+ report pipeline as an MMM's ROI (the report's family-agnostic latent-structure
section).

## Config (CONFIG_SCHEMA = LCAConfig)

- `n_classes` — number of latent segments (≥ 2).
- `item_prior_a` / `item_prior_b` — Beta prior on each item probability.
- `binarize_threshold` — threshold non-0/1 columns (default: per-item median).

## Scope

Binary indicators, a single cross-section. Pass `model_params={"n_classes": K}` to
choose the number of segments; recommended fit is **NUTS** for calibrated class
sizes, `method="map"` for a fast structural check.
"""

DATASET_SCHEMA = {
    "kpi": "any binary indicator column (the model treats ALL observed columns as items)",
    "kpi_kind": "binary",
    "level": "cross-section (one row per respondent/period)",
    "geo_product": "not used",
    "indicators": "two or more binary (0/1) columns",
}

RECOMMENDED_FIT = {
    "method": "nuts",
    "draws": 1000,
    "tune": 1000,
    "target_accept": 0.9,
    "note": "Use method='map' for a fast structural check; n_classes via model_params.",
}

TAGS = ["latent-class", "segmentation", "mixture", "non-mmm", "factor-analysis"]

DEFAULT_ESTIMANDS = ["class_size_1", "class_size_2"]
CAPABILITIES = ["HAS_LATENT:class_size_1", "HAS_LATENT:class_size_2"]


def _cell(cid: str, ctype: str, source: str) -> dict:
    return {"id": cid, "type": ctype, "source": source, "outputs": None}


# Curated segmentation walkthrough seeded into the Atelier notebook on first open.
DEMO_NOTEBOOK = [
    _cell(
        "c1",
        "markdown",
        "# Demo: Latent Class Analysis (customer segmentation)\n\n"
        "A **non-MMM** model — no channels, no spend. Given a set of **binary survey "
        "items**, it discovers latent **segments** and recovers each segment's size "
        "and item-endorsement profile.\n\n"
        "Run the cells to fit it on synthetic data with a known 2-segment structure "
        "(or upload your own binary MFF data) and recover the segments. Choose the "
        "number of segments via the model's bespoke `model_params={'n_classes': K}`.",
    ),
    _cell(
        "c2",
        "code",
        "# Synthetic 2-segment binary survey, or your uploaded `df`. Segment A "
        "endorses\n# items q1-q3, segment B endorses q4-q6.\n"
        "import numpy as np, pandas as pd\n"
        "try:\n"
        "    data, src = df, dataset_path\n"
        "except NameError:\n"
        "    rng = np.random.default_rng(11)\n"
        "    n, sizes = 600, [0.4, 0.6]\n"
        "    profiles = np.array([[.85,.85,.85,.15,.15,.15],\n"
        "                         [.15,.15,.15,.85,.85,.85]])\n"
        "    z = rng.choice(2, size=n, p=sizes)\n"
        "    Y = (rng.random((n, 6)) < profiles[z]).astype(int)\n"
        "    cols = [f'q{j+1}' for j in range(6)]\n"
        "    wide = pd.DataFrame(Y, columns=cols)\n"
        "    wide.insert(0, 'Period', pd.date_range('2021-01-04', periods=n,\n"
        "                freq='W-MON').strftime('%Y-%m-%d'))\n"
        "    for c in ['Geography','Product','Campaign','Outlet','Creative']:\n"
        "        wide[c] = None\n"
        "    idv = ['Period','Geography','Product','Campaign','Outlet','Creative']\n"
        "    data = wide.melt(id_vars=idv, var_name='VariableName',\n"
        "                     value_name='VariableValue')\n"
        "    src = 'segments.csv'; data.to_csv(src, index=False)\n"
        "    print('Planted 2 segments (sizes', sizes, ') over 6 binary items.')\n"
        "show_table(data.head(20), title='Binary survey items (MFF long-format)')\n",
    ),
    _cell(
        "c3",
        "code",
        "# Fit with the bespoke spec: model_params={'n_classes': K} (the model's\n"
        "# CONFIG_SCHEMA). All observed columns are treated as binary items.\n"
        "all_vars = list(dict.fromkeys(data['VariableName'].tolist()))\n"
        "spec = {\n"
        "    'kpi': all_vars[0],\n"
        "    'media_channels': [{'name': c} for c in all_vars[1:]],\n"
        "    'trend': {'type': 'none'},\n"
        "    'seasonality': {'yearly': 0, 'monthly': 0, 'weekly': 0},\n"
        "    'model_params': {'n_classes': 2},\n"
        "    'inference': {'method': 'map'},\n"
        "}\n"
        "from mmm_framework.agents.fitting import build_model\n"
        "mmm = build_model(spec, src, model_cls=GardenModel)\n"
        "results = mmm.fit(method='map')\n"
        "print('Fitted', GardenModel.__name__, 'with',\n"
        "      mmm.model_params.n_classes, 'classes on', mmm.n_items, 'items')\n",
    ),
    _cell(
        "c4",
        "code",
        "# Recovered segment sizes (the class-size estimands).\n"
        "import pandas as pd, plotly.express as px\n"
        "res = mmm.evaluate_estimands()\n"
        "sizes = pd.DataFrame([{'segment': k, 'size': r.mean}\n"
        "                      for k, r in res.items() if r.status == 'ok'])\n"
        "show_table(sizes, title='Recovered segment sizes (mixing proportions)')\n"
        "px.bar(sizes, x='segment', y='size',\n"
        "       title='Segment sizes').show()\n",
    ),
    _cell(
        "c5",
        "code",
        "# Recovered item-endorsement profiles P(item=1 | segment) — the\n"
        "# interpretable LCA output (segment A endorses q1-q3, B endorses q4-q6).\n"
        "import plotly.express as px\n"
        "prof = mmm.class_profile_summary().pivot(index='item', columns='class',\n"
        "                                         values='prob')\n"
        "show_table(prof.round(2).reset_index(),\n"
        "           title='P(item = 1 | segment) — recovered profiles')\n"
        "px.imshow(prof, color_continuous_scale='Blues', zmin=0, zmax=1,\n"
        "          aspect='auto', text_auto='.2f',\n"
        "          title='Segment item-endorsement profiles').show()\n",
    ),
    _cell(
        "c6",
        "markdown",
        "## Try it\n\n"
        "- **Change `model_params['n_classes']` in cell 3** to look for 3+ segments.\n"
        "- Upload your own binary MFF survey data with the control above.\n"
        "- The model declares `__garden_model_kind__ = 'latent_class'`, so it rides "
        "the same fit / estimand / report pipeline as an MMM — the report's "
        "**latent-structure** section shows these profiles + sizes.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Register/update only; skip the compat suite.",
    )
    parser.add_argument(
        "--new-version",
        action="store_true",
        help="Register a new version even if this model exists.",
    )
    args = parser.parse_args()

    from mmm_framework.agents.garden_registry import (
        register_garden_model_core,
        static_class_name,
        static_model_kind,
    )
    from mmm_framework.api import sessions
    from mmm_framework.garden.contract import GARDEN_CONTRACT_VERSION

    sys.path.insert(0, str(MODEL_SOURCE.parent))
    from bayesian_lca import BayesianLCA

    sessions.init_db()
    org_id = sessions.DEFAULT_ORG_ID
    source_code = MODEL_SOURCE.read_text(encoding="utf-8")
    config_schema = BayesianLCA.CONFIG_SCHEMA.model_json_schema()

    class_name, err = static_class_name(source_code)
    if err:
        raise SystemExit(f"Cannot register: {err}")
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
                "Re-run with --new-version."
            )
            return
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

    if args.skip_test:
        print(
            "\nSkipped compatibility testing (--skip-test). Click 'Test' in the Atelier."
        )
        _print_done(org_id)
        return

    print("\nRunning the compatibility suite (MAP fit; ~10s)…")
    from mmm_framework.garden import run_compatibility_check
    from mmm_framework.garden.loader import load_garden_class_from_path

    cls = load_garden_class_from_path(row["source_path"], class_name)
    report = run_compatibility_check(cls, fit_method="map", seed=7, n_weeks=60)
    sessions.set_garden_compat_report(model_id, report)
    passed = report.get("blocking_passed")
    print(f"  blocking_passed={passed}  advisory_score={report.get('score')}")
    for tier in report["tiers"]:
        mark = "skip" if tier["skipped"] else ("pass" if tier["passed"] else "FAIL")
        print(f"    [{mark}] {tier['name']}")

    if passed and status_now == "draft":
        sessions.transition_garden_model(
            model_id, "tested", note="Auto-tested by register_lca_garden_model.py"
        )
        print("  ✓ promoted DRAFT → TESTED (blocking tiers passed)")
    elif passed:
        print(f"  ✓ compat report refreshed (model stays '{status_now}')")
    else:
        print("  ⚠ blocking tiers did NOT pass — see the report above.")

    _print_done(org_id)


def _print_done(org_id: str) -> None:
    print(
        "\nDone. Start the app (run-app skill / CLAUDE.md) and open the Atelier at "
        f"/atelier — '{MODEL_NAME}' is listed under org '{org_id}' with a "
        "segmentation walkthrough notebook."
    )


if __name__ == "__main__":
    main()
