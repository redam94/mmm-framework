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

- `RETENTION_PRIOR` — `Beta(6, 2)` (mean 0.75) suits sticky categories
  (insurance, autos). Lower the mean for impulse/promotional brands.
- `LEVEL_INNOVATION_SIGMA` — kept tight (0.15) so media, not the latent level,
  explains the swings and channel effects stay identified.

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
    )
    from mmm_framework.api import sessions
    from mmm_framework.garden.contract import GARDEN_CONTRACT_VERSION

    sessions.init_db()
    org_id = sessions.DEFAULT_ORG_ID
    source_code = MODEL_SOURCE.read_text(encoding="utf-8")

    class_name, err = static_class_name(source_code)
    if err:
        raise SystemExit(f"Cannot register: {err}")
    manifest = {
        "contract_version": GARDEN_CONTRACT_VERSION,
        "class_name": class_name,
        "dataset_schema": DATASET_SCHEMA,
        "recommended_fit": RECOMMENDED_FIT,
        "tags": TAGS,
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
        )
        print(
            f"  ✓ registered v{row['version']} as DRAFT "
            f"(class {row['manifest'].get('class_name')}) → {row['source_path']}"
        )

    model_id = row["id"]
    status_now = row["status"]

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
