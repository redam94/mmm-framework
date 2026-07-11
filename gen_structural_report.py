"""Fit the NestedSurveyMediationMMM (survey-anchored structural mediation MMM)
on the aurora known-truth world and generate the full interactive results report.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

REPO = Path("/Users/redam94/mmm-framework/.claude/worktrees/interactive-results-report")
sys.path.insert(0, str(REPO / "examples" / "garden_models"))
# aurora.py moved to nbs/builders/ in the topic-folder reorg; put it on the path
# so aurora_mediation_dataset()'s `from aurora import ...` resolves.
sys.path.insert(0, str(REPO / "nbs" / "builders"))

import numpy as np  # noqa: E402

from mmm_framework.config import ModelConfig  # noqa: E402
from mmm_framework.model import TrendConfig, TrendType  # noqa: E402
from mmm_framework.reporting.interactive import InteractiveReportGenerator  # noqa: E402
from nested_survey_mediation_mmm import (  # noqa: E402
    NestedSurveyMediationMMM,
    aurora_mediation_dataset,
)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / (
    "structural_mmm_interactive_report.html"
)

print(">> building aurora role-tagged dataset (survey mediator)…", flush=True)
ds, A = aurora_mediation_dataset()
print(f"   channels: {list(A.spend.columns)}", flush=True)
print(
    "   TRUE proportion_mediated:",
    A.true_mediated_share[["TV", "Display"]].round(3).to_dict(),
    flush=True,
)

print(">> fitting NestedSurveyMediationMMM (NUTS / NumPyro)…", flush=True)
t0 = time.time()
mmm = NestedSurveyMediationMMM(
    ds, ModelConfig(), TrendConfig(type=TrendType.SPLINE)
)
results = mmm.fit(
    draws=1000, tune=1500, chains=4, target_accept=0.92, random_seed=0
)
print(f"   fit done in {time.time() - t0:.0f}s", flush=True)

med = mmm.get_mediation_effects()
print("   recovered proportion_mediated:\n" + med.round(3).to_string(), flush=True)
roas = mmm.get_channel_roas()
print("   recovered total-effect ROAS:\n" + roas.round(2).to_string(), flush=True)
try:
    rhat = results.diagnostics.get("rhat_max")
    print(f"   max R-hat: {rhat}", flush=True)
except Exception:
    pass

print(">> generating full interactive report…", flush=True)
t1 = time.time()
gen = InteractiveReportGenerator(
    mmm,
    results,
    max_draws=200,
    curve_max_draws=120,
    n_prior_samples=300,
    random_seed=42,
)
f = gen.facts
print(
    "   facts: mediation=%s latent=%s ppc_loopit=%s channels=%s periods=%d"
    % (
        f.get("mediation") is not None,
        f.get("latent") is not None,
        (f.get("ppc_stats") or {}).get("loo_pit") is not None,
        f["meta"]["channels"],
        f["meta"]["n_periods"],
    ),
    flush=True,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
gen.save_report(str(OUT))
print(f"   report written in {time.time() - t1:.0f}s", flush=True)
print(f">> REPORT: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)", flush=True)

# Emit a tiny fact digest for the summary.
import json  # noqa: E402

digest = {
    "out": str(OUT),
    "size_kb": round(OUT.stat().st_size / 1024),
    "channels": f["meta"]["channels"],
    "n_periods": f["meta"]["n_periods"],
    "has_mediation": f.get("mediation") is not None,
    "has_latent": f.get("latent") is not None,
    "has_loopit": (f.get("ppc_stats") or {}).get("loo_pit") is not None,
    "mediation_links": len((f.get("mediation") or {}).get("links", [])),
    "true_prop_mediated": A.true_mediated_share[["TV", "Display"]].round(3).to_dict(),
    "recovered_prop_mediated": med["proportion_mediated"].round(3).to_dict(),
    "recovered_roas": roas["roas"].round(2).to_dict(),
}
Path("/Users/redam94/.claude/jobs/1a2d0cc2/tmp/structural_digest.json").write_text(
    json.dumps(digest, indent=2)
)
print("DIGEST " + json.dumps(digest), flush=True)
