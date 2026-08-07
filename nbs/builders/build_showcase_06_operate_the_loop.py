"""Author showcase/showcase_06_operate_the_loop.ipynb (run from ``nbs/``).

    uv run --with nbformat python builders/build_showcase_06_operate_the_loop.py
    TQDM_DISABLE=1 PYTHONPATH=.. uv run --with nbconvert --with nbformat --with ipykernel \
        jupyter nbconvert --to notebook --execute --inplace \
        showcase/showcase_06_operate_the_loop.ipynb --ExecutePreprocessor.timeout=2400 \
        --ExecutePreprocessor.kernel_name=python3

Series finale: the operate/production layer. Experiment planning on a fitted
model (EIG/EVOI priorities, design + model-anchored opportunity cost, the
Pareto experiment optimizer, information decay and the re-test trigger), the
model-free continuous-learning bandit (CCD design + funding line, no pre-fit
MMM needed), reporting beyond to_html (pre-fit Model Design Readout,
interactive recompute-in-browser report, the Augur deck engine), async fits
via jobs.py, serialization flavors, security (encryption + PII scan), auth
(in-process JWT), the platform sessions store (record a run, list it), the
LangGraph oracle architecture, and the app layer. Every number is computed
in-notebook.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


SETUP = r"""
import warnings; warnings.filterwarnings("ignore")
import os, time
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
pd.set_option("display.width", 170)

ART = Path.cwd().parent / "artifacts" / "showcase"
ART.mkdir(parents=True, exist_ok=True)

# Point every stateful service at the artifacts sandbox BEFORE any
# mmm_framework import: the sessions store and the agent workspace both
# resolve their location at import time.
os.environ["MMM_SESSIONS_DB"] = str(ART / "showcase_06_sessions.db")
os.environ["MMM_AGENT_WORKSPACE"] = str(ART / "showcase_06_workspace")

import logging
from loguru import logger
logger.disable("mmm_framework")
for _n in ("pymc", "pymc.sampling", "pymc.stats.convergence",
           "numpyro", "jax", "arviz", "pytensor"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.propagate = False

INK, MUTED, TRUTH = "#1f2430", "#8a8f98", "#111418"
GOOD, BAD, GOLD = "#3d7a5c", "#b4552d", "#c9962e"
PALETTE = {"TV": "#4464ad", "Search": "#c9962e", "Social": "#3d7a5c", "Display": "#b4552d"}

def style(fig, height=380, title=None, **kw):
    fig.update_layout(height=height, title=title, margin=dict(t=64, l=64, r=30, b=52),
                      font=dict(size=12), **kw)
    return fig

print("Setup ready. Artifacts ->", ART)
"""

FIT = r"""
from mmm_framework.synth import dgp_geo
from mmm_framework.synth.mff import geo_scenario_to_mff
from mmm_framework.agents.fitting import build_and_fit

KPI = "Sales"
GEOS = ["North", "South", "East", "West", "Metro", "Coast", "Plains", "Hills"]
sc = dgp_geo.build("geo_heterogeneous", seed=3, geos=GEOS, n_weeks=78)
GEO_CSV = ART / "showcase_06_geo_world.csv"
geo_scenario_to_mff(sc).to_csv(GEO_CSV, index=False)
CHANNELS = ["TV", "Search", "Social", "Display"]

spec = {
    "kpi": KPI, "kpi_level": "geo",
    "media_channels": [{"name": n} for n in CHANNELS],
    "control_variables": [],
    "inference": {"draws": 150, "tune": 150, "chains": 2, "random_seed": 0},
    "seasonality": {"yearly": 2},
    "trend": {"type": "linear"},
}
t0 = time.time()
model, results, info = build_and_fit(spec, str(GEO_CSV))
print(f"build_and_fit in {time.time()-t0:.0f}s — {len(model.channel_names)} channels, "
      f"{model.n_geos} geos, {model.n_periods} periods (approximate={results.approximate})")
print(f"r-hat max {results.diagnostics['rhat_max']:.3f}, "
      f"divergences {int(results.diagnostics['divergences'])}")
"""

PRIORITY = r"""
from mmm_framework.planning import compute_experiment_priorities

t0 = time.time()
grid, portfolio = compute_experiment_priorities(model, max_draws=120, random_seed=42)
print(f"priority grid in {time.time()-t0:.0f}s")
gdf = pd.DataFrame([g.to_dict() for g in grid])[
    ["channel", "spend_share", "roi_mean", "roi_sd", "sigma_exp",
     "eig", "evoi", "quadrant", "priority"]
]
print(gdf.round(3).to_string(index=False))
print(f"\nportfolio EVPI (value of PERFECT information): {portfolio['evpi']:,.1f} KPI-units")

fig = go.Figure()
for _, r in gdf.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["eig"]], y=[r["evoi"]], mode="markers+text", text=[r["channel"]],
        textposition="top center",
        marker=dict(size=14 + 60 * r["spend_share"], color=PALETTE[r["channel"]],
                    line=dict(color="white", width=1.5)), showlegend=False))
fig.add_vline(x=portfolio["eig_threshold"], line=dict(color=MUTED, width=1, dash="dot"))
fig.add_hline(y=portfolio["evoi_threshold"], line=dict(color=MUTED, width=1, dash="dot"))
style(fig, 420, "A test must teach a lot AND move the budget — only the "
                "top-right quadrant earns an experiment")
fig.update_layout(xaxis_title="EIG (nats) — what a test would teach",
                  yaxis_title="EVOI (KPI units) — what the learning is worth")
fig.show()
TOP = gdf.iloc[0]["channel"]
print(f"top priority: {TOP} ({gdf.iloc[0]['quadrant']})")
"""

DECAY = r"""
from mmm_framework.planning.eig import (DEFAULT_RETEST_THRESHOLD_NATS, decayed_sigma,
                                        eig_gaussian, information_half_life,
                                        reexperiment_due)

sigma_post, sigma_exp = 0.30, 0.20     # a calibrated posterior; an achievable test
weeks = np.arange(0, 105)
fig = go.Figure()
for name, color in [("Paid_Search_Brand", PALETTE["Search"]),
                    ("Linear_TV", PALETTE["TV"]),
                    ("Mystery_Channel", MUTED)]:
    hl = information_half_life(name)
    eigs = [eig_gaussian(decayed_sigma(sigma_post, w, hl), sigma_exp) for w in weeks]
    fig.add_trace(go.Scatter(x=weeks, y=eigs, mode="lines", name=f"{name} (t½={hl:.0f}w)",
                             line=dict(color=color, width=2.2)))
fig.add_hline(y=DEFAULT_RETEST_THRESHOLD_NATS, line=dict(color=BAD, width=1.5, dash="dash"),
              annotation_text="re-test trigger")
style(fig, 380, "Evidence rots on a clock — a fresh test becomes worth running "
                "again once decay crosses the trigger")
fig.update_layout(xaxis_title="weeks since the last calibrated experiment",
                  yaxis_title="EIG of a fresh test (nats)")
fig.show()

for w in (8, 30, 80):
    due, eig = reexperiment_due(sigma_post, w, information_half_life("Paid_Search_Brand"),
                                sigma_exp)
    print(f"Search evidence {w:>2}w old -> EIG {eig:.3f} nats, re-test due: {due}")
"""

DESIGN = r"""
from mmm_framework.planning import design_experiment

plan = design_experiment(str(GEO_CSV), KPI, TOP, design="holdout", n_pairs=4, duration=8)
asg = pd.DataFrame(plan["assignment"])[["treatment", "control", "correlation"]]
print("matched pairs (treated market goes dark; its twin keeps spending):")
print(asg.round(3).to_string(index=False))
print(f"\nMDE(ROAS) at {plan['duration']}w: {plan['mde_roas']:.2f}")

pc = pd.DataFrame(plan["power_curve"])
fig = go.Figure(go.Scatter(x=pc["duration"], y=pc["mde_roas"], mode="lines+markers",
                           line=dict(color=PALETTE[TOP], width=3), showlegend=False))
fig.add_vline(x=plan["duration"], line=dict(color=MUTED, width=1.5, dash="dash"),
              annotation_text=f"chosen {plan['duration']}w")
style(fig, 340, f"{TOP} geo holdout: every extra week buys a smaller detectable effect")
fig.update_layout(xaxis_title="test duration (weeks)", yaxis_title="MDE (ROAS)")
fig.show()
"""

OPPCOST = r"""
from mmm_framework.planning import compute_opportunity_cost

MARGIN = 0.5
row = gdf.set_index("channel").loc[TOP]
oc = compute_opportunity_cost(model, plan, margin_per_kpi=MARGIN, kpi_kind="revenue",
                              evoi_kpi_units=float(row["evoi"]), max_draws=120,
                              random_seed=42)
print(f"expected KPI change in the window:  {oc.expected_kpi_delta:+,.1f} {KPI}-units")
print(f"spend change (signed):              ${oc.spend_delta:+,.0f}  "
      f"(a holdout SAVES money — the sign is computed, never assumed)")
print(f"net profit impact (median):         ${oc.net_profit_impact_median:+,.0f}")
print(f"P[net loss]:                        {oc.prob_net_loss:.0%}")
print(f"opportunity cost, $ (median):       ${oc.opportunity_cost_dollar_median:,.0f}")
print(f"learning-to-cost ratio:             {oc.learning_to_cost_ratio:.2f}")

comps = [("margin on KPI change", MARGIN * oc.kpi_delta_median),
         ("spend change (signed, flipped)", -oc.spend_delta),
         ("net impact", oc.net_profit_impact_median)]
fig = go.Figure(go.Bar(x=[c[0] for c in comps], y=[c[1] for c in comps],
                       marker_color=[GOOD if v >= 0 else BAD for _, v in comps],
                       text=[f"${v:+,.0f}" for _, v in comps], textposition="outside",
                       showlegend=False))
fig.add_hline(y=0, line=dict(color=INK, width=1))
style(fig, 360, f"The {TOP} holdout priced from the model itself — saved spend "
                f"offsets forgone margin")
fig.update_layout(yaxis_title="$ during the test window")
fig.show()
"""

PARETO = r"""
from mmm_framework.planning import suggest_experiment

t0 = time.time()
out = suggest_experiment(
    model, str(GEO_CSV), KPI, TOP,
    margin=MARGIN,
    duration_min=6, duration_max=14,
    intensity_min=-100, intensity_max=100,
    max_draws=60, random_seed=42,
)
print(f"design sweep in {time.time()-t0:.0f}s — {len(out['candidates'])} candidate "
      f"designs, net-value axis engaged: {out['net_value_axis']}")

cands = pd.DataFrame(out["candidates"])
front = cands[cands["on_pareto"]]
print("\nthe Pareto front (non-dominated designs), priced in net dollars:")
print(front[["mode", "footprint", "intensity_pct", "duration", "mde_roas",
             "power", "net_value", "powered", "is_recommended"]]
      .round(2).to_string(index=False))

rec = out["recommended"]
dom = cands[~cands["on_pareto"] & ~cands["is_recommended"]]
fr = cands[cands["on_pareto"] & ~cands["is_recommended"]]
fig = go.Figure()
fig.add_trace(go.Scatter(x=dom["mde_roas"], y=dom["net_value"], mode="markers",
    name="dominated", marker=dict(size=9, color=MUTED, opacity=0.45)))
fig.add_trace(go.Scatter(x=fr["mde_roas"], y=fr["net_value"], mode="markers",
    name="Pareto front", marker=dict(size=14, color=fr["duration"], colorscale="YlGnBu",
    reversescale=True, colorbar=dict(title="weeks", thickness=10, len=0.6),
    line=dict(width=2, color=INK))))
fig.add_trace(go.Scatter(x=[rec["mde_roas"]], y=[rec["net_value"]], mode="markers",
    name="recommended", marker=dict(size=20, symbol="star", color=GOLD,
    line=dict(width=1.5, color=INK))))
fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"),
              annotation_text="test pays for itself above this line")
style(fig, 430, f"The {TOP} design space priced in dollars — precise AND "
                f"net-positive wins", legend=dict(orientation="h", y=-0.22))
fig.update_layout(xaxis_title="MDE (ROAS) — lower is more precise",
                  yaxis_title="net value of testing ($)")
fig.show()
print(f"\nrecommended: {rec['mode']} / {rec['footprint']} / {rec['intensity_pct']:+.0f}% "
      f"/ {rec['duration']}w — net value ${rec['net_value']:,.0f}, "
      f"cool-down {out['cooldown']['cooldown_weeks']}w before a clean re-read")
"""

CL_DESIGN = r"""
import mmm_framework.continuous_learning as cl

world = cl.make_world(seed=0)                       # 4 channels + planted synergies
CENTER = np.array([0.8, 0.8, 0.8, 0.8])             # status-quo weekly spend / geo
dsg = cl.central_composite(CENTER, delta=0.6, probe_pairs=world.pairs)

K = len(world.channels)
labels = (["center"]
          + [f"{c} {s}" for c in world.channels for s in ("low", "high")]
          + [f"probe {world.channels[i]}+{world.channels[j]} {s}"
             for (i, j) in world.pairs for s in ("low", "high")]
          + [f"shutoff {c}" for c in world.channels])

fig = go.Figure(go.Heatmap(z=dsg, x=list(world.channels), y=labels,
                           colorscale="YlGnBu", colorbar=dict(title="spend / geo")))
fig.update_yaxes(autorange="reversed")
style(fig, 520, "The central-composite design varies every channel — and pairs, "
                "and shutoffs — on purpose")
fig.show()
print(f"{dsg.shape[0]} experiment cells x {K} channels; the designed variation is "
      f"what identifies the surface — no historical MMM required")
"""

CL_FIT = r"""
t0 = time.time()
data = cl.simulate_panel(world, CENTER, n_geo=80, t_pre=6, t_test=10,
                         delta=0.6, noise=0.5, seed=1)
post = cl.fit(data, channels=world.channels, pair_signs=cl.PAIR_SIGNS_EXAMPLE,
              num_warmup=400, num_samples=400, num_chains=2, seed=0)
print(f"designed panel (80 geos, 6 pre + 10 test weeks) fitted in {time.time()-t0:.0f}s")
rhat = post.diagnostics["max_rhat"]
print(f"max r-hat: {rhat if rhat is None else round(rhat, 3)}")

beta_hat = post.samples["beta"].mean(0)
from scipy.stats import spearmanr
rho = spearmanr(beta_hat, world.beta).correlation
print(f"\nmain effects recovered (posterior mean vs planted truth), "
      f"rank correlation {rho:.2f}:")
print(pd.DataFrame({"planted beta": world.beta, "recovered beta": beta_hat},
                   index=list(world.channels)).round(2).to_string())
gs = post.gamma_summary()
print("\nsign-informed synergies (planted: Chatter x Pulse cannibalize, "
      "Pulse x Orbit and Orbit x Vibe complement):")
for k, v in gs.items():
    print(f"  {k}: {v['mean']:+.2f}")
"""

CL_FUND = r"""
B, VALUE = float(CENTER.sum()), 5.0    # keep total budget; $5 per unit KPI
rec = cl.recommend_allocation(post, B=B, value=VALUE, q=200, mode="fixed")
mroas_mean, prob_above, _ = cl.marginal_roas(post, rec, VALUE, q=200)

CL_COLORS = ["#4464ad", "#c9962e", "#3d7a5c", "#b4552d"]
fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                    subplot_titles=("marginal ROAS at the recommended split",
                                    "the recommended allocation"))
fig.add_trace(go.Bar(x=list(world.channels), y=mroas_mean, marker_color=CL_COLORS,
                     text=[f"P(funded)={p:.0%}" for p in prob_above],
                     textposition="outside", showlegend=False), row=1, col=1)
fig.add_hline(y=1.0, line=dict(color=BAD, width=2, dash="dash"),
              annotation_text="funding line: $1 back per $1", row=1, col=1)
fig.add_trace(go.Bar(x=list(world.channels), y=rec, marker_color=CL_COLORS,
                     showlegend=False), row=1, col=2)
fig.update_yaxes(title="marginal ROAS", row=1, col=1)
fig.update_yaxes(title="spend / geo / week", row=1, col=2)
style(fig, 400, "The funding line: money flows to channels whose NEXT dollar "
                "still pays, learned from experiments alone")
fig.show()

funded = [c for c, p in zip(world.channels, prob_above) if p > 0.5]
print(f"funded channels (P > 50% that the marginal dollar pays): {funded}")
print(f"strongest planted channel: {world.channels[int(np.argmax(world.beta))]} — "
      f"funded with P={prob_above[int(np.argmax(world.beta))]:.0%}")
"""

PREFIT = r"""
from mmm_framework.agents.fitting import build_model
from mmm_framework.reporting.prefit import PrefitReadoutGenerator

unfitted = build_model(spec, str(GEO_CSV))    # configured, NEVER fitted
t0 = time.time()
gen = PrefitReadoutGenerator(unfitted, run_sbc=False, n_prior_samples=200,
                             random_seed=42)
prefit_path = Path(gen.save_report(str(ART / "showcase_06_prefit_readout.html")))
print(f"Model Design Readout in {time.time()-t0:.0f}s -> {prefit_path.name} "
      f"({prefit_path.stat().st_size/1e6:.1f} MB)")
f = gen.facts
print(f"\ngrounded in {len(f['priors'])} enumerated priors and "
      f"{len(f['assumptions'])} named model assumptions — sampled from the prior, "
      f"zero posterior access")
print("first three assumptions the readout makes the client sign off on:")
for a in f["assumptions"][:3]:
    print(f"  - {a['topic']}: {a['setting']}")
"""

INTERACTIVE = r"""
from mmm_framework.reporting.interactive import InteractiveReportGenerator

t0 = time.time()
igen = InteractiveReportGenerator(model, results, max_draws=60, curve_max_draws=30,
                                  include_counterfactual_spec=False)
ipath = ART / "showcase_06_interactive_report.html"
ipath.write_text(igen.generate_report())
print(f"interactive MMM Results Report in {time.time()-t0:.0f}s -> {ipath.name} "
      f"({ipath.stat().st_size/1e6:.1f} MB)")
print("the report embeds thinned per-draw posterior matrices; sliders recompute "
      "decompositions and intervals IN THE BROWSER — no server, no Python")
"""

DECK = r"""
from mmm_framework.reporting.deck import build_deck

t0 = time.time()
deck = build_deck(model, results, client="Showcase", kpi_name=KPI, currency="$",
                  margin=MARGIN)
print(f"deck computed in {time.time()-t0:.0f}s — {len(deck.slides)} slides, "
      f"every number and PNG chart deterministic (AI insights are an optional layer)")

inv = pd.DataFrame([{"kind": s.kind, "title": s.title, "summary": s.is_summary,
                     "chart": s.chart_png is not None} for s in deck.slides])
sizes = {"Model Design Readout (pre-fit)": prefit_path.stat().st_size / 1e6,
         "Interactive Results Report": ipath.stat().st_size / 1e6}
fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                    subplot_titles=("self-contained HTML artifacts (MB)",
                                    "Augur deck: slides by kind"),
                    specs=[[{"type": "bar"}, {"type": "bar"}]])
fig.add_trace(go.Bar(x=list(sizes.values()), y=list(sizes.keys()), orientation="h",
                     marker_color=[GOLD, GOOD], showlegend=False), row=1, col=1)
kc = inv.groupby("kind").size().sort_values()
fig.add_trace(go.Bar(x=kc.values, y=kc.index, orientation="h",
                     marker_color=PALETTE["TV"], showlegend=False), row=1, col=2)
fig.update_xaxes(title="MB", row=1, col=1)
fig.update_xaxes(title="slides", row=1, col=2)
style(fig, 380, "Three report surfaces, one fitted model — a readout for every "
                "audience")
fig.show()
print(inv[["kind", "title", "summary"]].to_string(index=False))
"""

JOBS = r"""
from mmm_framework.jobs import JobConfig, JobStatus, get_job_manager

manager = get_job_manager(ART / "showcase_06_jobs")
# A deliberately small national job: the demo's point is the lifecycle
# (submit -> poll -> completed, kernel free throughout), not the fit itself.
from mmm_framework.synth import dgp
job_panel = dgp.make_clean(seed=5, n_weeks=50).panel()
job = manager.submit_job(job_panel, JobConfig(
    name="showcase refit", description="async NUTS refit in a subprocess",
    n_chains=1, n_draws=100, n_tune=100, use_numpyro=True,
    trend_type="linear", yearly_order=2, random_seed=1,
))
print(f"submitted {job.id} -> status {job.status.value} (pid runs in a "
      f"separate process; the notebook kernel stays free)")

t0, seen = time.time(), set()
while True:
    j = manager.get_job(job.id)
    stage = j.progress.stage if j.progress else "?"
    if stage not in seen:
        seen.add(stage)
        print(f"  t+{time.time()-t0:5.0f}s  {j.status.value:<9} stage={stage}")
    if not j.is_active or time.time() - t0 > 1200:
        break
    time.sleep(5)

j = manager.get_job(job.id)
print(f"\nfinal: {j.status.value} in {j.duration_seconds:.0f}s")
print("jobs on disk:", [(x.config.name, x.status.value) for x in manager.list_jobs()])
assert j.status == JobStatus.COMPLETED
"""

SERIAL = r"""
import json
from mmm_framework.serialization import MMMSerializer

save_dir = ART / "showcase_06_model"
MMMSerializer.save(model, save_dir)
meta = json.loads((save_dir / "metadata.json").read_text())
print("files:", sorted(p.name for p in save_dir.iterdir()))
print(f"model_kind: {meta.get('model_kind')!r}   format: {meta.get('format_version')}   "
      f"inference: {meta.get('inference_method')}")

reloaded = MMMSerializer.load(save_dir, model.panel)   # core flavor: panel required
same = np.allclose(
    reloaded.predict(return_original_scale=True, random_seed=1).y_pred_mean,
    model.predict(return_original_scale=True, random_seed=1).y_pred_mean,
)
print("round-trip predictions identical:", same)
assert same
"""

SECURITY = r"""
from cryptography.fernet import Fernet
from mmm_framework.security import DatasetEncryptor, EncryptionError, scan_dataframe_for_pii

key = Fernet.generate_key().decode()
enc = DatasetEncryptor(key)
blob = GEO_CSV.read_bytes()
sealed = enc.encrypt(blob)
assert enc.decrypt(sealed) == blob and sealed[:8] != blob[:8]
print(f"dataset encrypted at rest: {len(blob):,} bytes -> {len(sealed):,} bytes, "
      f"round-trip exact")
print("legacy plaintext passes through unchanged:", enc.decrypt(blob) == blob)
try:
    enc.decrypt(sealed[:-4] + b"XXXX")
except EncryptionError as e:
    print(f"tampered ciphertext REFUSES: {e}")

leaky = pd.DataFrame({
    "geo": ["North", "South"],
    "contact": ["ops@example.com", "call 415-555-0134"],
    "note": ["ssn 123-45-6789 on file", "clean"],
})
for f in scan_dataframe_for_pii(leaky):
    print(f"PII found — column {f.location!r}: {f.kind} x{f.n_matches} "
          f"(sample {f.sample!r})")
"""

AUTH = r"""
from mmm_framework.auth.tokens import (ExpiredToken, decode_jwt, encode_jwt,
                                       make_claims)

SECRET = "showcase-demo-secret"
claims = make_claims(subject="user-42", org_id="acme", org_role="analyst",
                     email="analyst@acme.test", token_type="access",
                     ttl_seconds=3600, issuer="mmm-framework",
                     audience="mmm-app")
token = encode_jwt(claims, SECRET)
print(f"minted HS256 JWT ({len(token)} chars): {token[:40]}...")
back = decode_jwt(token, SECRET, audience="mmm-app", issuer="mmm-framework")
print(f"verified: sub={back['sub']} org={back['org']} role={back['role']} "
      f"tv={back['tv']}")

stale = encode_jwt(make_claims(subject="user-42", org_id="acme", org_role="analyst",
                               email="analyst@acme.test", token_type="access",
                               ttl_seconds=-1, issuer="mmm-framework",
                               audience="mmm-app"), SECRET)
try:
    decode_jwt(stale, SECRET)
except ExpiredToken as e:
    print(f"expired token REFUSES: {e}")
"""

PLATFORM = r"""
from mmm_framework.platform import sessions

print("sessions DB ->", sessions.resolve_db_path().name)
sessions.init_db()
sess = sessions.create_session(name="showcase operate demo", project_id="showcase")
print(f"session created: {sess['thread_id'][:8]}... ({sess['name']!r})")

roi_metrics = {c: {"mean": float(gdf.set_index("channel").loc[c, "roi_mean"])}
               for c in CHANNELS}
sessions.record_run_metrics(
    "showcase-run-1",
    {"schema_version": 1, "rhat_max": float(results.diagnostics["rhat_max"]),
     "divergences": int(results.diagnostics["divergences"]),
     "roi": roi_metrics, "n_draws": 150, "backend": "numpyro"},
    thread_id=sess["thread_id"], project_id="showcase",
)
runs = sessions.list_run_metrics(project_id="showcase")
print(f"\n{len(runs)} run(s) on record for project 'showcase':")
for r in runs:
    m = r["metrics"]
    print(f"  {r['run_id']}: rhat_max={m['rhat_max']:.3f} "
          f"divergences={m['divergences']} backend={m['backend']}")
    print("    ROI snapshot:", {k: round(v["mean"], 2) for k, v in m["roi"].items()})
"""

CELLS = [
    md(r"""
# Operate the loop — keep it honest in production

A fitted MMM that lives in a notebook is a study. A fitted MMM that survives in
production is a *system*: it has to decide which experiment to run next, price
that experiment in dollars, keep learning when there is no model at all, render
itself for three different audiences, fit asynchronously, persist, encrypt,
authenticate, and remember every run it ever made. This finale tours that
operational layer:

1. **experiment planning** on a fitted model — the EIG/EVOI priority grid,
   information decay and the re-test trigger, design + model-anchored
   opportunity cost, and the Pareto experiment optimizer
2. **continuous learning** — the model-free geo response-surface bandit
   (headline: *no pre-fit MMM required*)
3. **reporting beyond `to_html`** — the pre-fit Model Design Readout, the
   interactive recompute-in-browser report, the Augur deck engine
4. **operations plumbing** — async fits (`jobs.py`), serialization flavors,
   encryption + PII scanning (`security/`), in-process JWTs (`auth/`), and the
   platform sessions store (`platform/`)
5. **the agent and the app layer** — how the LangGraph oracle, the FastAPI
   server and the React frontend wrap all of it

As throughout this series: every number is computed in this notebook, fits are
deliberately small (2 chains x 150 draws on an 8-geo synthetic world), and
refusals are shown firing where they are part of the design.
"""),
    code(SETUP),
    md(r"""
## 1 · One fit, through the same door the agent uses

`mmm_framework.agents` is mostly the LangGraph oracle (section 8), but its
*service modules* — `fitting`, `workspace`, `tables`, `model_ops` — are plain
Python with no langchain dependency, importable straight from a notebook or a
kernel. `build_and_fit(spec, csv)` is the exact spec-to-fitted-model pipeline
every agent fit goes through: a declarative dict in, a fitted `BayesianMMM`
out. We use it here because the whole planning layer needs a fitted model to
anchor on — and because a session exported from the app replays as exactly
this call.
"""),
    code(FIT),
    md(r"""
## 2 · Which channel deserves a test? The EIG/EVOI priority grid

Wide posterior does not equal worth testing. `compute_experiment_priorities`
separates two questions per channel: **EIG** — how much would an experiment of
achievable precision *teach* (nats of information) — and **EVOI** — how much
that learning is *worth to the budget decision* (simulate test outcomes,
reweight the posterior, re-optimize the budget, difference the value). A
channel can score huge EIG and near-zero EVOI: you would learn a lot, and the
budget would not move a dollar. The quadrants name the verdicts: `test_now`,
`learn_cheaply`, `monitor`, `deprioritize`.
"""),
    code(PRIORITY),
    md(r"""
### Evidence has a half-life

An experiment you ran a year ago is not evidence you have today — markets
drift, creative rotates, competitors move. The priority engine models this as
information decay: posterior variance doubles every channel-class half-life
(brand search decays fast, linear TV slowly), and once the EIG of a *fresh*
test crosses an operational threshold, the channel is flagged `retest_due`.
This is the difference between a measurement *program* and a one-off study.
"""),
    code(DECAY),
    md(r"""
## 3 · Design the test, then price it from the model

`design_experiment` turns a channel + a geo history into a runnable plan:
markets matched into pairs on pre-period behaviour, treatment randomized
within pair, and a power curve giving the minimum detectable effect at every
duration. The MDE is the design's blind spot — a true effect smaller than it
will read "inconclusive" no matter what happens.
"""),
    code(DESIGN),
    md(r"""
Running the test means deviating from business-as-usual, and the fitted model
can price that deviation before you commit: `compute_opportunity_cost` pushes
the design's spend perturbation through the posterior and reports the KPI
delta, the **signed** spend change, and the net profit impact at a stated
margin. The sign convention is load-bearing — a go-dark holdout *saves* spend,
so its net cost can be small or even negative. The framework computes the sign
from the perturbed spend matrix itself, never from the design's magnitude, so
it cannot invert.
"""),
    code(OPPCOST),
    md(r"""
## 4 · The Pareto experiment optimizer — the design space priced in dollars

Within one channel there is still a design space: holdout vs scaling, how many
geos, how hard, how long. `suggest_experiment` sweeps a bounded grid and keeps
the non-dominated set on four lower-is-better axes — MDE, power shortfall,
cost, duration. With a margin supplied, the cost axis upgrades to the **net
value of testing**: a calibrated Gaussian EVOI surrogate (anchored on two
exact preposterior Monte-Carlo runs at the grid's extremes) prices every
candidate, so the front reads the way a CFO wants it read.
"""),
    code(PARETO),
    md(r"""
### Off-panel calibration, in one paragraph

The loop closes outside this notebook: when the recommended test finishes, its
readout enters the *next* fit as a calibration likelihood on the matching
estimand (`calibration/`), and `validation/` can score the model against the
experiment even when the test ran on markets or windows **outside the fitted
panel** — off-panel calibration maps the experiment's footprint onto the
model's estimand rather than demanding the panel contain it. The full
T0-to-T5 walk is the `lifecycle_` series; the experiment-methods registry
(geo estimators, ghost ads, switchbacks, the A/A-calibrated leaderboard) has
its own deep dive in `demos/experiment_planning_playbook.ipynb`.
"""),
    md(r"""
## 5 · No model yet? Learn anyway — the continuous-learning bandit

Everything above leaned on a fitted MMM. `continuous_learning/` deliberately
does not: it is a **model-free geo response-surface bandit** (NumPyro/JAX)
that learns channel response *directly from designed experiments*. The
headline is the prerequisite it removes — you can start measuring on week one
of an engagement, before any observational history is worth fitting. The
designed cross-sectional variation is what identifies the surface, and the
central-composite design below is that variation made visible: a center cell
at status quo, low/high axial cells per channel, paired probe cells for
planted synergies, and full shutoffs.
"""),
    code(CL_DESIGN),
    md(r"""
The recovery gate (the same one `tests/test_continuous_learning.py` enforces):
simulate the designed panel on a world with planted effects and synergies, fit
the surface, and check the posterior gets the ordering and the synergy signs
back.
"""),
    code(CL_FIT),
    md(r"""
The decision output is not a point estimate but a **funding line**: at the
recommended allocation, which channel's *next* dollar still returns more than
a dollar? The loop's other pieces — Thompson sampling across waves, the ENBS
stopping rule ("does another wave of testing still pay?"), knowledge-gradient
acquisition — live in `demos/continuous_learning.ipynb`.
"""),
    code(CL_FUND),
    md(r"""
## 6 · Reporting beyond `to_html` — a surface for every audience

Notebook 00 showed the classic single-file HTML report. Production needs more
than one register:

- the **Model Design Readout** (`reporting/prefit.py`) renders *before any
  fitting* — priors enumerated from the actual graph, named assumptions,
  prior-predictive implications — so the client signs off on the model design
  while changing it is still cheap
- the **interactive MMM Results Report** (`reporting/interactive/`) embeds
  thinned posterior draws and recomputes decompositions in the browser as the
  reader drags sliders — a self-contained file, no server behind it
- the **Augur deck engine** (`reporting/deck/`) computes every slide's
  numbers, tables and PNG charts deterministically; `build_pptx` then fills a
  client's own PowerPoint template (we stop at the `Deck` object here — the
  pptx step just needs a template file)
"""),
    code(PREFIT),
    code(INTERACTIVE),
    code(DECK),
    md(r"""
## 7 · Operations plumbing

### Async fits — `jobs.py`

A production fit should never block a request thread or a notebook kernel.
`JobManager` runs each fit in a separate OS process, persists status /
progress / result to disk (jobs survive an app restart), and exposes
submit / poll / cancel. This is the machinery behind the app's fit queue —
no Redis, no external worker.
"""),
    code(JOBS),
    md(r"""
### Serialization flavors

`MMMSerializer` writes a self-describing directory and stamps a
`model_flavor` into `metadata.json`. The **core** flavor (plain `BayesianMMM`)
stores the trace + configs and needs the panel handed back at `load()`; the
**extended** flavor (Nested / Multivariate / Combined / StructuralNested
models) carries its arrays inside the pickle and loads panel-free. The loader
dispatches on the stamp, and a format-version gate refuses saves from an
incompatible major version instead of deserializing garbage.
"""),
    code(SERIAL),
    md(r"""
### Security — encryption at rest and PII screening

`security/` is two dependency-light, opt-in blocks for the hosted posture:
`DatasetEncryptor` (Fernet, with comma-separated key rotation — first key
encrypts, all keys decrypt) and a PII scanner that flags emails, phone
numbers, SSNs and Luhn-valid card numbers in uploaded data *before* they
reach storage. Note the two designed behaviors below: legacy plaintext passes
through decrypt unchanged, and tampered ciphertext refuses loudly.
"""),
    code(SECURITY),
    md(r"""
### Auth — a stdlib JWT core

`auth/` implements org/user auth with **no third-party JWT dependency**:
HS256 signing via `hmac` + `hashlib`, standard claims, and a `tv` (token
version) claim that gives stateless "sign out everywhere" — bump the stored
version and every older token dies on next use. The same `TokenVerifier`
protocol accepts an external OIDC issuer in enterprise deployments.
"""),
    code(AUTH),
    md(r"""
### The platform sessions store — institutional memory

`platform/` is the web-free service layer under the FastAPI server: one
SQLite file (relocatable via `MMM_SESSIONS_DB`) holds sessions, artifacts,
recorded assumptions, locked analysis plans, the experiment lifecycle
registry, and **run metrics** — so "what did the model say last quarter, and
what changed?" is a query, not an archaeology project. Around it sit the
pacing service (planned vs delivered spend, alert sweeps), the project
scorecard, run comparison (`runs.compare_runs` diffs specs
assumption-by-assumption), history, backfill and backup. Here is the store
doing its two core moves in-process: record a run, list the record.
"""),
    code(PLATFORM),
    md(r"""
## 8 · The agent — a LangGraph oracle over everything above

`agents/` wraps the whole framework in a conversational analyst. Architecture,
in one pass:

- **the graph** (`agents/graph.py`) — a LangGraph state machine: an agent node
  with layered context management (summarize-then-trim, so long sessions do
  not blow the context window), a tool-execution node, and a pause node driven
  by a **per-turn workflow guard** (`workflow_guard.py`) that stops the model
  from auto-running all nine workflow steps off a single "looks good"
- **tools** — thin wrappers over the same functions this notebook called
  directly: fit, validate, plan, design, report. The agent has no private
  math; anything it claims, you can reproduce in a notebook
- **kernels** (`agents/kernels.py`) — `execute_python` runs in a session
  kernel (in-process, subprocess, or container for the hosted posture), with
  results persisted as session artifacts
- **workspace** (`agents/workspace.py`) — per-thread files under
  `$MMM_AGENT_WORKSPACE/threads/<id>/`, plus a project knowledge base with RAG
- **session export** — a session replays as a runnable Python script whose
  spine is the `build_and_fit(spec, dataset_path)` call from section 1

The LLM behind it is configuration (`config/model_config.yaml` — Vertex,
Anthropic, OpenAI, or a local LM Studio model), not code. And the lean-core
contract holds throughout: `pip install mmm-framework` pulls **none** of this
stack; `tests/test_lean_imports.py` gates that promise, and
`agents/__init__.py` is lazy so the service modules import without the
`[agents]` extra installed.

### The app layer

Two more packages complete the picture, covered here in prose because they
are services, not notebook APIs:

- **`mmm-framework-server`** (`server/src/mmm_framework_server/main.py`) — the
  FastAPI app: REST endpoints for fits (via `jobs.py`), planning, the
  experiment lifecycle, reports, auth, and the agent's streaming chat. Fits
  run in-kernel; there is no external queue to operate.
- **`frontend/`** — the React/TypeScript app (Vite, Tailwind 4, Zustand).
  Its pages mirror the measurement loop: Program, Experiments, Performance,
  Agent. `uv run uvicorn mmm_framework_server.main:app` plus `npm run dev`
  brings the whole thing up locally.
"""),
    md(r"""
## The series, closed

Seven notebooks, one loop:

> **fit** a model you can defend → **validate** it against worlds with sealed
> answer keys → **plan** budget and experiments in dollars, with the value of
> information priced explicitly → **commit** a plan of record → **measure**
> what happened — then feed every experiment back into the next fit, and let
> the platform remember all of it.

This notebook was the "and then it has to run every week" chapter: priorities
that decay, tests priced before they run, a bandit that needs no model, four
report surfaces, and the plumbing — jobs, serialization, security, auth,
sessions — that turns a model into a system.

**The showcase series** (`nbs/showcase/`): `00 the loop` · `01 data` ·
`02 models` · `03 families` · `04 trust` · `05 planning` · `06 operate`
(this one).

**Where to go deeper**

- `technical-docs/experiment-economics.md` — opportunity cost, net test
  value, the A/A / A/B methodology simulator
- `technical-docs/continuous-learning.md` — the bandit's math and its
  feasibility gates
- `technical-docs/prefit-model-design-readout.md`,
  `technical-docs/agent-session-kernels.md`, `technical-docs/packaging.md` —
  the readout, the kernels, the lean-core split
- `technical-docs/engineering-notes.md` — the subsystem index for everything
  touched here (*Measurement loop*, *Experiment optimizer*, *Variance to
  plan*, *Hosted multi-user profile*, ...)
- sibling series under `nbs/`: `lifecycle_00..06` (the T0-T5 loop in full),
  `causal/00..10` (why experiments and MMMs need each other),
  `demos/experiment_planning_playbook.ipynb` and
  `demos/continuous_learning.ipynb` (the two planning deep dives),
  `workshop_`, `math_`, `stress_` (start, formalize, break)
- `docs/` — the documentation site; `docs/getting-started.html` to install,
  the Cookbook and API reference to build
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = "showcase/showcase_06_operate_the_loop.ipynb"
    with open(out, "w") as fh:
        nbformat.write(nb, fh)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
