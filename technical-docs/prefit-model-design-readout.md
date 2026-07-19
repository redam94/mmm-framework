# Pre-fit Model Design Readout

**Status: shipped (2026-07-02).** The pre-fit sibling of the Augur "Media
Performance Readout": a pre-registration HTML document generated **before** the
model is fitted, so there is a durable, auditable record of the model's
assumptions, priors, prior checks and specification changes before the final
fit ever saw the data's verdict.

## Why

The fitted report (Augur readout, `reporting/generator.py` + `augur_sections.py`)
documents what the model *found*. Nothing documented what the model *assumed* —
which priors were chosen, what they imply, whether the inference machinery is
calibrated, and how the specification evolved during design. Pre-registering
those choices separates them from findings and makes the final model auditable.

## Shape of the document

Same editorial Augur shell (masthead, numbered sticky contents nav with
scrollspy, cream/ink palette, evidence chips) via `augur_theme.py`. Eleven
sections:

| id | Section | Content |
|----|---------|---------|
| `purpose` | The model design, on record | standfirst, KPI strip (channels / window / free params), verdict chips (prior coverage, negativity, SBC), "why pre-register" callout |
| `spec` | What is being modeled | KPI, channels, controls, panel shape, likelihood/link |
| `assumptions` | What we are assuming | `model_assumptions()` rows — likelihood, adstock family per channel group, saturation family, effect sign, trend, seasonality, pooling, controls (+causal roles), inference plan |
| `priors` | Every prior | grouped table (family + empirical prior mean/sd/90% range, experiment-calibrated chip) + prior-density small multiples |
| `response` | Implied response shapes | prior-implied saturation bands + adstock-decay bands per channel (dropped if no curve samples) |
| `components` | Structural priors in time | prior draws of the graph's component deterministics (`trend_component`, `seasonality_component`, `controls_total`, `media_total`) on the **original KPI scale**, per period: 90% band + median + individual prior traces per component, zero reference line (dropped when the graph lacks the deterministics) |
| `prior-predictive` | Could this model have produced our data? | fan chart (observed vs 50/90% prior bands on the period axis **plus individual prior-draw spaghetti traces**), KPI strip with coverage / **original-scale distance** (mean \|observed − prior median\| in prior-sd units, `scale_z_abs_mean`) / negativity, replicate mean/sd histograms vs observed |
| `prior-estimands` | What the priors already say about returns | the PRIOR distribution of the flagship estimands: per-channel contribution ROI (prior `channel_contributions` summed × `y_std` ÷ the measurement-aware divisor from `resolve_channel_divisor` — same semantics as the fitted `contribution_roi`, efficiency-basis channels keep their 0 reference), table (prior mean / 90% range / P(above break-even)) + per-channel density charts with a break-even line + blended prior return + prior marketing share of the observed KPI (dropped when `channel_contributions` is absent) |
| `sbc` | Can the machinery recover known answers? | verdict table (χ² p, miscalibration, bias/dispersion z) + rank-histogram (and ECDF-diff when ranks present) grid; renders a "not yet run" note when absent |
| `revisions` | Change record | numbered table date/author/change/rationale |
| `signoff` | Next steps | measurement-loop chips (Design → Fit → …) + who-does-what |

## SBC runs by default

`PrefitReadoutGenerator(model, run_sbc=True)` (the default): when no `sbc`
result is supplied and a model is given, a smoke-level SBC runs during
generation (`DEFAULT_SBC_KWARGS = n_sims=20, L=50`, override via
`sbc_kwargs`), serialized **with raw ranks** so the ECDF-difference panel
renders. Failures degrade to the "not yet run" section (never block the
readout). `run_sbc=False` skips it; a `facts=`-only construction never runs it
(no model). The agent tool mirrors this (`run_sbc=True, sbc_sims=20`), reuses
the session's stored `dashboard_data["sbc"]` first, and **persists a fresh run
back to the dashboard** so later turns / the validation UI reuse it.

## Code map

- `src/mmm_framework/reporting/helpers/prefit.py` — the facts layer (read-only
  on the model, JSON-safe out): `enumerate_model_priors` (free RVs → grouped
  rows; family via the pytensor op's `_print_name`, empirical stats from prior
  samples; `roi_prior`-calibrated channels flagged), `model_assumptions`,
  `prior_predictive_facts` (period-axis aggregation via `time_idx` bincount,
  original scale via `y_mean`/`y_std`, 90%-band coverage, `scale_z_abs_mean`
  original-scale distance, negativity share, replicate mean/sd, spaghetti
  `traces`), `prior_component_facts` (component deterministics → period-axis
  bands + traces × `y_std`), `prior_estimand_facts` (prior contribution-ROI
  draws per channel via `resolve_channel_divisor`, blended + marketing share),
  `prior_response_curves` (saturation families
  logistic/hill/michaelis_menten/tanh; adstock geometric/delayed/weibull),
  `sample_prior` (one shared prior draw through `arviz_compat`).
- `src/mmm_framework/reporting/charts/prior.py` — the chart kit:
  `create_prior_predictive_fan`, `create_prior_stat_distribution`,
  `create_prior_density_chart` (KDE, histogram fallback),
  `create_prior_saturation_band`, `create_prior_adstock_band`,
  `create_sbc_rank_histogram` (renders from `bin_counts`, so a stored
  `to_dashboard()` result works without raw ranks; band via
  `diagnostics.sbc.rank_hist_band`), `create_sbc_ecdf_diff` (needs
  `int_ranks`; returns `""` otherwise). All `create_plotly_div`-style, themed
  by `ReportConfig.color_scheme`.
- `src/mmm_framework/reporting/prefit.py` — `prefit_facts(model, sbc=…,
  revisions=…)`, `build_prefit_insights(facts, llm=None)` (templated fallback
  for every slot in `PREFIT_INSIGHT_SLOTS`; optional single-call LLM enrichment
  parsing labelled STANDFIRST/ASSUMPTIONS/PRIORS/PRIOR_PREDICTIVE/SBC/NEXT_STEPS
  parts, mirroring `insights.py`'s grounding discipline), and
  `PrefitReadoutGenerator` (accepts a model OR precomputed `facts`; `sbc` may be
  an `SBCResult` or its dashboard dict).

## Templated vs AI-generated

Both readouts now have an explicit knob:

- Pre-fit: `PrefitReadoutGenerator(model, llm=None)` → fully templated;
  `llm=<chat model>` → AI-enriched glosses (best-effort, silent fallback).
- Fitted: `generate_client_report(template="augur", ai_insights=False)` skips
  the LLM entirely for the deterministic templated version (previously the
  augur path always tried the LLM).

## Wiring

- Agent tool `generate_model_design_readout` (`agents/tools.py`, registered in
  `TOOLS` under Reporting): builds an **unfitted** model from the active
  `spec` + `dataset_path` (mirrors `prior_predictive_check`'s pre-fit
  philosophy — reflects the priors the *next* fit would use; fitted-model graph
  only as fallback), reuses the latest SBC dashboard from
  `dashboard_data["sbc"]` (does NOT re-run SBC — that stays behind
  `run_calibration_check` / the validation job), derives the change record from
  the session's versioned assumption log
  (`sessions.list_assumptions(include_history=True)`, tombstones dropped,
  `change_note` per version), writes `agent_prefit_readout.html` via
  `workspace.report_path`, and sets `dashboard_data["prefit_report_path"]`.
- REST: `GET /prefit-report` (+ `/prefit-report/download`) in
  `src/mmm_framework/api/main.py`, same `_serve_report` pattern (thread-scoped via
  `?thread_id=`).

## Gotchas

- `prior_predictive_facts` unstandardizes with `y_mean`/`y_std` — correct for
  Gaussian-family models (count/bounded families keep natural scale where those
  attrs are 0/1, so the bridge is identity).
- The SBC ECDF-diff chart silently degrades: the model op serializes params via
  `to_dashboard()` *without* raw ranks, so stored results render histogram-only.
  Serialize with `to_dashboard(max_ranks=…)` upstream to get the ECDF panel.
- The response-curve section reads per-channel `sat_*_/adstock_*_<ch>` RVs by
  the core model's naming convention; bespoke garden models without those names
  just drop the section (nav renumbers automatically).

## Tests

`tests/reporting/test_prefit_readout.py` — chart-kit units, insight-slot
completeness + LLM-enrichment parsing (fake llm), canned-facts HTML assembly
(section ids, XSS-escaping), and a fast real-unfitted-model end-to-end
(`prefit_facts` → full HTML, no MCMC).
