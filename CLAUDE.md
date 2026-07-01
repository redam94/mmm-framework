# CLAUDE.md - MMM Framework Developer Guide

## Project Overview

**MMM Framework** is a production-ready Bayesian Marketing Mix Modeling framework extending PyMC-Marketing with advanced capabilities for modeling marketing effectiveness. Emphasizes methodological rigor, genuine uncertainty quantification, and pre-specified analyses to reduce researcher degrees of freedom.

**Version**: 0.1.0
**Author**: Matthew Reda
**Python**: 3.12+

## Quick Commands

```bash
# Install dependencies
uv sync --group dev --group app

# Run tests
make tests              # All tests with coverage (parallel)
make fast_tests         # Skip slow tests (parallel)
make slow_tests         # Slow tests only (parallel)

# Format code
make format             # Black formatting (src, tests, examples, api, app)

# Build API reference (Sphinx)
uv run sphinx-build -b html docs/api/source docs/api/build/html

# Static documentation site (hand-authored HTML in docs/*.html)
# Edit pages directly; preview locally with `python3 -m http.server` from docs/.
# Navigation/footer are injected by docs/shared/components.js — update NAV_LINKS there.

# Start the application

## Option 1: React UI (modern, supported) — talks to the agent API
# The agent API runs fits in-kernel, so no Redis/ARQ worker is required.
uv run uvicorn mmm_framework.api.main:app --host 0.0.0.0 --port 8000 --reload  # Terminal 1: Agent API
cd frontend && npm run dev                # Terminal 2: React UI (Vite, port 5173, proxies /api → :8000)

## Option 2: Streamlit UI (legacy, deprecated) — targets the separate legacy REST API (api/main.py)
redis-server                              # Terminal 1: Start Redis
cd api && uvicorn main:app --reload       # Terminal 2: legacy REST API (port 8000)
cd api && arq worker.WorkerSettings       # Terminal 3: ARQ worker
cd app && streamlit run Home.py           # Terminal 4: Streamlit UI (port 8501)

# Run example
uv run python examples/ex_model_workflow.py
```

## Configure the agent LLM (Vertex AI / ADC / API keys)

The LangGraph agent's LLM is chosen by a model configuration file, not hard-coded.
See `docs/model-configuration.md` and `src/mmm_framework/agents/llm.py`.

```bash
cp config/model_config.example.yaml config/model_config.yaml   # then edit
# Vertex AI on a GCP VM uses Application Default Credentials (no API key):
#   provider: vertex_anthropic | vertex_gemini ; set project + location
# Direct providers use API keys (anthropic / openai / google_genai).
# Override any field via env: MMM_LLM_PROVIDER, MMM_LLM_MODEL, MMM_LLM_PROJECT, ...
uv run python examples/ex_vertex_agent.py      # Vertex/ADC smoke test
```

## Directory Structure

```
mmm-framework/
├── src/mmm_framework/          # Core modeling library
│   ├── config/                 # Configuration enums and Pydantic dataclasses
│   ├── data_loader.py          # MFF parsing, validation, loading
│   ├── jobs.py                 # Async job management (ARQ)
│   ├── analysis.py             # Counterfactual & marginal analysis
│   ├── data_preparation.py     # Data scaling and transformation prep
│   ├── serialization.py        # Model save/load (MMMSerializer)
│   ├── builders/               # Fluent configuration builders
│   │   ├── base.py             # Base builder classes
│   │   ├── mff.py              # MFF config builders
│   │   ├── model.py            # Model config builders
│   │   ├── prior.py            # Prior config builders
│   │   └── variable.py         # Variable config builders
│   ├── model/                  # Core BayesianMMM implementation
│   │   ├── base.py             # Main BayesianMMM class
│   │   ├── results.py          # MMMResults, PredictionResults
│   │   ├── trend_config.py     # Trend configuration
│   │   └── components/         # Model components (trend, etc.)
│   ├── transforms/             # Transform functions
│   │   ├── adstock.py          # Adstock transformations
│   │   ├── saturation.py       # Saturation curves
│   │   ├── seasonality.py      # Seasonality components
│   │   └── trend.py            # Trend components
│   ├── utils/                  # Utility functions
│   │   ├── standardization.py  # Data standardization
│   │   └── statistics.py       # Statistical helpers
│   ├── synth/                  # Synthetic DGP worlds with causal ground truth
│   │   ├── dgp.py              # National scenarios (realistic, clean, violations)
│   │   ├── dgp_geo.py          # Geo / geo x product panel scenarios
│   │   └── mff.py              # Scenario -> MFF dataset + JSON answer key
│   ├── dag_model_builder/      # DAG-based model configuration
│   │   ├── builder.py          # Main DAG builder
│   │   ├── dag_spec.py         # DAG specification
│   │   ├── node_configs.py     # Node configurations
│   │   ├── validation.py       # DAG validation
│   │   ├── config_translator.py
│   │   ├── frontend_adapter.py
│   │   └── model_type_resolver.py
│   ├── mmm_extensions/         # Advanced models
│   │   ├── config.py           # Extension configurations
│   │   ├── results.py          # Extension results
│   │   ├── builders.py         # Extension builders
│   │   ├── models/             # Model implementations
│   │   │   ├── base.py
│   │   │   ├── nested.py       # NestedMMM (mediation)
│   │   │   ├── multivariate.py # MultivariateMMM (multi-outcome)
│   │   │   └── combined.py     # CombinedMMM
│   │   └── components/         # Extension components
│   │       ├── cross_effects.py
│   │       ├── observation.py
│   │       ├── priors.py
│   │       ├── transforms.py
│   │       └── variable_selection.py
│   └── reporting/              # HTML report generation
│       ├── generator.py        # MMMReportGenerator
│       ├── config.py           # Report configuration
│       ├── sections.py         # Report sections
│       ├── design_tokens.py    # Design tokens/themes
│       ├── charts/             # Plotly chart functions
│       │   ├── base.py
│       │   ├── decomposition.py
│       │   ├── diagnostic.py
│       │   ├── extended.py
│       │   ├── fit.py
│       │   ├── geo.py
│       │   └── roi.py
│       ├── extractors/         # Data extraction for reports
│       │   ├── base.py
│       │   ├── bayesian.py
│       │   ├── bundle.py
│       │   ├── extended.py
│       │   ├── mixins.py
│       │   └── pymc_marketing.py
│       └── helpers/            # Report helper utilities
│           ├── adstock.py
│           ├── decomposition.py
│           ├── mediated.py
│           ├── prior_posterior.py
│           ├── protocols.py
│           ├── results.py
│           ├── roi.py
│           ├── saturation.py
│           ├── summary.py
│           └── utils.py
├── api/                        # FastAPI backend
│   ├── main.py                 # App factory
│   ├── worker.py               # ARQ worker for async jobs
│   ├── config.py               # API configuration
│   ├── auth.py                 # Authentication
│   ├── middleware.py           # Middleware utilities
│   ├── rate_limiter.py         # Rate limiting
│   ├── redis_service.py        # Redis client
│   ├── schemas.py              # Pydantic schemas
│   ├── storage.py              # Storage utilities
│   └── routes/                 # API endpoints
│       ├── configs.py          # Configuration endpoints
│       ├── data.py             # Data endpoints
│       ├── models.py           # Model endpoints
│       └── extended_models.py  # Extended model endpoints
├── app/                        # Streamlit frontend (legacy)
│   ├── Home.py                 # Main entry point
│   ├── api_client.py           # API client
│   ├── components/             # Reusable components
│   └── pages/                  # UI pages (1-6)
├── frontend/                   # React/TypeScript frontend (modern)
│   ├── src/
│   │   ├── api/                # API hooks and services
│   │   ├── components/         # React components
│   │   │   ├── common/         # Shared (ProjectSwitcher, ModelSwitcher…)
│   │   │   ├── layout/         # AppShell, Header, Sidebar
│   │   │   └── ui/             # Token-native kit (Card, StatHero, Drawer…)
│   │   ├── pages/              # IA mirrors the measurement loop
│   │   │   ├── Program/        # Home: T₀–T₅ stage, KPIs, coverage map
│   │   │   ├── Experiments/    # EIG/EVOI matrix, lifecycle board, drawer
│   │   │   ├── Performance/    # Trajectories, agreement log, runs timeline
│   │   │   ├── Agent/          # Chat workspace (/workspace)
│   │   │   └── Login/
│   │   ├── theme/              # Design tokens: tokens.css (@theme — Tailwind 4,
│   │   │                       #   NO tailwind.config), colors.ts, plotlyTheme.ts
│   │   └── stores/             # Zustand state stores
│   ├── vite.config.ts
│   └── tailwind.config.js      # INERT under Tailwind 4 (tokens live in CSS)
├── examples/                   # Working usage examples
├── tests/                      # Test suite
├── docs/                       # Sphinx documentation
├── technical-docs/             # Mathematical specifications
└── nbs/                        # Jupyter notebooks
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `model/base.py` | BayesianMMM class - saturation, adstock, hierarchical modeling |
| `model/results.py` | MMMResults, PredictionResults classes |
| `config/` | Pydantic configs: PriorConfig, AdstockConfig, MediaChannelConfig |
| `builders/` | Fluent builders: ModelConfigBuilder, MediaChannelConfigBuilder |
| `data_loader.py` | MFFLoader for Master Flat File format data |
| `analysis.py` | Counterfactual analysis, MarginalAnalysisResult |
| `data_preparation.py` | Data scaling with ScalingParameters |
| `serialization.py` | MMMSerializer for model persistence |
| `transforms/` | Adstock, saturation, seasonality, trend transforms |
| `utils/` | Standardization and statistics utilities |
| `synth/` | Synthetic worlds with known causal truth (moved from `tests/synth`, which now shims to it); `generate_mff()` powers the agent's `generate_synthetic_data` (default scenario `realistic`; writes `synthetic_truth.json` answer key) |
| `dag_model_builder/` | DAG-based model configuration from frontend |
| `mmm_extensions/` | NestedMMM (mediation), MultivariateMMM (multi-outcome), CombinedMMM |
| `reporting/` | MMMReportGenerator, charts, extractors, helpers |

## Code Style

- **Formatter**: Black (via `make format`)
- **Linter**: Ruff
- **Type hints**: Required (py.typed markers present)
- **Docstrings**: Google-style
- **Naming**: PascalCase classes, snake_case functions, UPPER_SNAKE constants

## Architecture Patterns

- **Builder Pattern**: Fluent configuration (e.g., `ModelConfigBuilder().with_kpi("sales").build()`)
- **Factory Functions**: Pre-configured builders (e.g., `awareness_mediator()`)
- **Lazy Imports**: mmm_extensions delays PyMC import until needed
- **Pydantic Models**: Type-safe configuration validation
- **Subpackage Organization**: Related modules grouped into subpackages (builders/, model/, transforms/, etc.)
- **DAG-based Configuration**: Frontend-driven model specification via dag_model_builder/

## Core Technologies

- **PyMC 5.26+** - Bayesian inference
- **NumPyro 0.19+** - Fast NUTS sampling
- **NutPie 0.16+** - Alternative fast NUTS sampler
- **Numba 0.63+** - JIT compilation for performance
- **FastAPI 0.124+** - REST API
- **Streamlit 1.52+** - Legacy web UI
- **React + TypeScript** - Modern web UI (Vite, TailwindCSS, Zustand)
- **Redis 7.1+ / ARQ 0.25+** - Async job queue
- **Plotly 6.5+** - Interactive visualizations
- **Pydantic 2.12+** - Data validation
- **Sphinx** - Documentation

## Testing

```bash
# Run specific test file
uv run pytest tests/test_model.py -v

# Run with coverage report
uv run pytest tests/ --cov=mmm_framework --cov-report=html

# Skip slow tests (useful during development)
uv run pytest tests/ -m 'not slow'

# Run tests in parallel (default in Makefile)
uv run pytest tests/ -n logical
```

Test markers:
- `@pytest.mark.slow` - Long-running tests (model fitting)

Test organization:
- `tests/` - Core module tests
- `tests/reporting/` - Reporting module tests
- `tests/mmm_extensions/` - Extension module tests
- `tests/test_docs_snippets.py` - Docs code-snippet gate: verifies `docs/*.html` Python blocks only reference real APIs (see `technical-docs/doc-snippet-testing.md`)

## Common Development Tasks

### Add feature to BayesianMMM
1. Edit `src/mmm_framework/model/base.py`
2. Update results in `src/mmm_framework/model/results.py`
3. Update config in `src/mmm_framework/config/`
4. Add builder method in `src/mmm_framework/builders/model.py`
5. Write tests in `tests/test_model.py`

### Add new transform
1. Create transform function in `src/mmm_framework/transforms/`
2. Export in `src/mmm_framework/transforms/__init__.py`
3. Add tests in `tests/test_transforms.py`

### Add report section
1. Chart function in `reporting/charts/`
2. Extractor in `reporting/extractors/`
3. Section builder in `reporting/sections.py`
4. Helper in `reporting/helpers/`
5. Update `reporting/config.py`

### Add extended model
1. Config in `mmm_extensions/config.py`
2. Builder in `mmm_extensions/builders.py`
3. Components in `mmm_extensions/components/`
4. Model class in `mmm_extensions/models/`
5. Results in `mmm_extensions/results.py`

### Add API endpoint
1. Create route in `api/routes/`
2. Add schemas in `api/schemas.py`
3. Register in `api/main.py`
4. Add tests in `api/tests/`

### Add frontend page (React)
1. Create page component in `frontend/src/pages/`
2. Add API hooks in `frontend/src/api/hooks/`
3. Update store if needed in `frontend/src/stores/`
4. Add route in `frontend/src/App.tsx`

## API Usage Examples

```python
# Basic model fitting
from mmm_framework import BayesianMMM
from mmm_framework.builders import ModelConfigBuilder

config = ModelConfigBuilder().with_kpi("sales").build()
model = BayesianMMM(X_media, y, channel_names, config)
results = model.fit(draws=2000, tune=1000)

# Data loading
from mmm_framework import MFFLoader
loader = MFFLoader(config=mff_config)
data = loader.load("data.csv")

# Report generation
from mmm_framework.reporting import MMMReportGenerator, ReportConfig
generator = MMMReportGenerator()
html = generator.generate_report(results, config)

# Model serialization
from mmm_framework import MMMSerializer
serializer = MMMSerializer()
serializer.save(model, results, "model.pkl")
model, results = serializer.load("model.pkl")

# Analysis
from mmm_framework.analysis import MarginalAnalysisResult
analysis = MarginalAnalysisResult.from_model(model, results)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection fails | Run `redis-server` in separate terminal |
| Worker not processing | Start ARQ: `cd api && arq worker.WorkerSettings` |
| Model fitting slow | Use numpyro sampler: `model.fit(nuts_sampler="numpyro")` |
| Approximate / fast fit for model checking | `BayesianMMM.fit(method=...)` runs an *approximate* posterior in seconds to spot problems (bad priors, broken geometry, pathological saturation/adstock) before paying for NUTS. `method`: `"map"` (point estimate), `"advi"`/`"fullrank_advi"` (variational; `find_MAP`/`pm.fit` from core PyMC — no extra deps), or `"pathfinder"` (lazy-imports `pymc_extras`; **not a declared extra** — it pins pymc>=6 and would force-upgrade the core stack, so `pip install pymc-extras blackjax` manually and accept the upgrade). Returns a normal `MMMResults` with `approximate=True` and `diagnostics["fit_method"]`; the posterior is a drop-in for the NUTS trace (deterministics included, `predict`/reporting work) but R-hat/ESS are `None` and uncertainty is **not** calibrated — re-fit with NUTS before trusting intervals/decisions. Enum `FitMethod`; builder `.with_fit_method(...)`/`.map_fit()`/`.advi(full_rank=)`/`.pathfinder()` set `ModelConfig.fit_method` (default `nuts`). Agent: `fit_mmm_model(method=...)` or spec `inference.method`. Impl: `model/base.py::_fit_approx`; tests `tests/test_approx_fit.py`. |
| arviz/pymc version-drift shims | `utils/arviz_compat.py` centralizes the version-robust shims for arviz/pymc API drift — `sample_prior_predictive` (pymc `samples`→`draws` rename), `dataset_extremum` (reduce `az.rhat`/`az.ess` over `.data_vars`, not the removed `.to_array`), `group_names`/`has_group` (arviz `.groups` became a slash-prefixed *property* on the DataTree migration), `attach_prior` (DataTree has no `.extend`), `point_to_idata` (`az.from_dict`'s convention flipped AND the wrong form fails *silently* → validate vars landed). Use these instead of calling the raw arviz/pymc APIs. Callers: `model/base.py`, `diagnostics/{snapshot,learning}.py`, `reporting/helpers/prior_posterior.py`, `reporting/extractors/bayesian.py`, `mmm_extensions/models/base.py`, `api/routes/models.py`. |
| Tests hanging | Run `make fast_tests` to skip slow tests |
| Memory overflow | Reduce draws/chains: `fit(draws=1000, tune=500, chains=2)` |
| Import errors | Run `uv sync --group dev` to install dependencies |
| Frontend not starting | Run `cd frontend && npm install` first |
| Rate limiting errors | Check `api/rate_limiter.py` configuration |
| Serialization errors | Ensure cloudpickle version matches across environments |
| DAG validation fails | Check `dag_model_builder/validation.py` for requirements |
| Agent LLM auth / wrong provider | Check `config/model_config.yaml` (or `MMM_LLM_*` env); see `docs/model-configuration.md`. On GCP, Vertex uses ADC — grant the VM service account `roles/aiplatform.user` |
| Use a local model (LM Studio) | Set `provider: lmstudio`, `model: <id from LM Studio>`, `base_url: http://localhost:1234/v1` (or `MMM_LLM_PROVIDER=lmstudio`/`MMM_LLM_BASE_URL`). Start LM Studio's server and load a model; the login screen lists loaded models via `/lmstudio-models`. No API key needed. Tool-calling needs a tool-capable model. For the KB, load an embedding model too and set `MMM_EMBED_MODEL`. |
| Vertex "model not found" / 404 | Use the exact Model Garden id (may have `@version`) and a `location` region that serves it (Claude: e.g. `us-east5`) |
| Knowledge-base ingest shows "error" / "no embedding backend" | The chat LLM and the embedder are separate (Anthropic has no embeddings). With a `vertex_*` chat provider, KB uses Vertex `text-embedding-005` over the same ADC — run `gcloud auth application-default login` and ensure a GCP project is set. Override with `MMM_EMBED_PROVIDER` (`vertex`/`openai`/`google_genai`), `MMM_EMBED_MODEL`, `MMM_EMBED_LOCATION` (default `us-central1`). See `technical-docs/agent-knowledge-workspace.md` §5. |
| Agent output files / KB location | Per-session output lands in `$MMM_AGENT_WORKSPACE/threads/<thread_id>/` (default `./agent_workspace`); project KB sources in `$MMM_AGENT_WORKSPACE/projects/<project_id>/kb/`. The agent greps/reads these via `list_workspace_files`/`grep_workspace`. |
| Agent tables / formatted output | Tabular tool output streams as content-addressed refs in `dashboard_data.tables` (rows served once via `GET /tables/{id}`, thread-salted + size-capped via `MMM_TABLE_MAX_BYTES`, default 1 MiB; store in `agents/workspace.py`, builders in `agents/tables.py`). In `execute_python`, `show_table(df, title=...)` renders a sortable dashboard table — the prompt forbids printing full DataFrames. Model ops return a `tables` key across the kernel MIME boundary; EDA tools also fill `dashboard_data.eda` (issues + outlier actions) for the UI EDA tab, whose confirm buttons hit `POST /outliers/{thread_id}/apply` (state-only update). |
| Measurement loop / experiments | The product story is the adaptive cycle T₀ fit → T₁ EIG/EVOI priorities (`planning/eig.py`,`evoi.py`,`priority.py`) → T₂ pre-registered experiments (lifecycle registry in `src/mmm_framework/api/sessions.py`: draft→planned→running→completed→calibrated, `POST /experiments/{id}/transition`) → T₃ calibrated refit (`spec["experiments"]` → `add_experiment_calibration`; `fit_mmm_model` auto-marks calibrated) → T₄ allocate → T₅ re-evaluate (information decay triggers re-tests). Per-run history metrics persist at fit time (`planning/history.py` → `run_metrics` table; knob `inference.metrics_draws`, 0 disables; backfill: `python -m mmm_framework.api.backfill`). Endpoints: `/projects/{id}/experiment-priorities|history|calibration-coverage`. Agent tools: `compute_experiment_priorities`, `design_experiment_plan` (randomized matched-pair geo lift / matched-market DiD with DiD power+placebo math, or budget-neutral randomized flighting for national data — `planning/design.py`, pure pandas, works pre-fit; UI: "Design experiment" studio on /experiments via `POST /projects/{id}/experiment-design`), `plan_experiment`, `preregister_experiment`, `record_experiment_readout`, `apply_experiment_calibration`. Demo: `uv run python scripts/seed_demo_project.py [--synthetic-records]` (replaces the prior demo project; seeds chats + workspace state). |
| Model-anchored experiment economics | `design_experiment_plan` is **enriched by the fitted model** (+ a heavier `simulate_experiment` tool). Three pieces (`technical-docs/experiment-economics.md`, an adversarially-verified spec): (1) **model anchor** (`planning/design_anchor.py`) — perturbs ONLY treated geo×test-window rows via `sample_channel_contributions` (NOT the global response curve) → the channel's expected **incremental ROAS** + a powered/underpowered/overpowered/inconclusive verdict (signed two-sided assurance); feeds the realized `sigma_exp`+incremental-ROAS draws back into `compute_experiment_priorities` (new `roi_draws_overrides` kwarg, draw-paired at matching `max_draws`). (2) **opportunity cost / short-term risk** (`planning/opportunity_cost.py`) — counterfactual KPI delta vs BAU with posterior uncertainty; **signed** `spend_delta` computed internally (NEVER `design['weekly_spend_delta']`, which is `abs()` → would invert holdout net); forgone KPI, spend-at-risk, net-$ (optional margin via param or project `economics` preference; `kpi_kind` revenue/units/other), `opportunity_cost_dollar=max(0,-net)`, learning-vs-cost vs EVOI. A holdout can be **net-positive** (saved spend > forgone margin). Geo/window mapping uses `mmm.geo_idx`/`time_idx`; ragged panels intersect per-geo coverage → `duration_effective`. (3) **A/A·A/B simulation** (`planning/simulation.py`) — estimator-pluggable (pooled/per-pair DiD, synthetic-control geo, national on/off); A/A measures the **empirical false-positive rate** (autocorrelation inflates the analytic rule — a block-calibrated critical value restores size); A/B injects the model's predicted lift onto real history → empirical power/MDE (probit-fit); `methodology_leaderboard` recommends the valid+powered+cheapest method (runs pre-fit too, fixed injector). Wiring: model-op `experiment_economics` (`agents/model_ops.py`, `allow_unfitted`); non-blocking `POST /projects/{id}/experiment-design/simulate` + `GET .../simulate/{job_id}` (loads the latest saved model in ONE `asyncio.to_thread` per the ContextVar rule, persists a pollable `experiment_simulation` artifact via new `sessions.update_artifact_payload`); React DesignStudio "Opportunity cost & risk" + "Methodology comparison" panels. Tests: `tests/test_planning_opportunity_cost.py`,`_simulation.py`,`_design_anchor.py`,`test_experiment_economics_wiring.py`. |
| Off-panel calibration (experiment ran in a different period) | An experiment from a window the model was **not** fit on still calibrates — the agent's old "the window must overlap the training data" was an implementation limit, not a statistical law. Trigger: set `ExperimentMeasurement.eval_spend` (per-period, per-treated-unit raw $) + `eval_periods` (W) + `eval_units` (default 1) + `adstock_state` (`steady_state`\|`cold_start`). Then `_add_experiment_likelihoods` (`model/base.py`) routes to `_offpanel_contribution_std`, which evaluates the channel's **global** response curve `beta·sat(adstock(s_norm))` at that spend (steady-state = `s_norm·Σweights`; cold-start = ramp via `cumsum(weights)`), summed over W and scaled by `eval_units` — **no training-row indexing**, so window overlap is irrelevant. Requires the **parametric** adstock path (legacy fixed-alpha blend warns+skips); `eval_spend` normalized by `_media_raw_max`; off-panel mROAS + `holdout_regions` rejected in `__post_init__` (use `eval_units`). Kernel weights exposed by refactoring `_channel_adstock_apply` → `(closure, weights)` and captured in the channel handle as `adstock_weights`. Rests on **structural stationarity** (response curve stable across the two periods) — stated in `calibration/likelihood.py`'s "Assumed semantics". Agent flow: `record_experiment_readout(spend_per_period=…, n_treated_units=…, adstock_state=…)` stores it in the readout; `apply_experiment_calibration` stages out-of-window experiments off-panel (inferring `eval_periods` from the date span × dataset cadence) or returns a non-blocking advisory asking for the spend level (no more hard "outside the dataset's date range" block). Tests: `tests/test_calibration_likelihood.py::TestOffPanel*`, `tests/test_calibrated_fit_spec.py::test_apply_calibration_offpanel*`. |
| Experiment optimizer (suggest setup + Pareto front) | `planning/experiment_optimizer.py` suggests a runnable setup and the **Pareto front** of designs trading **MDE ↓ × power shortfall ↓ × short-term cost ↓ × duration ↓** (FOUR objectives). **Statistical power** is a first-class axis: `power = mean_d[Φ(eff_d/se−z)+Φ(−eff_d/se−z)]` (se=mde_roas/2.8, eff_d = reference incremental-ROAS draws → power≈80% at effect=MDE), and the objective is `power_shortfall = max(0, power_target−power)` (default target 0.80; 0 once met, so above-target designs compete on the other axes and below-target ones are pushed to the bar). No extra posterior passes. **National flighting is multi-level**: the spend range becomes the schedule's spend LEVELS (`flighting_design(levels=…)`); a ≥3-level schedule traces the response CURVE (vs a binary on/off). Power for flighting is computed for **ROAS, contribution, and mROAS separately** (`_flighting_power_breakdown` + `design.flighting_estimand_ses`: quadratic `σ²(XᵀX)⁻¹` → LEVEL `g(x₀)` SE = contribution/avg-ROAS detectability [coincide, rescaled by known spend]; SLOPE `g'(x₀)` SE = mROAS power, tangent only with ≥3 levels else a secant flagged `mroas_identified=False`); binding `min(roas,mroas)` drives the Pareto objective. Surfaced in op markdown/table + DesignStudio power-by-estimand card. The design space is **range-bounded**: `duration_min`/`duration_max` (weeks) + **signed** `intensity_min`/`intensity_max` (spend-variation %, `-100` go dark → `+150` scale up; `_duration_grid`/`_intensity_grid` auto-sample endpoint-inclusive points, explicit lists override) — exposed on the op/tool/endpoint (`ExperimentOptimizeRequest`) + DesignStudio range inputs. `cooldown_weeks(mmm, channel)` = adstock washout (smallest `k` with `alpha**k < 5%`, from `compute_adstock_weights`; unknown→4, none→1, alpha≥1→max) — post-test recovery + min flighting block. `evaluate_experiment_grid` sweeps footprint×intensity×duration: MDE from ONE pure-pandas power-curve call per (footprint,intensity), opportunity cost reuses ONE shared BAU posterior pass (`contrib_bau` kwarg). Tradeoff axis = net-$ downside when margin known (money-saving holdout → ~0), else forgone KPI (holdout) / spend-at-risk (scaling-up, so +100% doesn't tie +50% at 0). `pareto_front` = non-dominated over the 4 objectives; `recommend` = the knee among *powered* (≥target) front designs; `suggest_experiment` returns the recommended design's test/control geo groups (or flighting schedule+block) + duration + cool-down. Wiring: model-op `experiment_optimizer` (requires a fit), agent tool `suggest_experiment`, non-blocking `POST/GET /projects/{id}/experiment-design/optimize` (reuses the generalized `_load_and_run_op`/`_run_model_op_job`/`_spawn_job_task` job machinery), React DesignStudio "Optimize (Pareto front)" panel (power column + rust ring on under-powered designs). Tests: `tests/test_planning_experiment_optimizer.py`. |
| Declarative estimands (counterfactual causal lens) | `src/mmm_framework/estimands/` is a first-class, **named, serializable** estimand subsystem subsuming the framework's four scattered estimand notions into one registry while keeping every number **bit-stable** (`technical-docs/estimands.md`). An `Estimand` (`spec.py`, pure Pydantic, no numpy/pytensor at import) is `reduce(op(quantity|intervention, quantity|baseline))/normalizer` realized as mean+HDI. Two engines share ONLY `spec.py`: `evaluate.py` (**post-hoc**, numpy, `EstimandEvaluator` → `BayesianMMM.predict_under`) and `graph.py` (**in-graph**, pytensor; canonical home of `build_estimand_expr` — `calibration/likelihood.build_estimand_expr` now re-exports it). The FOUR legacy notions = FOUR built-ins (`registry.py`): `contribution_roi` (=`compute_roi_with_uncertainty`, the **dashboard/UI** number — in-graph Deterministic, `mean(roi_samples)`, `az.hdi`, drops `spend<=0`), `counterfactual_roi` (=`compute_channel_roi`, zero-out predict, diff-of-means, percentile HDI, **unpaired** seeds — a *different* number), `marginal_roas` (=`compute_marginal_contributions`, `_hdi_finite`, **paired** seed), `contribution` (=`compute_counterfactual_contributions` total); plus demonstrators `awareness_lift` (mean lift, no denom) + `cost_per_conversion` (inverted ratio). Bit-stability is pinned by `Realization{point_rule,hdi_method}` + the `evaluate.py` reuse of the *exact* legacy sample-extractors; equivalence gate `tests/test_estimands.py`. Capability gating: `capabilities.model_capabilities(model)` (duck-typed flags incl. `HAS_LATENT:<name>`) → `registry.defaults_for(caps)` (MMM defaults `[contribution_roi, marginal_roas, contribution]`, filtered by capability not class). Wildcard `target="*"` expands per channel → key `"{name}:{channel}"`. Model surface: `predict_under`/`model_capabilities`/`declared_estimands`/`evaluate_estimands`. Threading: `spec["estimands"]`→`declared_estimands` (`agents/fitting.build_model`), `MMMResults.estimands` (best-effort at fit, **only when declared**), serialization round-trip (w/ `schema_version`), agent op `compute_estimands` (`roi_metrics` untouched — flipping it would regress models with only `beta_<channel>` since the engine's `contribution_roi` is capability-gated stricter than `compute_roi_with_uncertainty`'s `beta*media` fallback), advisory garden manifest `default_estimands`/`capabilities` (or class-level `DEFAULT_ESTIMANDS`). Deferred: alt likelihoods, per-family config schema, contract REQUIRED_ATTRS gating, a real non-MMM family. |
| Guide chat / onboarding / team | Floating guide bubble (AppShell pages) talks to a per-project "✦ Project guide" session (`POST /projects/{id}/guide`, idempotent) through the normal `/chat` SSE with a `page_context` field. Project onboarding (`POST /projects/{id}/onboarding`) saves `projects.meta_json` (client/goals/KPIs/channels/constraints) and renders+ingests `project_brief.md` into the project KB so guide AND session chats retrieve it. Team roster: `users` + `project_members` tables (owner/analyst/viewer — attribution, not authentication), `/users` CRUD, `/projects/{id}/members`, UI at `/team`. |
| Client branding / preferences | Stored in the sessions-store `preferences` table: global (`GET/PUT /preferences`; PUT 403s when hosted) + per-project branding (`GET/PUT /projects/{id}/branding`, hex-validated `agents/branding.py:Branding`). Agent tools: `get_preferences`, `save_preference`, `list_templates`, `extract_brand_from_website` (+`POST /projects/{id}/branding/extract` — SSRF-guarded server-side fetch in `agents/brand_extract.py`; hosted: disabled unless `MMM_BRAND_FETCH_ALLOW=1`). **Confirmed** branding auto-recolors agent/EDA plots (`apply_brand_colors`, applied host-side at plot-store time — old plots keep their colors) and brands client reports (`generate_client_report(template=client\|minimal\|presentation\|full)`) + project reports/slides (`report_builder.apply_branding_html`). Extracted branding saves with `confirmed:false` and never styles output until confirmed. |
| Hosted multi-user profile | Set `MMM_AGENT_HOSTED=1` to flip from the single-user dev posture to the **hosted** posture (Phase 3 PR-F.6, `agents/profile.py`): the kernel defaults to the sandboxed `container` impl + fail-closed isolation, egress is denied, `Path.cwd()` is dropped from the download allow-roots, agent reports go **per-session** under the workspace (`/report*` endpoints take `?thread_id=`), and `/chat` refuses guessable/unknown `thread_id`s (server-minted `POST /sessions` only). Requires the kernel image built (`deploy/kernel/Containerfile`) + a container runtime. Inert/unsafe unless the Tier-2 sandbox is present — don't set it without the image. |
| `execute_python` kernel mode | `execute_python` runs behind a `KernelManager` (`agents/kernels.py`). Default `MMM_AGENT_KERNEL=inprocess` (the in-process warm namespace; `mmm`/`results` available after a fit). `MMM_AGENT_KERNEL=subprocess` runs one isolated `ipykernel` per session; `MMM_AGENT_KERNEL=container` runs each session's kernel inside `podman run` (`agents/container_kernel.py`) — sandboxed (scrubbed env, read-only rootfs, `--cap-drop ALL`, cgroup mem/pids/cpu caps, egress-deny, ephemeral overlay; build the image with `deploy/kernel/Containerfile`; knobs: `MMM_KERNEL_RUNTIME`/`_RUNTIME_BIN`, `MMM_KERNEL_IMAGE`, `MMM_KERNEL_TRANSPORT` (ipc/tcp), `MMM_KERNEL_MEM`/`_PIDS`/`_CPUS`, `MMM_KERNEL_EGRESS`, `MMM_KERNEL_REQUIRE_SANDBOX`). All three need no extra service (`jupyter_client`/`ipykernel` are deps). As of **Phase 2**, fits run **in** the kernel, so `mmm`/`results` are kernel globals under `subprocess` too (a cold/evicted kernel rehydrates the last fit from `<work_dir>/mmm_models/`). **Phase 3 PR-E.1:** the subprocess kernel is spawned with a **scrubbed env** (no `*_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, `MMM_LLM_API_KEY`, etc. — it never calls the LLM/embedder); fail-closed allowlist + secret-pattern denylist. Opt out with `MMM_KERNEL_SCRUB_ENV=0` (debug only); add a rare needed var with `MMM_KERNEL_ENV_PASSTHROUGH=NAME1,NAME2` (denylist still wins). NB: env-scrub does **not** block the cloud metadata server (ADC theft) — that's egress (Phase 3 Tier 2). **Phase 3 PR-E.3/E.4:** captured plots are thread-salted + schema-validated + size-capped (`MMM_PLOT_MAX_BYTES`, default 5 MiB) before storage; kernel lifecycle/security events log to the `mmm_audit` logger (`kernel_spawn`/`_evict_lru`/`_died`/`_timeout_kill`, `plot_rejected`). Tunables (subprocess): `MMM_MAX_KERNELS` (live-kernel LRU cap, default 8), `MMM_CELL_TIMEOUT` (per-cell wall-clock seconds before interrupt→kill, default 600), `MMM_FIT_TIMEOUT`, `MMM_KERNEL_RECV_TIMEOUT`/`MMM_KERNEL_READY_TIMEOUT`. See `technical-docs/agent-session-kernels.md` + `-phase1.md`/`-phase2.md`/`-phase3.md`. |
| Atelier notebook (demo/test a bespoke model) | A Jupyter-like space in the Atelier (Model Garden) 3rd center tab ("Notebook"): upload a dataset, run free-form Python cells against the **live editor buffer** (no register step), track plot/table/markdown outputs, persisted per model. Backed by the existing kernel + non-blocking job machinery — cells run via `KernelManager.execute` in a per-notebook synthetic thread (`_notebook_tid` → `__atelier_nb__{org}__{name}__draft`); the live source is staged into `garden_loaded_dir(...)` and imported kernel-side (bound to `GardenModel`, only re-imported when `source_rev` changes); the uploaded CSV auto-binds as `df`. Output mapping reuses `store_plot`/`publish_tables` (same content-addressing as `execute_python`). Endpoints (src/mmm_framework/api/main.py, registered BEFORE the parametric `/model-garden/{name}` routes): `GET/PUT /model-garden/notebook` (doc + seeded starter `_notebook_starter`), `POST /model-garden/notebook/dataset`, `POST /model-garden/notebook/cell` + `GET .../cell/{job_id}`. All org-scoped + analyst-gated; untrusted source execs only in the kernel (container sandbox when hosted). Frontend: `components/modelGarden/AtelierNotebook.tsx` + `NotebookCell.tsx` (reuse `PlotCard`/`TableCard`), `api/services/atelierNotebookService.ts`, `api/hooks/useAtelierNotebook.ts`. Tests: `tests/test_atelier_notebook.py`. Spec: `technical-docs/atelier-notebook.md`. **Notebook copilot** (diagnose cell errors + tips + rewrites): a toolbar **Copilot** toggle opens a chat rail; an errored cell shows **Diagnose with copilot** → auto-asks the assistant to fix THAT cell, with **Apply to cell** writing the rewrite back + clearing the stale error (else **Insert as new cell**/Copy); a model-class fix instead offers **Apply to editor (model source)** (`onApplyToEditor`→page `applyCode`, since the kernel imports the model from the editor buffer, not a cell). Reuses the existing `POST /model-garden/copilot` SSE (a grounded LLM, no agent thread) with a new optional `notebook` context (`NotebookCopilotContext`: cell_code/traceback/dataset_preview/other_cells/is_error); `build_copilot_system_prompt(source_code, notebook=…)` appends `NOTEBOOK_DIAGNOSIS_KNOWLEDGE` (real PyMC/MMM failure modes incl. `-inf` start-point logp) + a fix-this-cell instruction. New FE: `NotebookCopilotPanel.tsx`, `copilotMarkdown.tsx` (shared md renderer/`lastCodeBlock`), `copilotService.streamCopilot(…, notebook?)` + `readCopilotStream`. Chat is ephemeral. Tests: `TestDiagnosisPrompt`. |
| Data Studio (upload → interactive EDA → clean → convert to dataset) | A dedicated **"Upload & clean data"** button in the Oracle **Data** tab opens a full-screen **Data Studio** that stages a raw upload (NOT yet the working dataset), runs interactive EDA on it (tabs: Overview / Distributions / Correlation / Missingness / Outliers / Transform), and lets the user build an ordered, replayable cleaning **pipeline** before committing. On **"Use as dataset"** the cleaned frame becomes the session's working dataset with **no chat/LLM round-trip**. **Backend** `src/mmm_framework/data_studio/`: `transforms.py` = pure shape-aware `apply_pipeline(df, steps)` ops (rename/drop_columns/cast/parse_date/fill_missing/drop_duplicates/filter_rows/date_range/winsorize/impute/event_dummy) that work on BOTH wide & MFF-long frames (unknown op / missing param → `TransformError`→400; data-level errors → `warnings`, step skipped); `service.py` = staging manifest IO under `thread_dir/data_studio/` (raw file + `manifest.json` with steps+roles; **heavy data stays on disk**, only a light `dashboard_data["data_studio"]` pointer rides agent state), `run_eda_on_frame` (reuses the whole `mmm_framework.eda` engine — `validate_dataset`/`profile_panel`/`collinearity_analysis`/`detect_outliers`/`recommend_treatments`/the `fig_*` charts — returning figures **INLINE** as `{key,title,data,layout}`, never pushed to `dashboard_data["plots"]`), and `commit_core` (mirrors `_apply_outlier_treatment_core`). **Loader:** `eda/loading.py::load_eda_panel_from_df(df, spec)` (path-free sibling of `load_eda_panel`) runs EDA on the in-memory transformed frame. **Commit emits MFF-long** (NOT `spec["dataset"]`): wide uploads are melted on the date-role column over role-bearing columns and the canonical MFF dim columns (Geography/Product/… = None for national) are added so it rides the tested `build_model` MFF branch (`load_mff` + the data-quality gate); sets `dataset_path` + spec roles (`kpi`/`media_channels`/`control_variables`/`time_granularity`) via `aupdate_state(..., as_node="agent")` through `reconcile_with_locks`; `kpi_level="geo"` only when a real `group` role exists. **Endpoints** (`src/mmm_framework/api/main.py`, next to `/outliers/apply`): `POST /data-studio/{tid}/upload` (multipart, `_safe_upload_name`+`safe_join`), `GET /data-studio/{tid}` (hydrate), `PUT .../pipeline` (full-replace + re-preview), `POST .../eda`, `POST .../commit` (no-LLM, no messages), `POST .../discard`. **Gotcha:** `graph.py::should_continue` now returns `END` on empty messages so UI-driven `as_node="agent"` writes work on a cold thread. **Frontend** `pages/Agent/components/dataStudio/`: `DataStudioModal` (bespoke 3-zone full-screen shell) + `useDataStudio` (raw fetch + `authHeaders`; out-of-order pipeline responses dropped via a monotonic txn id; per-(rev,analyses) EDA cache); `StudioEdaChart` passes `{data,layout}` (no id) so `PlotCard`/`usePlotFigure` render inline; reuses `IssueRow`/`OutlierActionRow` (now exported from `EdaTab`), `DataTable`, `DashWidget`, `common/form.tsx`. Wired via `WorkspaceTabs` Data tab + `index.tsx::handleDatasetCommitted` (merges `{dataset,eda,model_spec}` into `dashboardData` + the same refresh trio as `handleResolveOutlierAction`). The chat paperclip (quick agent-led path) stays. Tests: `tests/data_studio/{test_transforms,test_service,test_endpoints}.py` (incl. a slow wide→commit→fit smoke proving the MFF branch), FE `dataStudio/*.test.tsx`. |
| Per-model config + pluggable likelihood (e.g. a binomial awareness model) | A custom model declares its **own** config fields with defaults + a non-default observation likelihood, so the monolithic `ModelConfig` (`extra:"forbid"`) no longer blocks bespoke params like a binomial awareness model's `number_of_trials`. **Config schema:** a model sets a class attr `CONFIG_SCHEMA` (a Pydantic model) → bespoke params arrive via the optional kw ctor arg `model_params` (`base.py::_coerce_model_params` validates/defaults through the schema; base model passes through). Threaded by `agents.fitting.build_model` (`spec["model_params"]`, clear build-context error), round-tripped by `MMMSerializer` (plain dict + `model_params_schema_version`; reload re-validates), and the garden manifest carries an advisory `config_schema` JSON Schema for a dynamic UI form. **Likelihood:** `ModelConfig.likelihood: LikelihoodConfig` (`{family, link, params}`, default `normal`/`identity` ⇒ **byte-identical** old graph; `config/likelihood.py`, `LikelihoodFamily`/`LinkFunction` in `config/enums.py`, builder `with_likelihood`, spec `spec["likelihood"]`). `_prepare_data` **conditionally standardizes** (Gaussian families z-score `y`; count/bounded keep natural scale, `y_mean=0/y_std=1` so `y_obs_scaled` + every `* y_std` bridge stay identity). The built-in additive dispatch `base.py::_build_likelihood` fits ONLY Gaussian families (`normal` byte-identical, `student_t`); **non-Gaussian families raise** `NotImplementedError` on the additive model (its priors assume standardized-Normal `y`) — they belong to models that **own their observation block** (override `_build_model`), reading `self.model_config.likelihood`. `n_trials` is **not** required in `LikelihoodConfig.params` (the family declares scale/observation type; a model sources its count from `CONFIG_SCHEMA`, e.g. `number_of_trials`). Estimand refs resolve names too (`evaluate_estimands`/`DEFAULT_ESTIMANDS` accept built-in names via `_resolve_estimand`). Worked example: `examples/garden_models/awareness_structural_mmm.py` (`AwarenessParams` CONFIG_SCHEMA + Normal index KPI **or** Binomial survey-count KPI + `DEFAULT_ESTIMANDS=["awareness_lift","contribution_roi"]`). Spec: `technical-docs/custom-model-config.md`. Tests: `tests/test_likelihood_config.py`, `test_model_likelihood_dispatch.py`, `test_model_params.py`, `test_awareness_garden_model.py`. |
| Non-MMM model families (CFA / LCA / EFA) | A genuinely **non-MMM** Bayesian family (no channels/spend/single-KPI) rides the same garden → fit → estimand → serialize → report pipeline. **Gate:** a class declares `__garden_model_kind__` (default `"mmm"`); `garden/contract.py::is_mmm_model(obj)` is **True unless explicitly declared non-`"mmm"`** (duck-typed/unknown → treated as MMM, historical default). MMM-only checks gate behind it: `validate_class` read surface (`predict`+`sample_channel_contributions`), `validate_instance` channel attrs (`REQUIRED_ATTRS` split into `_BASE` + `_MMM`), `validate_fitted` `beta_<channel>`, compat `scaling`/`ops_smoke`/`accuracy` tiers (self-skip; `static`/`build`/`fit`/`instance`/`trace` still run), and the serializer's channel/control panel-match + y/media re-standardization (records `model_kind` in metadata; `_resolve_model_class` now also imports a non-garden subclass by `model_class_qualname`). Manifest gains advisory `model_kind` (AST-detected, `garden_registry.static_model_kind`). **Estimands:** the engine auto-exposes every posterior var as `HAS_LATENT:<var>`; `evaluate.py::_latent_quantity` realizes a **bare `LatentVar`** (scalar→mean+HDI, obs-indexed→window mean, matrix→`unsupported`). `registry.latent_scalar`/`fit_index`/`factor_loading` build them, gated by `HAS_LATENT:<var>`. **Worked example:** `examples/garden_models/bayesian_cfa.py` — a confirmatory factor analysis (`CFAConfig`: `n_factors`/`factor_assignment`/priors), marginal MvNormal likelihood `MvNormal(0, ΛΛᵀ+Ψ)` with positive HalfNormal loadings on a fixed simple structure, per-draw `srmr`/`cov_fit` deterministics, `DEFAULT_ESTIMANDS=[fit_index("srmr"),fit_index("cov_fit")]`, `factor_loadings_summary()`; overrides only `_prepare_data` (indicators from panel: kpi+media+controls uniformly; sets `channel_names=[]`) + `_build_model`. Recovers a planted 2-factor structure end-to-end. Spec: `technical-docs/non-mmm-families.md`. Tests: `tests/test_non_mmm_families.py`, `tests/test_cfa_garden_model.py`. Deferred: LCA/EFA. |
| Latent-variable contrasts + CFA HTML report | Two follow-ons to the non-MMM work. **(1) Latent contrasts:** a `LatentVar` used in a counterfactual `Contrast` (intervention vs baseline) is now realized — `BayesianMMM.sample_latent_under(var_name, intervention)` re-evaluates ANY registered deterministic under intervention-perturbed media (`set_data` + `sample_posterior_predictive([var_name])`, generalizes `sample_channel_contributions`), and `evaluate.py::_eval_latent_contrast` reduces intervention−baseline over the window (scalar or obs-indexed `(n_obs,)`; higher-dim → `unsupported`; native model scale, so constant scaling cancels; degrades if the model has no `sample_latent_under`). Demo: the awareness model's `media_total` goodwill stock under media-on vs a channel-off. Tests: `tests/test_latent_contrasts.py`. **(2) CFA HTML report:** a `FactorAnalysisSection` (loadings table + fit-index cards using the existing `metrics-grid`/`metric-card` CSS) + `FactorAnalysisExtractor` (reads `factor_loadings_summary()` + `evaluate_estimands()`; sets `bundle.model_kind`). `reporting/extractors/create_extractor` routes non-MMM models (`is_mmm_model==False`) to it; `MMMReportGenerator._initialize_sections` gates the channel/ROI/decomposition/saturation/geo/mediator/cannibalization sections OFF and the factor-analysis section ON for non-MMM (via `bundle.model_kind`). `ReportConfig.factor_analysis` SectionConfig. Also hardened `DiagnosticsSection` for **approximate (MAP) fits** (R-hat/ESS `None` → "N/A", not a crash). Tests: `tests/test_cfa_report.py`. |
| Second non-MMM family: Bayesian LCA + family-agnostic report | `examples/garden_models/bayesian_lca.py` — a **latent class analysis** (mixture over **binary** indicators, `K` discrete classes), `__garden_model_kind__="latent_class"`, `LCAConfig` (`n_classes`/priors). Discrete labels **integrated out**: per-obs loglik = `logsumexp` log-mixture of Bernoulli products via `pm.Potential` (no discrete latents → NUTS-able); mixing `π`=softmax of an **ordered** logit (pins class order by size → resolves label-switching); per-class item probs `θ`=Beta. Estimands = **class sizes** (`class_size_k`) via an overridden `_default_estimands` (dynamic in `n_classes`); `class_profile_summary()` = per-(class,item) table. Recovers a planted 2-class structure end-to-end. Overrides only `_prepare_data` (binarizes all observed columns) + `_build_model`. The non-MMM **report was generalized**: `FactorAnalysisSection`/`FactorAnalysisExtractor` are now **family-agnostic** — render a summary table (CFA `factor_loadings_summary` OR LCA `class_profile_summary`) + declared estimands as cards, with per-family headings from `bundle.{latent_section_title,latent_table_title,latent_estimands_title}` + a column-agnostic table renderer (CFA report tests stay green). Tests: `tests/test_lca_garden_model.py`. Spec: `technical-docs/non-mmm-families.md`. EFA (rotation) still deferred. |
| Joint latent-factor MMM (combine many factors into one latent → MMM covariate) | A genuinely **joint** Bayesian model: a latent-factor *measurement block* (many indicators → one latent construct, e.g. "economic health") estimated IN THE SAME PyMC graph as an MMM that uses the latent as a KPI covariate — so the factor's uncertainty propagates into the media coefficients (NOT a 2-stage plug-in / "generated regressor"). The point: economic health is a **common cause** (boom → more spend AND more sales), so a naive MMM over-credits demand-chasing channels; conditioning on the de-noised latent closes the back-door `spend ← econ → sales`. **Worked example:** `examples/garden_models/latent_factor_mmm.py` — `LatentFactorMMM(CustomMMM)`, sibling of `awareness_structural_mmm.py` (overrides only `_prepare_data`+`_build_model`, reuses `_channel_adstock_apply`/`_build_channel_saturation`/`_apply_saturation_pt`/`_sample_from_prior_config`/`_build_control_betas`/seasonality). **Measurement block** on the PERIOD axis (national factor; `_prepare_data` collapses indicators to one row per period via `time_idx` so a geo panel doesn't G-fold over-weight it): AR(1) latent via the awareness model's scan-free lower-triangular **decay-matrix** trick (`econ = decay@eps`), `factor_dynamics="static"` swaps an iid `N(0,1)` factor; `indicatorₖ ~ Normal(λₖ·econ+aₖ, ψₖ)`. **CRITICAL identification (two fixes, both load-bearing):** (1) the realized factor is **standardized to unit variance in-graph** (`(econ-mean)/std`) — without it the AR(1) variance `1/(1−ρ²)` trades off against the loadings and the sampler **collapses loadings→0** (factor drifts as noise; corr crashes to ~0.4); standardizing pins scale so loadings carry the identified indicator↔factor correlations and ρ only sets *shape*; (2) the FIRST (anchor) loading is `HalfNormal` (positive → fixes orientation/sign), the rest `Normal` (free sign → a genuinely negative indicator like unemployment recovers a NEGATIVE loading; **don't force all-positive loadings — that breaks mixed-sign DGPs**). **Coupling:** `beta_economic_health·econ[time_idx]` added to the MMM mean. Registers the full read-op contract (`channel_contributions`/`media_total`/`controls_total`/`y_obs_scaled`/`beta_<ch>`/`adstock_alpha_<ch>`) PLUS `factor_loadings`/`loading_<ind>`/`economic_health`(period)/`economic_health_obs`/`economic_health_contribution`/`beta_economic_health` — each auto-exposes `HAS_LATENT:<var>`. **Data contract:** indicators tagged `DatasetRole.INDICATOR` (NOT `CONTROL` — `Dataset.X_controls` is CONTROL-only so they can't leak into the regression) via the role-tagged `spec["dataset"]`/`Dataset.from_wide` path; `REQUIRED_ROLES=(TARGET,PREDICTOR,INDICATOR)`+`REQUIRED_DATASET_CAPABILITIES=("HAS_INDICATORS",)` reject a plain MFF panel; `model_params.indicator_columns` is a by-name fallback. **Stays `__garden_model_kind__="mmm"`** (full channel/ROI/compat). **Estimands:** dynamic `_default_estimands` = `contribution_roi`+`marginal_roas` (de-biased ROI) + `latent_scalar("economic_health_level", var="economic_health_obs")` + per-indicator `factor_loading("loading_<ind>")`; `factor_loadings_summary()` for the table. **Hybrid report (the only `src/` changes besides the DGP):** keep kind `"mmm"` but turn the factor section ON *data-driven* — `garden/contract.has_latent_structure(obj)` (duck-types `factor_loadings_summary`/`class_profile_summary`), `BayesianMMMExtractor._extract_latent_structure` fills `bundle.factor_loadings`/latent titles when present (reuses `FactorAnalysisExtractor._table/_estimands`), and `generator._initialize_sections` gates `factor_analysis` on `has_latent` (`bundle.factor_loadings|cfa_fit_indices` non-empty) instead of `_non_mmm` — so a pure MMM (empty fields) stays off, pure CFA/LCA unchanged, hybrid gets channel/ROI **and** factor sections. **DGP:** `synth/dgp.make_economic_health` (latent AR(1) econ confounds spend [`kappa`, Search/Social chase] and sales [`theta` back-door]; 4 indicators with known `true_loadings` incl. a **negative** unemployment loading; answer key in `Scenario.notes`). **Gotchas:** MAP is **too unstable** for the ~150-param latent model (loadings/ROI swing run-to-run) — the test fits **NUTS, 4 chains, ≥800 tune** (a 2-chain quick fit under-mixes → corr collapses); loading *magnitudes* aren't recovered (AR(1) scale absorbed) so assert **signs + factor |corr|**, not magnitudes; serialization round-trip must reload against the role-tagged **`Dataset`** (not a bare `PanelDataset`, which has no INDICATOR columns). Tests: `tests/test_latent_factor_mmm.py` (headline: joint chaser-ROAS error << naive-MMM-without-indicators). Deferred: multiple latent factors (CFA-style `factor_assignment`), reusable `src/` measurement-block primitive (extract to `mmm_extensions/components/latent_factor.py` on a 2nd consumer), per-geo factors, agent/API/UI. |
| Atelier / Model Garden docs + demo | **Docs page** `docs/model-garden.html` ("Model Garden & Atelier", in the Platform nav group + sitemap): the consumer/agent flow (`list_garden_models`→`load_garden_model`→`fit_mmm_model`→`get_estimands`), the Atelier IDE (Code/Docs/Notebook tabs), the authoring recipe (CONFIG_SCHEMA/likelihood/DEFAULT_ESTIMANDS, non-MMM `__garden_model_kind__`), the 9-tier compatibility contract, and the governance/ecosystem story. Snippet gate (`tests/test_docs_snippets.py`) green — note `is_mmm_model` is **not** re-exported from `mmm_framework.garden` (import from `.garden.contract`); built-in estimand ctors (`contribution_roi`…) are private (reference by string name). **Demo seeder** `scripts/seed_atelier_demo.py` [`--fast` no-MCMC / `--real-compat` genuine suite]: registers + **publishes** the 3 example garden models into the org (`register_garden_model_core` → `set_garden_compat_report` → `transition_garden_model` draft→tested→published; AST-detected `model_kind`, `config_schema` from `CONFIG_SCHEMA.model_json_schema()`, `default_estimands`), then seeds project "Demo: Atelier Custom Models" with one **Oracle session per model** — real MAP fits (reusing `synthetic_cfa_panel`/`synthetic_lca_panel` + the awareness binomial fixture), real declared estimands woven into a scripted list→load→fit→estimands chat, and `model_spec` carrying the `garden_ref`/`model_params`/likelihood (model config). Each session's workspace is also populated with **model-specific Plotly charts** (awareness lift/goodwill-efficiency/retention; CFA loadings + fit-indices; LCA class-profiles + sizes) via `agents.workspace.store_plot` → `dashboard_data["plots"]`, and **tables** (declared estimands, loadings, class profiles, a "Model configuration" table) via `agents.tables.publish_tables` (group `results`) → `dashboard_data["tables"]`, plus native `roi_metrics` for the awareness MMM — all rendered in the **Oracle Results tab** (verified in-browser via the run-app skill; every `/plots//tables/` fetch 200). Reuses `seed_demo_project._seed_chat` (LangGraph `AsyncSqliteSaver` checkpointer). Replaces prior project + purges prior model versions on re-run. No Redis needed. **Gotcha:** plots/tables resolve relative to `workspace_root()`, so run the seeder with the SAME `MMM_AGENT_WORKSPACE` as the API server (the run-app skill uses `/tmp/mmm_ws`), else they 404; the sessions DB is a fixed path (`src/mmm_framework/api/sessions.db`) so the project/sessions/models/chat show regardless. Report HTML is a disk artifact only (dev `/report` is global — not wired to the dashboard). |
| Estimand results (CI) + posterior-predictive GoF in reports | The MMM report has two **default-on, MMM-gated** sections: **EstimandsSection** (`reporting/sections.py`) renders the model's declared/default estimands (contribution ROI, marginal ROAS, incremental contribution per channel) as a point-estimate + credible-interval table with a Strong/Below/Uncertain evidence flag vs the no-effect reference (ROI/ROAS↔1.0, contribution↔0); and **PosteriorPredictiveSection** renders four goodness-of-fit views — observed-vs-predicted (45° + predictive-interval error bars), a replicated-dataset density overlay, predictive-interval **calibration** (empirical vs nominal coverage), and residuals-vs-fitted — plus an R²/coverage card row and a posterior-predictive **p-value** table (mean/std/min/max). Charts are native to the reporting kit (`reporting/charts/ppc.py`, `create_ppc_*` via `create_plotly_div`, themed by `ColorScheme` — NOT the `go.Figure`-style `validation/charts/diagnostics.py`). Data flow: the orchestration lives in a shared `EstimandPPCMixin` (`extractors/mixins.py`) that BOTH `BayesianMMMExtractor` and `ExtendedMMMExtractor` inherit — best-effort `_extract_estimands` (prefers `results.estimands` else `model.evaluate_estimands()`; only `status=="ok"`, finite `mean`) + `_extract_posterior_predictive` (coverage curve, `bayes_p`=P(T(y_rep)≥T(y_obs)) for mean/std/min/max, ≤40 linspace-thinned replicate curves, R²) into new `MMMDataBundle.estimands` / `.posterior_predictive` (+ `has_estimands`/`has_posterior_predictive`). Two hooks adapt per family: `_estimand_model()` (the model exposing `evaluate_estimands`) and `_ppc_arrays()` → aligned original-scale `(observed, y_rep, pred_mean, pred_lower, pred_upper)`. **Core MMM** `_ppc_arrays` resamples via `model.predict(return_original_scale=True)`; bails on any non-finite array (no NaN leaks into the report). **Extended models** (`ExtendedMMMExtractor`, NestedMMM/MultivariateMMM/CombinedMMM — `BaseExtendedMMM` has NO `predict()`/`evaluate_estimands()` and `fit()` samples no PPC group): `_ppc_arrays` **samples the model's PyMC graph directly** (`with model.model: pm.sample_posterior_predictive(var_names=["y_obs"])`), unstandardizes via `_outcome_scale()` (primary outcome for multi-outcome), so they get a REAL PPC section; estimands degrade gracefully (no `evaluate_estimands` → empty table). Gating: `generator._initialize_sections` wires both via `_mmm(...)` (`bundle.model_kind=="mmm"`) — **non-MMM (CFA/LCA) hides both**; those surface estimands via `FactorAnalysisSection`/`cfa_fit_indices`. New `ReportConfig.estimands`/`.posterior_predictive` SectionConfigs (both default-on; `minimal()`/`presentation()` disable PPC, keep estimands). **Gotcha:** estimand display label comes from the result-key **name** prefix (`contribution_roi`), NOT the result `kind` — built-ins share kind `"roi"` (would all render "Roi"). **Security:** EstimandsSection (and the pre-existing ChannelROI/Decomposition/Geographic/Mediator/Cannibalization/CausalAssumptions sections) now `html.escape()` all user-controlled NAME interpolations (channel/geo/product/component/source/target) — closes a stored-XSS via malicious column names + fixes the `format_channel_names` post-processor (which searches for `html.escape(ch)`). **Also fixed** a pre-existing crash: `charts/base.py::_hex_to_rgb` now parses hex (incl. `#rgb` shorthand) + `hsl(...)` (the hash-based `ChannelColors.get` fallback for channels outside the palette) + `rgb(...)`, with a neutral-gray fallback — previously it did `int('hs',16)` and threw, so a full NestedMMM report render (mediator pathway chart) crashed. Tests: `tests/reporting/test_ppc_estimands_reporting.py` (slow real Bayesian + NestedMMM fits, fast fake-model unit tests for finiteness/NaN/escaping), `test_charts.py::TestPPCCharts`, `test_sections.py::TestEstimandsSection`/`TestPosteriorPredictiveSection`, `test_generator.py::TestEstimandsAndPPCSections`. |
| Impression-/click-measured media (ROI vs efficiency) | When a channel's modeled variable is **impressions/clicks**, not dollars, ROI can't divide by the variable sum. A per-channel **measurement descriptor** on `MediaChannelConfig` (`measurement_unit: spend\|impressions\|clicks\|other` + optional `spend_column`/`cpm`/`cpc`; validator: one cost source, cost⇒non-spend unit, cpc⇒clicks) drives one **resolver** `reporting/helpers/measurement.py::resolve_channel_divisor(model, channel, mask) -> ChannelDivisor(total, found, meta)`. `total` is the **spend-/volume-equivalent of the (masked) window**, serving BOTH average (`contribution/total`) and marginal (`total*(factor-1)`) metrics. Precedence: SPEND default (= legacy `X_media_raw` sum, **byte-identical**) → `spend_column` (external $ series, loaded by `MFFLoader.build_panel`→`PanelDataset.spend_raw`→`model.spend_raw`; NOT in the PyMC graph — curve always fit on the modeled variable) → `cpm` (`impr/1000*cpm`) / `cpc` (`clicks*cpc`) → **efficiency** (`impr/1000` or `clicks`/unit), break-even **0** not 1.0, `prob_profitable` dropped. `MetricMeta` (is_monetary/cost_basis/roi_label/marginal_label/value_units/divisor_units/reference) labels every surface. **Wired**: `roi.py` (`compute_roi_with_uncertainty`+`_extract_spend_from_model`→`resolve_spend_dict`), `analysis.compute_channel_roi`, `model/base.compute_marginal_contributions`, `estimands/evaluate._observed_spend`/`_marginal_spend` (per-result `extra[metric_*]`; byte-stable for spend — `test_estimands` gate green), report `ChannelROISection`+forest line+`EstimandsSection` (reference from meta), extractors `bayesian`/`extended._compute_channel_roi`+`_compute_blended_roi` (**suppressed for mixed/efficiency portfolios** — Q4 per-channel-only), `mixins._extract_estimands`, `agents/estimand_rows`+`api/estimands` (Performance dashboard evidence is server-driven), agent ops `roi_metrics`/`compute_estimands`, `agents/fitting._mff_config_from_spec` (`measurement_unit`/`spend_column`/`cpm`/`cpc` spec keys). Resolver defaults to SPEND for mock/duck-typed configs (`_unit_of` catches bad enum). **Deferred**: in-graph experiment-calibration ROI bridge (`model/base.py`≈1919, still `X_media_raw`); time-varying cpm column; `spend_column` on RaggedMFFLoader; bespoke FE panels. Spec `technical-docs/impression-level-roi.md`; tests `tests/test_measurement_metrics.py` (incl. slow E2E fit). |
| Continuous sequential learning (model-free, no MMM) | `src/mmm_framework/continuous_learning/` is a self-contained Bayesian **geo response-surface bandit** that learns how spend drives outcome **directly from designed geo experiments** — NO pre-fit `BayesianMMM` required (the inverse of `planning/`, which sits on top of a fit). Implements `assets/continous_learning.md`; backend is **NumPyro/JAX** (chosen over PyMC so one differentiable surface is shared by the likelihood AND the allocator). **`surface.py`** = the single source of truth: `incremental(s)=Σβ·Hill(s_c;κ,α)+Σ_{c<c'}γ_{cc'}f_c f_c'`; the Hill activation is **convention-identical** to `SaturationType.HILL` (`slope=α`,`sat_half=κ`), so a CL posterior is directly comparable to a BayesianMMM Hill fit. `jax.jit` the scalar response + `grad` (compiled once, reused across draws); used by the model likelihood, the DGP, AND `planner` (which `jax.grad`s it) → optimizer can't drift from the fit. **Activation is PLUGGABLE** (not Hill-specific): `ACTIVATIONS` registry {name→(param_names, jax fn)} = `hill`(κ,α)+`logistic`(λ, f=1−exp(−λs), concave); `surface_value`/`surface_over_rows` eval any family. `fit(activation=…)` samples the family's shape (`_sample_activation_shape`); `Posterior.activation`+`shape_names`+generic `draw_params`→{beta,gamma,shape,act_fn}; `planner` activation-agnostic (`_params_kernel` handles new {shape,act_fn} AND legacy {κ,α} dicts; `_jitted` lru_cache per act_fn) + `response_grid(post,spend,draws)` for viz; `dgp.TrueWorld` generic (kappa/alpha back-compat kwargs) + `make_world_logistic` + `make_world_hill_mixture` (weighted-sum-of-two-Hills truth: `hill_mixture` activation w·Hill(κ1,α1)+(1−w)·Hill(κ2,α2) — a two-phase shape a single Hill can only average over and a concave logistic can't represent); `acquisition` Hill-only (guards, packs κ,α). Hill byte-identical. Notebook §13 (logistic) + §14 (**misspecification study** + a **wrong-family learning animation** `build_misspecification_animation.py` → `continuous_learning_misspecification.gif`: two accumulating loops [correct mixture vs wrong single-Hill] on the mixture world, 2×2 all-Scatter — top row curve recovery [correct band covers truth; wrong band tightens but MISSES it, red overlay + 0/4 coverage], bottom row profit-gap [both overlap → decision converges] + CI-width [wrong shrinks *below* correct → narrow-and-wrong/overconfident]; env `MIS_WRONG=logistic` for the severe case) + a logistic animation. **Misspecification finding** (empirically verified; `technical-docs/continuous-learning.md` §"When the response family is wrong"): fitting mixture-truth data with the WRONG family barely dents the **decision** (profit gap hill 0.9% / logistic 1.4% / correct-mixture 0.9% — the profit surface is flat at an interior optimum so any monotone-saturating curve gets the local marginal ordering right) but wrecks **calibration** (marginal-ROAS 90% CI covers truth: mixture 4/4 & widest, hill 3/4, logistic 2/4 yet *narrowest* → overconfident; a single Hill on two-Hill data often won't converge, R̂≈1.5 — a misspecification tell). The **sequential trust-region loop erases the mild misspecification** (real `LearningState`+`simulate_wave` accumulation, fixed `a_geo`: single-Hill on mixture truth converges 0.9%→0.5%→0.2%→0.3%, tracking the correct-mixture loop 0.9→0.5→0.1→0.1 to a fraction of a percent). **Gotcha the study surfaced**: the loop's guarantee needs a **stable geo set** — re-drawing `a_geo` each wave (calling `simulate_panel` per wave instead of `simulate_wave` with fixed `a_geo`) conflates two intercept draws under one `geo_idx` and makes it *diverge*. Rule: trust the ranking/funded-set, distrust channel-by-channel magnitudes; use the most flexible activation you can identify; watch CCD-cross-section residuals/R̂. Tests `test_hill_mixture_activation_properties`, `test_true_world_hill_mixture_response_matches_surface`, slow `test_misspecified_single_hill_still_makes_a_near_optimal_decision`. **`model.py`** = NumPyro model + priors (β~HalfNormal, κ~LogNormal, α~TruncNormal[0.5,5], geo intercept `a_geo~N(A,σ_a)`) + **sign-informed interaction priors** `PAIR_SIGNS` (neg/pos→±HalfNormal deterministic, zero/weak→Normal); `Posterior` container (`draw_params`/`gamma_matrix`/`gamma_summary`), `fit` (NUTS, chain_method=sequential, drops `__mag` latents), `demote_channel`/`probe_pairs_excluding` (walled gardens). **`design.py`** = `central_composite(center,delta,probe_pairs)` (1 center+2K axial+2·\|probe\| off-axis+K **shutoff** cells — shutoffs break β/γ collinearity; `delta` is MULTIPLICATIVE) + `assign_geos` (round-robin + holdouts). **`dgp.py`** = `TrueWorld`/`make_world` (known β/κ/α/γ incl. a negative cannibalization + positive complementarity) + `simulate_panel` (recovery harness: pre-period pins `a_geo`, test-period = CCD variation — the **identification requirement**) + `simulate_wave` (later waves, same geos/`a_geo`). **`planner.py`** = `allocate_under_sample` (multi-start SLSQP, surface non-concave from −γ; mode `fixed` eq-constraint `Σs=B` / `free` box), `thompson_wave`/`recommend_allocation`, `marginal_roas` (funding line `P(value·∂R/∂s>1)>0.5`), `expected_regret` (warm-started from consensus ⇒ `regret_d≥0`) + `enbs`/`should_stop` (`E[regret]·margin·pop−cost≤0`), `knowledge_gradient` (one-step EVSI; per-fantasy NUTS refit via `refit_fn_from_data` — expensive, Laplace-update deferred). **`loop.py`** = `LearningState` (carries posterior by **refitting on ALL accumulated data** each wave) + `run_closed_loop` (demo/closure backbone; swap `simulate_wave` for real holdouts in prod) + `due_for_retest` (REUSES `planning.eig.reexperiment_due`). Three feasibility gates green in `tests/test_continuous_learning.py` (recovery: signs not magnitudes; prior-sensitivity: weak-pair sd grows with `gamma_scale`; closure+stopping: E[regret]↓, ENBS fires). **`preprocess.py`** (guide §9.3/9.4) = `adstock_panel`/`adstock_prepass` (geometric-adstock the spend series within each geo, reusing `transforms.adstock`; `simulate_panel(adstock_alpha=…)` adds carryover to the DGP — response sees adstocked spend, observed spend is raw; pre-pass halves β error on a SHORT window, ~no-op in steady state since CCD holds each cell constant) + `cuped_adjust`/`cuped_covariate` (pre-period KPI covariate; `y_adj=y−θ·x_pre[geo]`, variance reduction `1−ρ²`). **`acquisition.py`** (guide §9.1/9.2, **no MCMC**) = moment-match posterior over `θ=[β,κ,α,γ_pairs]` to a Gaussian, jax-jacobian param gradients → `design_information` (Fisher info Λ=σ⁻²Σwₖgₖgₖᵀ, geo intercept profiled by centering cell gradients), `gaussian_eig`/`design_eig(target="all"|"gamma")` (D-/**D_s-optimal** pure EIG = ½(logdet Σ₀_SS−logdet Σpost_SS)), `laplace_knowledge_gradient` (EVSI via preposterior mean sampling `θ~N(μ,Σ₀−Σpost)`, ~milliseconds vs NUTS-refit KG). Key demo result: **Laplace KG (decision value) and D_s-opt EIG (γ-information) rank designs DIFFERENTLY** — off-axis cells identify synergies but main-effect cells move the allocation more (guide: EIG≠decision value). Notebook §7/8/9 visualize all three. Demo `examples/ex_continuous_learning.py` + baked notebook `nbs/continuous_learning.ipynb` (`build_continuous_learning.py`) + a **narrative** sibling `nbs/continuous_learning_story.ipynb` (`build_continuous_learning_story.py`) — the "Nomi" brand measurement-cycle walkthrough (business questions → confounding → experiment brief → 3-wave loop → funding line/synergy map → reallocation-with-confidence → ENBS stop → info-decay re-test → client one-slide + honesty caveats + truth reveal; make_world seed 7, 3 fits, no animations); spec `technical-docs/continuous-learning.md`. **Gotchas**: never normalize/center/log `y` (only scale spend, by a fixed global constant not a cluster mean); JAX float32 default (grad-vs-FD tests use float32-appropriate atol); R-hat needs ≥2 chains (1-chain fit → diagnostics None); notebook nbformat/nbconvert via `uv run --with`, inner `"""` breaks a `code(r"""…""")` cell. **Deferred**: only agent tools/API/UI (`learning_program`/`learning_wave` tables) + wiring the Laplace update INTO the loop's `knowledge_gradient` (it's a standalone acquisition for now). |
| LaTeX / math in chat + copilot + docs (frontend) | Every `react-markdown` surface renders LaTeX via `remark-math` + `rehype-katex` + `katex`, configured ONCE in `frontend/src/lib/markdownMath.ts` (`remarkPlugins`, `rehypePlugins`, `normalizeMath`); KaTeX CSS is imported in `main.tsx`. **Critical gotcha — currency vs math:** with single-`$` math on (needed for inline `$\beta$`), `remark-math` parses `"$5,000 and $3,000"` as the math `"5,000 and "` — corrupting the dollar amounts that pervade this app. So `normalizeMath` runs first (code-aware — skips ``` fences + `inline` code): `\[…\]`→`$$…$$`, `\(…\)`→`$…$` (models emit backslash delimiters that remark-math can't parse natively), and a lone `$` directly before a digit → `\$` (escaped, can't open math). Result: `$\beta$`/`\(\alpha\)`/`\[…\]` render; `$5,000` stays literal; only rare digit-leading inline `$5x$` is sacrificed. `rehypeKatex` uses `throwOnError:false` (bad LaTeX → inline error, never blanks the message). Wired into ALL 8 surfaces: `ChatMessageBubble` (agent chat), `CopilotPanel`+`NotebookCopilotPanel` (copilots), `GuideBubble`, `NotebookCell`, `ModelGarden/index` (3 docs), `GardenModelConfigWidget`. **Adding a new markdown surface:** import from `lib/markdownMath` + wrap children in `normalizeMath(...)`; don't re-add bare `remarkGfm`. No FE test runner — verify via a standalone `react-dom/server` render (math → `class="katex"`, currency → plain) or in-browser. |
