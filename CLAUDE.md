# CLAUDE.md — MMM Framework Developer Guide

**MMM Framework** is a production-ready Bayesian Marketing Mix Modeling framework built directly
on PyMC (it is *not* a PyMC-Marketing subclass — that library is only an optional reporting
interop target, `reporting/extractors/pymc_marketing.py`). It emphasizes methodological rigor,
genuine uncertainty quantification, and pre-specified analyses to reduce researcher degrees of
freedom.

- **Author**: Matthew Reda · **Python**: 3.12+
- **Version**: source of truth is `pyproject.toml` + `src/mmm_framework/__init__.py` (both must
  match; gated by `tests/test_api_contracts.py::test_package_version_matches_pyproject`).

> Deep subsystem detail — invariants, traps, and the reasons behind non-obvious choices — lives in
> **[`technical-docs/engineering-notes.md`](technical-docs/engineering-notes.md)**. The index at
> the bottom of this file says which section to open. Keep this file short; put new long-form
> findings there.

## Packaging layout (2026-07-24 split)

The repo is a **uv workspace** with a lean core and optional layers:

- **`mmm-framework`** (`src/mmm_framework/`) — business logic only. `pip install mmm-framework`
  gives a notebook user fit/analyze/report/validate with **no web or LLM deps**.
  `tests/test_lean_imports.py` pins this — never add a module-level fastapi/langchain/langgraph/
  httpx import to core modules.
- **`mmm-framework[agents]`** — the LangGraph oracle stack (langgraph, langchain-\*, httpx, pypdf,
  python-docx). `agents/__init__.py` is lazy (PEP 562), so the langchain-free service modules
  (`fitting`, `workspace`, `tables`, `estimand_rows`, `spec_locks`, `spec_normalize`, `model_ops`,
  `report_builder`, `branding`) import without the extra.
- **`mmm-framework-server`** (`server/src/mmm_framework_server/`) — the FastAPI app (`main.py`).
  The old `mmm_framework.api` service modules now live in **`mmm_framework/platform/`** (sessions
  store, history, runs, pacing, scorecard, triangulation, backfill, backup). Workspace member; dev
  sync installs it; deploys use `uv sync --frozen --no-dev --package mmm-framework-server`.

See `technical-docs/packaging.md`.

## Quick Commands

```bash
# Install (dev default = core + [agents] + the server package)
uv sync --group dev --group app
uv sync --no-dev                 # lean core only (what pip users get)

# Tests
make tests / make fast_tests / make slow_tests    # coverage, parallel (-n logical)
uv run pytest tests/test_model.py -v
uv run pytest tests/ -m 'not slow'

# Format / lint — BOTH are CI gates
make format        # Black (src, server/src, tests, examples) — writes
make format_check  # Black --check --diff
make lint          # Ruff check
make hooks         # install the pre-commit hook that runs both

# Docs
uv run --group docs sphinx-build -b html docs/api/source docs/api/build/html   # API reference
cd docs && python3 tools/build_search_index.py && python3 tools/build_seo.py   # static site

# Run the app (agent API runs fits in-kernel — no Redis or external worker)
uv run uvicorn mmm_framework_server.main:app --host 0.0.0.0 --port 8000 --reload
cd frontend && npm run dev       # Vite :5173, proxies /api → :8000

# Example
uv run python examples/ex_model_workflow.py
```

Build-system details worth reading before touching them: *Black formatting gate*, *Sphinx / Read
the Docs API reference*, *Static docs site*, *Docs test gates* in `engineering-notes.md`.

## Release checklist (do NOT skip the docs site)

`docs/changelog.html` is hand-authored, nothing generates it from `CHANGELOG.md`, and no test
gates it — so it rots silently (1.1.0 and 1.2.0 both shipped while the site still announced
1.0.0). Cutting `vX.Y.Z`:

1. Bump `version` in `pyproject.toml` **and `__version__` in `src/mmm_framework/__init__.py`**.
   (`server/pyproject.toml` versions independently and has not tracked core since 1.2.0.)
2. Write the release section in `CHANGELOG.md` (source of truth, full detail).
3. **Mirror it into `docs/changelog.html`** — a summarised `<div class="release">` at the top of
   `#releases`, **move the `<span class="release-tag">Current</span>` chip** onto the new release,
   and update the version in the `.lead` paragraph and the `==X.Y.Z` pin example.
4. Sweep other pages stating a version or pin: `getting-started.html`, `faq.html`,
   `evaluator.html`, `troubleshooting.html` (its `__version__` expectation), `about.html`
   (citation), `api-contracts.html`. Find them with
   `grep -rn '==1\.[0-9]*\.[0-9]*\|version 1\.' docs/*.html`.
5. From `docs/`: `build_search_index.py` then `build_seo.py`; commit `shared/seo-manifest.json`
   with the regenerated `shared/*.json`.
6. Tag and push — `.github/workflows/release.yml` builds, publishes to PyPI (trusted publishing)
   and cuts the GitHub release.

## Configure the agent LLM (Vertex AI / ADC / API keys)

The LangGraph agent's LLM comes from a config file, not hard-coded. See
`docs/model-configuration.md` and `src/mmm_framework/agents/llm.py`.

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
src/mmm_framework/          # Core modeling library (no web/LLM deps)
├── model/                  # BayesianMMM (base.py), MMMResults (results.py), trend components
├── config/                 # Pydantic configs + enums (PriorConfig, AdstockConfig, MediaChannelConfig)
├── builders/               # Fluent builders (base, mff, model, prior, variable)
├── transforms/             # adstock, saturation, seasonality, trend
├── data_loader.py          # MFFLoader (Master Flat File)
├── data_preparation.py     # scaling / ScalingParameters
├── analysis.py             # counterfactual + marginal analysis (MMMAnalyzer)
├── serialization.py        # MMMSerializer (core + extended flavors)
├── jobs.py                 # async job management (multiprocessing)
├── utils/                  # standardization, statistics, arviz_compat
├── synth/                  # synthetic DGP worlds w/ causal ground truth (dgp, dgp_geo, mff)
├── dag_model_builder/      # DAG-driven model configuration (builder, dag_spec, node_configs, …)
├── mmm_extensions/         # NestedMMM, MultivariateMMM, CombinedMMM, StructuralNestedMMM
├── estimands/              # declarative estimand registry (spec, evaluate, graph, registry)
├── diagnostics/            # sbc, coverage, identification, bias_sensitivity, learning, snapshot
├── validation/             # validator, backtest, spec_curve, refutation, calibration checks
├── calibration/            # experiment likelihoods
├── planning/               # EIG/EVOI, experiment design, economics, optimizer, history
├── frequentist/            # ridge / cvxpy estimation (design, search, bootstrap)
├── continuous_learning/    # model-free geo response-surface bandit (NumPyro/JAX)
├── garden/                 # Model Garden contract + compat suite (CustomMMM)
├── finance/                # margin/valuation basis behind every dollar recommendation
├── ltv/                    # LTV/CLV preprocessing + likelihood
├── estimators/             # causal estimators
├── datasets/               # bundled example data (load_example)
├── data_studio/            # upload → EDA → clean → dataset pipeline
├── eda/                    # pre-fit data quality / profiling
├── excel_config/           # Excel-based model configuration (parser, generator, heuristics)
├── integrations/           # ad-platform / BigQuery connectors → pandas
├── storage/                # object store abstraction
├── security/               # encryption + PII helpers for the hosted posture
├── reporting/              # MMMReportGenerator + charts/ extractors/ helpers/ interactive/ prefit
├── agents/                 # LangGraph oracle: graph, tools, model_ops, fitting, kernels, workspace
├── platform/               # sessions store + history/runs/pacing/scorecard services
└── auth/                   # org/user auth core (stdlib JWT)

server/src/mmm_framework_server/main.py   # FastAPI app (separate workspace package)
frontend/src/                             # React/TS: api/ components/ pages/ theme/ stores/
examples/  tests/  docs/  technical-docs/  nbs/  scripts/
```

Frontend notes: pages mirror the measurement loop (Program / Experiments / Performance / Agent /
Login). Design tokens live in `theme/tokens.css` (`@theme`, Tailwind 4 — `tailwind.config.js` is
INERT).

## Code Style & Patterns

- **Black** formatter, **Ruff** linter (both gated in CI), type hints required (`py.typed`),
  Google-style docstrings, PascalCase classes / snake_case functions / UPPER_SNAKE constants.
- **Builder pattern** for config (`ModelConfigBuilder().with_kpi("sales").build()`), factory
  functions for presets, Pydantic for validation, lazy imports to keep PyMC off the import path.
- **DAG-based configuration**: frontend-driven model specification via `dag_model_builder/`.

## Core Technologies

PyMC 6.0+ (lockstep with PyTensor 3.x + ArviZ 1.x — ArviZ containers are xarray DataTrees, so
route arviz calls through `utils/arviz_compat.py`), NumPyro 0.19+, NutPie 0.16.10+, Numba 0.63+,
Pydantic 2.12+, Plotly 6.5+, FastAPI 0.124+ (fits run in-kernel, no external queue), React +
TypeScript (Vite, Tailwind, Zustand), Redis 7.1+ (optional rate-limit backend), Sphinx.

## Testing

Marker `@pytest.mark.slow` for long-running (model-fitting) tests. Layout: `tests/` core,
`tests/reporting/`, `tests/mmm_extensions/`, `tests/frequentist/`, `tests/data_studio/`.

Docs are gated too — `test_docs_snippets.py` (Python snippets in `docs/*.html`, `README.md`,
`CLAUDE.md`, `technical-docs/*.md` must reference real APIs), `test_docs_nav_registration.py`,
`test_docs_seo_build.py`, `test_blog_style.py`. See *Docs test gates* in `engineering-notes.md`.

## Writing a blog post

**Read [`technical-docs/writing-guide.md`](technical-docs/writing-guide.md) before writing or
editing anything in `docs/blog*.html`.** It is the prose spec for the research blog: no em dashes,
semicolons at roughly one per thousand words, no negation pivots or participial tails or summary
closers, deliberately varied sentence rhythm. Then check your work:

```bash
python3 docs/tools/check_blog_style.py docs/blog-your-post.html   # gated by tests/test_blog_style.py
python3 docs/tools/check_blog_style.py --advisory                 # + smells needing human judgment
python3 docs/tools/check_blog_style.py --against HEAD             # rewriting an existing post
```

The repo-specific half — what the guide means for these HTML pages, and the traps (em dashes hide
as `&mdash;`; `build_seo.py` can reinject them; removing a negation pivot loses information; index
reading times rot silently) — is in
[`technical-docs/blog-style-enforcement.md`](technical-docs/blog-style-enforcement.md).

## Common Development Tasks

| Task | Steps |
|------|-------|
| Feature on BayesianMMM | `model/base.py` → `model/results.py` → `config/` → builder in `builders/model.py` → `tests/test_model.py` |
| New transform | `transforms/<x>.py` → export in `transforms/__init__.py` → `tests/test_transforms.py` |
| Report section | chart in `reporting/charts/` → extractor in `reporting/extractors/` → section in `reporting/sections.py` → helper in `reporting/helpers/` → `reporting/config.py` |
| Extended model | `mmm_extensions/config.py` → `builders.py` → `components/` → `models/` → `results.py` |
| API endpoint | route in `server/src/mmm_framework_server/main.py`; web-free logic in `src/mmm_framework/platform/`; Pydantic schemas beside the route; `tests/test_*_endpoint*.py` |
| Frontend page | `frontend/src/pages/` → hooks in `api/hooks/` → store in `stores/` → route in `App.tsx` |

## API Usage Examples

```python
# Data loading — the loader is fluent; build_panel() yields the PanelDataset
# every model takes. The KPI and channel roles come from the MFFConfig.
from mmm_framework import MFFLoader
panel = MFFLoader(mff_config).load("data.csv").build_panel()

# Basic model fitting — BayesianMMM takes (panel, model_config, trend_config)
from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig, TrendType

model_config = (
    ModelConfigBuilder()
    .bayesian_numpyro()      # NUTS backend: .bayesian_pymc()/.bayesian_numpyro()/.bayesian_nutpie()
    .with_chains(4)
    .with_draws(2000)
    .with_tune(1000)
    .build()
)
model = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
results = model.fit(random_seed=42)

# Report generation — the generator takes the model/results, then writes a file
from mmm_framework.reporting import MMMReportGenerator, ReportConfig
generator = MMMReportGenerator(model=model, results=results, config=ReportConfig())
path = generator.to_html("report.html")

# Model serialization — static methods; load() returns the MODEL only, and
# needs the panel back for a core (non-extended) save
from mmm_framework.serialization import MMMSerializer
MMMSerializer.save(model, "models/national_mmm")
model = MMMSerializer.load("models/national_mmm", panel)

# Analysis
from mmm_framework.analysis import MMMAnalyzer
analyzer = MMMAnalyzer(model)
marginals = analyzer.compute_marginal_contributions()   # list[MarginalAnalysisResult]
roi = analyzer.compute_channel_roi()
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model fitting slow | Pick the NUTS backend on the config: `.bayesian_numpyro()` (default) / `.bayesian_pymc()` / `.bayesian_nutpie()`, or `fit(nuts_sampler="nutpie")` for one fit. For a seconds-long check use `fit(method="map"/"advi"/"pathfinder"/"laplace")` — approximate, uncertainty NOT calibrated |
| Fit fails / diverges / won't sample | Escalation ladder in `technical-docs/sampling-failure-playbook.md` |
| `ZeroDivisionError` from a MAP/Laplace fit | Weak identification, not a numerical accident — the fit is auto-retried inside a safe box and emits a report. See the notes entry |
| 90% interval covers ~50% of the truth | `diagnostics/coverage.py`; top cause is an approximate fit. `technical-docs/coverage-diagnostics.md` |
| Tests hanging | `make fast_tests` to skip slow tests |
| Memory overflow | `fit(draws=1000, tune=500, chains=2)` |
| Import errors | `uv sync --group dev` |
| Frontend not starting | `cd frontend && npm install` first |
| Rate limiting errors | `src/mmm_framework/auth/ratelimit.py` (env `MMM_RATELIMIT_*`; optional redis backend) |
| Serialization errors | Ensure cloudpickle versions match across environments |
| DAG validation fails | `dag_model_builder/validation.py` |
| Agent LLM auth / wrong provider | `config/model_config.yaml` or `MMM_LLM_*`; on GCP, Vertex uses ADC — grant the VM SA `roles/aiplatform.user` |
| Vertex "model not found" / 404 | Use the exact Model Garden id (may carry `@version`) and a `location` that serves it (Claude: e.g. `us-east5`) |
| Local model (LM Studio) | `provider: lmstudio`, `model: <id>`, `base_url: http://localhost:1234/v1`; no API key; needs a tool-capable model, plus an embedding model + `MMM_EMBED_MODEL` for the KB |
| KB ingest "no embedding backend" | Chat LLM ≠ embedder (Anthropic has no embeddings). With a `vertex_*` chat provider, KB uses Vertex `text-embedding-005` over ADC — run `gcloud auth application-default login`. Override with `MMM_EMBED_PROVIDER`/`_MODEL`/`_LOCATION` |
| Where did the agent write my file? | `$MMM_AGENT_WORKSPACE/threads/<thread_id>/` (default `./agent_workspace`); project KB at `projects/<project_id>/kb/` |
| Sessions DB location | Defaults to `src/mmm_framework/platform/sessions.db`; relocate with `MMM_SESSIONS_DB`. Seeders/backfill/backup CLI must use the SAME value as the server |
| PyTensor won't compile (macOS) | Pin `cxx=/usr/bin/clang++` in `~/.pytensorrc` — conda-base clang breaks the linker |

## Subsystem index → `technical-docs/engineering-notes.md`

Fitting & inference

- Approximate/fast fits (MAP, ADVI, Pathfinder, Laplace, SMC) — *Approximate / fast fit for model checking*
- Weak identification, ridges, effective parameters — *ZeroDivisionError … MAP / Laplace fit*
- Frequentist ridge / cvxpy, and when to prefer it — *Frequentist estimation* · `technical-docs/frequentist-estimation.md`
- arviz/pymc version-drift shims — *arviz/pymc version-drift shims*
- Interval coverage & SBC coverage — *Interval coverage* · `technical-docs/coverage-diagnostics.md`
- Confounding sensitivity / tipping points — *"What if we missed a confounder?"* · `technical-docs/confounding-sensitivity.md`

Model families

- Extension models (Nested / Multivariate / Combined): priors, trend, seasonality, likelihood — *Extension-model priors …* · `technical-docs/extension-model-priors.md`
- Extension `predict()` and counterfactuals — *Extension-model `predict()`*
- StructuralNestedMMM (survey mediators, AR(1) states, latent factors) — *StructuralNestedMMM* · `technical-docs/structural-nested-mmm.md`
- Per-model config + pluggable likelihood — *Per-model config …* · `technical-docs/custom-model-config.md`
- Non-MMM families (CFA / LCA) — *Non-MMM model families*, *Second non-MMM family* · `technical-docs/non-mmm-families.md`
- Latent contrasts + joint latent-factor MMM — *Latent-variable contrasts*, *Joint latent-factor MMM*
- Declarative estimands — *Declarative estimands* · `technical-docs/estimands.md`
- Impression/click-measured media (ROI vs efficiency) — *Impression-/click-measured media* · `technical-docs/impression-level-roi.md`
- ROI-parameterized default media priors — *ROI-based default media priors*
- Priors that actually reach the graph — *Agent-set priors don't change the model*

Experiments & planning

- The T₀–T₅ measurement loop, lifecycle registry — *Measurement loop / experiments*
- Model-anchored economics, opportunity cost, A/A·A/B sim — *Model-anchored experiment economics* · `technical-docs/experiment-economics.md`
- Off-panel calibration — *Off-panel calibration*
- Pareto-front experiment optimizer — *Experiment optimizer*
- Continuous sequential learning (model-free bandit) — *Continuous sequential learning* · `technical-docs/continuous-learning.md`

Reporting

- Estimand CIs + posterior-predictive GoF sections — *Estimand results (CI) …*
- Interactive recompute-in-browser results report — *Interactive MMM Results Report*
- Pre-fit Model Design Readout — *Pre-fit Model Design Readout* · `technical-docs/prefit-model-design-readout.md`

Agent / platform / UI

- Per-turn workflow-step guard — *Agent auto-runs the whole 9-step workflow*
- Kernel modes + hosted posture — *`execute_python` kernel mode*, *Hosted multi-user profile* · `technical-docs/agent-session-kernels.md`
- Session artifacts, tables, branding, guide chat — *Agent reports = session artifacts*, *Agent tables*, *Client branding*, *Guide chat*
- Data Studio (upload → EDA → clean → dataset) — *Data Studio*
- Atelier / Model Garden + notebook — *Atelier notebook*, *Atelier / Model Garden docs + demo*
- Oracle model-settings visibility, load restores spec — *Oracle model-settings visibility*
- LaTeX/math rendering in every markdown surface — *LaTeX / math in chat + copilot + docs*

Writing

- Research-blog prose spec (read before writing a post) — `technical-docs/writing-guide.md`
- What it means for `docs/blog*.html`, plus the traps and the checker — `technical-docs/blog-style-enforcement.md`
