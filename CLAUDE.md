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

# Build documentation
uv run sphinx-build -b html docs/ docs/_build/html

# Start the application

## Option 1: Streamlit UI (legacy)
redis-server                              # Terminal 1: Start Redis
cd api && uvicorn main:app --reload       # Terminal 2: FastAPI backend (port 8000)
cd api && arq worker.WorkerSettings       # Terminal 3: ARQ worker
cd app && streamlit run Home.py           # Terminal 4: Streamlit UI (port 8501)

## Option 2: React UI (modern)
redis-server                              # Terminal 1: Start Redis
cd api && uvicorn main:app --reload       # Terminal 2: FastAPI backend (port 8000)
cd api && arq worker.WorkerSettings       # Terminal 3: ARQ worker
cd frontend && npm run dev                # Terminal 4: React UI (Vite)

# Run example
uv run python examples/ex_model_workflow.py
```

## Directory Structure

```
mmm-framework/
├── src/mmm_framework/          # Core modeling library
│   ├── config.py               # Configuration enums and Pydantic dataclasses
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
│   │   │   ├── common/         # Shared components
│   │   │   ├── layout/         # Layout components
│   │   │   └── workflow/       # Workflow components
│   │   ├── pages/              # Page components
│   │   │   ├── Dashboard/
│   │   │   ├── DataUpload/
│   │   │   ├── Diagnostics/
│   │   │   ├── Login/
│   │   │   ├── ModelConfig/
│   │   │   ├── ModelFit/
│   │   │   ├── Planning/
│   │   │   └── Results/
│   │   └── stores/             # Zustand state stores
│   ├── vite.config.ts
│   └── tailwind.config.js
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
| `config.py` | Pydantic configs: PriorConfig, AdstockConfig, MediaChannelConfig |
| `builders/` | Fluent builders: ModelConfigBuilder, MediaChannelConfigBuilder |
| `data_loader.py` | MFFLoader for Master Flat File format data |
| `analysis.py` | Counterfactual analysis, MarginalAnalysisResult |
| `data_preparation.py` | Data scaling with ScalingParameters |
| `serialization.py` | MMMSerializer for model persistence |
| `transforms/` | Adstock, saturation, seasonality, trend transforms |
| `utils/` | Standardization and statistics utilities |
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

## Common Development Tasks

### Add feature to BayesianMMM
1. Edit `src/mmm_framework/model/base.py`
2. Update results in `src/mmm_framework/model/results.py`
3. Update config in `src/mmm_framework/config.py`
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
| Tests hanging | Run `make fast_tests` to skip slow tests |
| Memory overflow | Reduce draws/chains: `fit(draws=1000, tune=500, chains=2)` |
| Import errors | Run `uv sync --group dev` to install dependencies |
| Frontend not starting | Run `cd frontend && npm install` first |
| Rate limiting errors | Check `api/rate_limiter.py` configuration |
| Serialization errors | Ensure cloudpickle version matches across environments |
| DAG validation fails | Check `dag_model_builder/validation.py` for requirements |
