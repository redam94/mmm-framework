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
make tests              # All tests with coverage
make fast_tests         # Skip slow tests
make slow_tests         # Slow tests only

# Format code
make format             # Black formatting

# Start the application (requires 3 terminals)
redis-server                              # Terminal 1: Start Redis
cd api && uvicorn main:app --reload       # Terminal 2: FastAPI backend (port 8000)
cd api && arq worker.WorkerSettings       # Terminal 3: ARQ worker
cd app && streamlit run Home.py           # Terminal 4: Streamlit UI (port 8501)

# Run example
uv run python examples/ex_model_workflow.py
```

## Directory Structure

```
mmm-framework/
├── src/mmm_framework/          # Core modeling library
│   ├── config.py               # Configuration enums and Pydantic dataclasses
│   ├── builders.py             # Fluent configuration builders
│   ├── data_loader.py          # MFF parsing, validation, loading
│   ├── model.py                # BayesianMMM model implementation
│   ├── jobs.py                 # Async job management (ARQ)
│   ├── mmm_extensions/         # Advanced: NestedMMM, MultivariateMMM, CombinedMMM
│   └── reporting/              # HTML report generation
├── api/                        # FastAPI backend
│   ├── main.py                 # App factory
│   ├── worker.py               # ARQ worker for async jobs
│   └── routes/                 # API endpoints
├── app/                        # Streamlit frontend
│   ├── Home.py                 # Main entry point
│   └── pages/                  # UI pages
├── examples/                   # Working usage examples
├── tests/                      # Test suite
├── technical-docs/             # Mathematical specifications
└── nbs/                        # Jupyter notebooks
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `model.py` | BayesianMMM class - saturation, adstock, hierarchical modeling |
| `config.py` | Pydantic configs: PriorConfig, AdstockConfig, MediaChannelConfig |
| `builders.py` | Fluent builders: ModelConfigBuilder, MediaChannelConfigBuilder |
| `data_loader.py` | MFFLoader for Master Flat File format data |
| `mmm_extensions/` | NestedMMM (mediation), MultivariateMMM (multi-outcome) |
| `reporting/` | MMMReportGenerator, charts.py (Plotly), helpers.py (ROI analysis) |

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

## Core Technologies

- **PyMC 5.26+** - Bayesian inference
- **NumPyro/NutPie** - Fast NUTS sampling (4-10x speedup)
- **FastAPI** - REST API
- **Streamlit** - Web UI
- **Redis + ARQ** - Async job queue
- **Plotly** - Interactive visualizations

## Testing

```bash
# Run specific test file
uv run pytest tests/test_model.py -v

# Run with coverage report
uv run pytest tests/ --cov=mmm_framework --cov-report=html

# Skip slow tests (useful during development)
uv run pytest tests/ -m 'not slow'
```

Test markers:
- `@pytest.mark.slow` - Long-running tests (model fitting)

## Common Development Tasks

### Add feature to BayesianMMM
1. Edit `src/mmm_framework/model.py`
2. Update config in `src/mmm_framework/config.py`
3. Add builder method in `src/mmm_framework/builders.py`
4. Write tests in `tests/test_model.py`

### Add report section
1. Chart function in `reporting/charts.py`
2. Section builder in `reporting/sections.py`
3. Helper in `reporting/helpers.py`
4. Update `reporting/config.py`

### Add extended model
1. Config in `mmm_extensions/config.py`
2. Builder in `mmm_extensions/builders.py`
3. Components in `mmm_extensions/components.py`
4. Model class in `mmm_extensions/models.py`

## API Usage Examples

```python
# Basic model fitting
from mmm_framework import BayesianMMM, ModelConfigBuilder

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
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection fails | Run `redis-server` in separate terminal |
| Worker not processing | Start ARQ: `cd api && arq worker.WorkerSettings` |
| Model fitting slow | Use numpyro sampler: `model.fit(nuts_sampler="numpyro")` |
| Tests hanging | Run `make fast_tests` to skip slow tests |
| Memory overflow | Reduce draws/chains: `fit(draws=1000, tune=500, chains=2)` |
