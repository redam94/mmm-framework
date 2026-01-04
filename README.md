# MMM Framework

A modular Marketing Mix Model framework built on PyMC-Marketing with full Bayesian uncertainty quantification, async model fitting, and interactive visualization.

## Overview

This framework provides a production-ready implementation for marketing mix modeling that emphasizes methodological rigor over specification shopping. It handles variable-dimension MFF (Master Flat File) data, multiplicative specifications with Hill/logistic saturation, hierarchical panel structures, and offers fast frequentist alternatives for rapid iteration.

### Why This Framework?

Traditional MMM practices often involve iterating on specifications until results "look right"—adjusting lags, decay rates, and controls until coefficients achieve desired properties. This approach inflates false positive rates, biases coefficients, and generates confidence intervals that do not reflect actual uncertainty.

This framework is designed around different principles:

- **Pre-specified analyses** reduce researcher degrees of freedom
- **Bayesian inference** provides genuine uncertainty quantification through posterior distributions
- **Hierarchical modeling** enables partial pooling across geographies and products
- **Experimental validation** where model predictions can be tested against holdout results

## Features

### Core Modeling

- **Bayesian MMM Engine** — Full PyMC implementation with proper uncertainty quantification
- **Flexible Saturation Functions** — Logistic and Hill saturation with interpretable parameterization
- **Geometric Adstock** — Configurable carryover effects with multiple decay structures
- **Hierarchical Effects** — Partial pooling across geographies, products, or other dimensions
- **Trend Modeling** — Linear, piecewise, B-spline, and Gaussian Process trend options
- **Fourier Seasonality** — Configurable seasonal harmonics

### Advanced Capabilities (mmm_extensions)

- **Nested/Mediated Models** — Causal pathways through intermediate outcomes (Media → Awareness → Sales)
- **Multivariate Outcomes** — Joint modeling of multiple KPIs with correlated errors
- **Cross-Product Effects** — Cannibalization, halo effects, and spillovers between products
- **Partial Observation** — Handle sparse mediator data (e.g., monthly surveys in weekly model)
- **Effect Decomposition** — Separate direct vs. indirect effects with uncertainty
- **Flexible Priors** — Constrained effects (positive/negative) with configurable priors

### Inference & Analysis

- **Counterfactual Contributions** — Proper contribution analysis with uncertainty bands
- **Scenario Planning** — Budget reallocation simulations with posterior predictive checks
- **Prior/Posterior Comparison** — Visualize how data updates beliefs
- **Component Decomposition** — Break down predictions into trend, seasonality, media, controls

### Infrastructure

- **FastAPI Backend** — RESTful API with OpenAPI documentation
- **Async Job Processing** — Redis + ARQ for non-blocking model fitting
- **Streamlit Frontend** — Interactive dashboards for configuration, fitting, and analysis
- **LangGraph Integration** — AI-assisted model interpretation with multiple LLM providers

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                           │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│   │   Data   │ │  Config  │ │  Model   │ │ Results  │ │   Chat   │ │
│   │  Upload  │ │  Builder │ │  Fitting │ │  Viewer  │ │Interface │ │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────────┐
│                         FastAPI Backend                              │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐   │
│   │  /data/*     │ │  /configs/*  │ │  /models/*               │   │
│   │  Upload/List │ │  CRUD        │ │  Fit/Status/Results      │   │
│   └──────────────┘ └──────────────┘ └──────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────────┐
    │  Redis   │◄────►│   ARQ    │─────►│  PyMC Model  │
    │  Queue   │      │  Worker  │      │   Fitting    │
    └──────────┘      └──────────┘      └──────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- Redis server
- uv (recommended) or pip

### Quick Install

```bash
# Clone the repository
git clone https://github.com/redam94/mmm-framework.git
cd mmm-framework

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Install app dependencies for Streamlit frontend
uv sync --group app
```

### Development Install

```bash
# Install all dependencies including dev tools
uv sync --group dev --group app
```

## Quick Start

### 1. Start Redis

```bash
redis-server
```

### 2. Start the API Server

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the ARQ Worker

```bash
cd api
arq worker.WorkerSettings
```

### 4. Launch the Streamlit App

```bash
cd app
streamlit run Home.py
```

### 5. Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Usage

### Python API

```python
from mmm_framework import (
    MFFLoader,
    ModelConfigBuilder,
    MediaChannelConfigBuilder,
    BayesianMMM,
)

# Load data in MFF format
loader = MFFLoader(config=mff_config)
panel_data = loader.load("data.csv")

# Build model configuration
config = (
    ModelConfigBuilder()
    .with_kpi("sales", log_transform=True)
    .with_media_channel(
        MediaChannelConfigBuilder()
        .with_name("tv")
        .with_adstock(alpha_prior=(1, 3), l_max=8)
        .with_saturation(lam_prior=(1, 2))
        .build()
    )
    .with_seasonality(yearly=True, n_fourier=2)
    .with_trend(trend_type="linear")
    .build()
)

# Fit the model
model = BayesianMMM(
    X_media=panel_data.media,
    y=panel_data.kpi,
    channel_names=panel_data.channel_names,
    config=config,
)

results = model.fit(
    draws=2000,
    tune=1000,
    chains=4,
    nuts_sampler="numpyro",  # 4-10x faster than CPU PyMC
)

# Get contributions with uncertainty
contributions = model.compute_contributions()
print(contributions.mean_contributions)
print(contributions.hdi_contributions)  # 94% credible intervals
```

### Fluent Configuration API

The framework provides fluent builders for all configuration objects:

```python
from mmm_framework import (
    ModelConfigBuilder,
    MediaChannelConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    HierarchicalConfigBuilder,
)

# Build a hierarchical model configuration
config = (
    ModelConfigBuilder()
    .with_kpi("transactions")
    .with_hierarchical(
        HierarchicalConfigBuilder()
        .with_geo_dimension("dma")
        .with_partial_pooling(True)
        .build()
    )
    .with_media_channel(
        MediaChannelConfigBuilder()
        .with_name("digital")
        .with_adstock(
            AdstockConfigBuilder()
            .with_type("geometric")
            .with_alpha_prior("Beta", alpha=1, beta=3)
            .with_l_max(4)
            .build()
        )
        .with_saturation(
            SaturationConfigBuilder()
            .with_type("logistic")
            .with_lam_prior("Gamma", alpha=2, beta=1)
            .build()
        )
        .build()
    )
    .build()
)
```

### Extended Models

For complex scenarios with mediated effects or multiple outcomes:

```python
from mmm_framework.mmm_extensions import (
    CombinedModelConfigBuilder,
    awareness_mediator,
    cannibalization_effect,
    CombinedMMM,
)

# Build a model with awareness mediation and product cannibalization
config = (
    CombinedModelConfigBuilder()
    .with_mediator(awareness_mediator(decay=0.9))
    .with_outcomes("single_pack", "multipack")
    .with_cross_effect(
        cannibalization_effect("multipack", "single_pack", promo_col="multi_promo")
    )
    .build()
)

model = CombinedMMM(
    X_media=X_media,
    outcome_data={"single_pack": y1, "multipack": y2},
    channel_names=channels,
    config=config,
    mediator_data={"awareness": survey_data},
)
```

## MMM Extensions Module

The `mmm_extensions` subpackage provides advanced modeling capabilities for scenarios beyond standard single-outcome MMM. It supports nested/mediated causal pathways, multivariate outcomes with cross-effects, and combined models that handle both.

### Module Architecture

```
mmm_framework/mmm_extensions/
├── config.py       # Dataclasses for all configuration objects
├── builders.py     # Fluent builders and factory functions
├── components.py   # PyMC/PyTensor building blocks (lazy-loaded)
└── models.py       # NestedMMM, MultivariateMMM, CombinedMMM classes
```

### Nested/Mediated Models

Nested models capture causal pathways where media affects intermediate outcomes (mediators) which in turn drive final outcomes:

```
Media → Awareness → Sales
     ↘____________↗
       (direct effect)
```

This decomposes the total media effect into:
- **Direct effect**: Media → Sales (bypassing mediator)
- **Indirect effect**: Media → Awareness → Sales
- **Total effect**: Direct + Indirect

```python
from mmm_framework.mmm_extensions import (
    MediatorConfigBuilder,
    NestedModelConfigBuilder,
    NestedMMM,
    awareness_mediator,
)

# Method 1: Factory function for common configurations
awareness = awareness_mediator(
    name="brand_awareness",
    observation_noise=0.15,  # Survey measurement error
)

# Method 2: Builder for full control
awareness_custom = (
    MediatorConfigBuilder("brand_awareness")
    .partially_observed(observation_noise=0.15)
    .with_positive_media_effect(sigma=1.0)
    .with_slow_adstock(l_max=12)  # Brand metrics have long carryover
    .with_direct_effect(sigma=0.3)  # Allow some direct effect
    .build()
)

# Build model configuration
nested_config = (
    NestedModelConfigBuilder()
    .add_mediator(awareness_custom)
    .map_channels_to_mediator(
        "brand_awareness",
        ["tv", "digital"]  # Only these channels build awareness
    )
    .share_adstock(True)
    .build()
)

# Fit the model
model = NestedMMM(
    X_media=X_media,
    y=sales,
    channel_names=["tv", "digital", "social", "search"],
    config=nested_config,
    mediator_data={"brand_awareness": survey_data},  # Sparse observations OK
)

results = model.fit(draws=2000, tune=1000)

# Decompose effects
mediation_effects = model.get_mediation_effects()
for channel_effect in mediation_effects:
    print(f"{channel_effect.channel}:")
    print(f"  Direct: {channel_effect.direct_effect:.3f}")
    print(f"  Indirect via awareness: {channel_effect.indirect_effects['brand_awareness']:.3f}")
    print(f"  Proportion mediated: {channel_effect.proportion_mediated:.1%}")
```

#### Mediator Types

| Type | Description | Use Case |
|------|-------------|----------|
| `FULLY_OBSERVED` | Complete time series available | Foot traffic, web visits |
| `PARTIALLY_OBSERVED` | Sparse observations (surveys) | Brand awareness, consideration |
| `FULLY_LATENT` | No direct observations | Latent brand equity |

### Multivariate Outcome Models

Multivariate models jointly estimate effects across multiple outcomes, capturing correlations and cross-product effects:

```python
from mmm_framework.mmm_extensions import (
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    MultivariateModelConfigBuilder,
    MultivariateMMM,
    cannibalization_effect,
    halo_effect,
)

# Define outcomes
single_pack = (
    OutcomeConfigBuilder("single_pack", column="sales_single")
    .with_positive_media_effects(sigma=0.5)
    .include_trend()
    .include_seasonality()
    .build()
)

multipack = (
    OutcomeConfigBuilder("multipack", column="sales_multi")
    .with_positive_media_effects(sigma=0.5)
    .include_trend()
    .include_seasonality()
    .build()
)

# Define cross-effects
# Cannibalization: multipack promotions steal from single-pack
cannib = (
    CrossEffectConfigBuilder("multipack", "single_pack")
    .cannibalization()
    .modulated_by_promotion("multipack_promo")
    .lagged()  # Effect appears next period
    .with_prior_sigma(0.3)
    .build()
)

# Or use factory function
cannib_simple = cannibalization_effect(
    source="multipack",
    target="single_pack",
    promotion_column="multipack_promo",
    lagged=True,
)

# Build configuration
mv_config = (
    MultivariateModelConfigBuilder()
    .add_outcome(single_pack)
    .add_outcome(multipack)
    .add_cross_effect(cannib)
    .with_lkj_eta(2.0)  # Prior on correlation structure
    .share_media_adstock(True)
    .share_media_saturation(False)  # Different saturation per product
    .share_seasonality(True)
    .build()
)

# Fit model
model = MultivariateMMM(
    X_media=X_media,
    outcome_data={"single_pack": y_single, "multipack": y_multi},
    channel_names=channels,
    config=mv_config,
    promotion_data={"multipack_promo": promo_indicator},
)

results = model.fit()

# Analyze cross-effects
cross_effects = model.get_cross_effect_summary()
print(cross_effects)

# Get correlation matrix
corr = model.get_correlation_matrix()
```

#### Cross-Effect Types

| Type | Description | Prior Constraint |
|------|-------------|------------------|
| `CANNIBALIZATION` | Source steals from target | Negative effect |
| `HALO` | Source lifts target | Positive effect |
| `SPILLOVER` | Bidirectional relationship | Unconstrained |

### Combined Models

For the most complex scenarios, `CombinedMMM` supports both nested pathways and multivariate outcomes:

```python
from mmm_framework.mmm_extensions import (
    CombinedModelConfigBuilder,
    CombinedMMM,
)

# Full c-store scenario:
# - Media builds awareness (nested)
# - Awareness drives both product sales
# - Multipack promotions cannibalize single-pack (cross-effect)
# - Correlated errors across products

config = (
    CombinedModelConfigBuilder()
    # Nested component
    .with_awareness_mediator("brand_awareness", observation_noise=0.15)
    .map_channels_to_mediator("brand_awareness", ["tv", "digital"])
    # Multivariate component
    .with_outcomes("single_pack", "multipack")
    .with_cannibalization("multipack", "single_pack", promotion_column="multi_promo")
    # Link mediator to outcomes
    .map_mediator_to_outcomes("brand_awareness", ["single_pack", "multipack"])
    .with_lkj_eta(2.0)
    .build()
)

model = CombinedMMM(
    X_media=X_media,
    outcome_data={"single_pack": y1, "multipack": y2},
    channel_names=["tv", "digital", "social", "search"],
    config=config,
    mediator_data={"brand_awareness": survey_data},
    promotion_data={"multi_promo": promo_flags},
)

results = model.fit(draws=2000, tune=1000, chains=4)
```

### Factory Functions

Common configurations are available as factory functions:

```python
from mmm_framework.mmm_extensions import (
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
)

# Awareness mediator with slow decay and partial observation
awareness = awareness_mediator(
    name="brand_awareness",
    observation_noise=0.15,
)

# Foot traffic mediator with full observation
traffic = foot_traffic_mediator(
    name="store_visits",
    observation_noise=0.05,
)

# Cross-effects
cannib = cannibalization_effect("product_b", "product_a", promotion_column="b_promo")
halo = halo_effect("premium", "value")  # Premium brand lifts value brand
```

### Results and Diagnostics

All extended models provide structured result containers:

```python
# Mediation decomposition
effects = model.get_mediation_effects()
for e in effects:
    print(e.to_dict())

# Cross-effect summary with HDI
cross_df = model.get_cross_effect_summary()
# Returns: source, target, effect_type, mean, sd, hdi_3%, hdi_97%

# Correlation matrix for multivariate outcomes
corr_matrix = model.get_correlation_matrix()

# Standard ArviZ diagnostics
results.summary(var_names=["beta_media", "alpha"])
results.plot_trace(var_names=["beta_media"])
```

## Variable Selection for Control Variables

The `mmm_extensions` module includes Bayesian variable selection priors for precision control variables. These methods provide principled shrinkage and selection, improving precision when many potential controls exist but only a few are truly relevant.

> ⚠️ **CAUSAL WARNING**: Variable selection should ONLY be applied to **precision control variables**—variables that affect the outcome but do NOT affect treatment assignment (media spending). **Confounders** (variables affecting both media and sales) must be EXCLUDED from selection and always included with standard priors. Shrinking a confounder toward zero does not remove confounding bias.

### Variable Classification

Before applying variable selection, classify each control variable:

| Variable Type | Examples | Selection OK? | Reason |
|---------------|----------|---------------|--------|
| **Precision Controls** | Weather, gas prices, minor holidays, sports events | ✅ Yes | Affect outcome only |
| **Confounders** | Distribution/ACV, price, competitor media | ❌ No | Affect both media AND outcome |
| **Core Components** | Trend, seasonality | ❌ No | Fundamental model structure |
| **Mediators** | Brand awareness (if on causal path) | ❌ No | Blocks causal effect |

### Available Methods

| Method | Best For | Key Feature |
|--------|----------|-------------|
| **Regularized Horseshoe** | Sparse signals (few relevant controls) | Strong shrinkage of noise, preserves signals |
| **Finnish Horseshoe** | Same as regularized horseshoe | Emphasizes slab regularization |
| **Spike-and-Slab** | Explicit inclusion probabilities | Direct posterior P(included) for each variable |
| **Bayesian LASSO** | Many small effects | L1-like shrinkage in Bayesian framework |

### Quick Start

```python
from mmm_framework.mmm_extensions import (
    # Builders
    VariableSelectionConfigBuilder,
    HorseshoeConfigBuilder,
    
    # Factory functions
    sparse_controls,
    selection_with_inclusion_probs,
    
    # Components
    build_control_effects_with_selection,
    summarize_variable_selection,
)

# Method 1: Factory function (simplest)
config = sparse_controls(
    expected_nonzero=3,
    "distribution", "price", "competitor_media",  # Confounders to exclude
)

# Method 2: Builder with full control
config = (VariableSelectionConfigBuilder()
    .regularized_horseshoe(expected_nonzero=3)
    .with_slab_scale(2.0)
    .exclude_confounders("distribution", "price", "competitor_media")
    .build())
```

### Configuration Classes

#### VariableSelectionConfig

The main configuration object:

```python
from mmm_framework.mmm_extensions import (
    VariableSelectionConfig,
    VariableSelectionMethod,
    HorseshoeConfig,
)

config = VariableSelectionConfig(
    method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
    horseshoe=HorseshoeConfig(
        expected_nonzero=3,      # Prior belief: ~3 controls are relevant
        slab_scale=2.0,          # Max expected effect in std units
        slab_df=4.0,             # Slab tail weight
    ),
    exclude_variables=("distribution", "price"),  # Always include these
)
```

#### HorseshoeConfig

Controls the regularized horseshoe prior:

```python
from mmm_framework.mmm_extensions import HorseshoeConfigBuilder

horseshoe = (HorseshoeConfigBuilder()
    .with_expected_nonzero(5)      # Expect 5 relevant controls
    .with_slab_scale(2.5)          # Allow effects up to 2.5 std
    .with_heavy_tails()            # slab_df=2.0 for larger effects
    .with_aggressive_shrinkage()   # Stronger shrinkage of noise
    .build())
```

#### SpikeSlabConfig

For explicit inclusion probabilities:

```python
from mmm_framework.mmm_extensions import SpikeSlabConfigBuilder

spike_slab = (SpikeSlabConfigBuilder()
    .with_prior_inclusion(0.3)     # 30% prior prob of inclusion
    .with_sharp_selection()        # Lower temperature, sharper selection
    .continuous()                  # Required for NUTS sampling
    .build())
```

### Builder API

The `VariableSelectionConfigBuilder` provides a fluent interface:

```python
from mmm_framework.mmm_extensions import VariableSelectionConfigBuilder

# Regularized horseshoe (recommended default)
config = (VariableSelectionConfigBuilder()
    .regularized_horseshoe(expected_nonzero=3)
    .with_slab_scale(2.0)
    .with_slab_df(4.0)
    .exclude_confounders("distribution", "price", "competitor_media")
    .build())

# Spike-and-slab for inclusion probabilities
config = (VariableSelectionConfigBuilder()
    .spike_slab(prior_inclusion=0.3)
    .with_sharp_selection()
    .exclude_confounders("distribution", "price")
    .apply_only_to("weather", "gas_price", "minor_holiday")  # Limit scope
    .build())

# Bayesian LASSO for many small effects
config = (VariableSelectionConfigBuilder()
    .bayesian_lasso(regularization=2.0)
    .exclude_confounders("distribution", "price")
    .build())
```

### Factory Functions

For common configurations:

```python
from mmm_framework.mmm_extensions import (
    sparse_controls,
    selection_with_inclusion_probs,
    dense_controls,
)

# Sparse: expect few relevant controls
config = sparse_controls(3, "distribution", "price")

# Inclusion probs: want P(included) for each variable
config = selection_with_inclusion_probs(0.5, "distribution", "price")

# Dense: expect many small effects
config = dense_controls(1.0, "distribution", "price")
```

### Integration with Models

Use `build_control_effects_with_selection` to handle the split between confounders and precision controls:

```python
import pymc as pm
from mmm_framework.mmm_extensions import (
    VariableSelectionConfigBuilder,
    build_control_effects_with_selection,
)

# Define variable roles
all_controls = ["distribution", "price", "weather", "gas_price", "holiday"]
confounders = ["distribution", "price"]

# Configure selection (excludes confounders automatically)
selection_config = (VariableSelectionConfigBuilder()
    .regularized_horseshoe(expected_nonzero=2)
    .exclude_confounders(*confounders)
    .build())

# Build model
with pm.Model() as model:
    sigma = pm.HalfNormal("sigma", 0.5)
    
    # This handles the split automatically:
    # - Confounders get standard Normal priors
    # - Precision controls get horseshoe priors
    control_result = build_control_effects_with_selection(
        X_controls=X_controls,
        control_names=all_controls,
        n_obs=len(y),
        sigma=sigma,
        selection_config=selection_config,
        name_prefix="ctrl",
    )
    
    # Use in likelihood
    mu = intercept + media_effect + control_result.contribution
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)
```

### Interpreting Results

After fitting, analyze variable selection:

```python
from mmm_framework.mmm_extensions import (
    compute_inclusion_probabilities,
    summarize_variable_selection,
)

# Get inclusion probabilities
inclusion = compute_inclusion_probabilities(
    trace=idata,
    config=selection_config,
    name="ctrl_select",
)
print(f"Effective nonzero: {inclusion['effective_nonzero']:.2f}")

# Full summary table
summary = summarize_variable_selection(
    trace=idata,
    control_names=precision_controls,
    config=selection_config,
    name="ctrl_select",
)
print(summary)
#    variable      mean    std   hdi_3%  hdi_97%  inclusion_prob  selected
# 0   weather     0.15   0.08     0.01     0.28           0.89      True
# 1  gas_price    0.02   0.05    -0.06     0.10           0.23     False
# 2   holiday    -0.11   0.06    -0.21    -0.02           0.85      True
```

For horseshoe priors, shrinkage factors (κ) indicate selection:
- κ ≈ 1: Strongly shrunk toward zero (excluded)
- κ ≈ 0: Preserved at estimated magnitude (included)

```python
# Access shrinkage factors directly
kappa = idata.posterior["ctrl_select_kappa"].mean(dim=["chain", "draw"]).values
for name, k in zip(precision_controls, kappa):
    status = "SHRUNK" if k > 0.5 else "PRESERVED"
    print(f"{name}: κ={k:.3f} ({status})")
```

### Mathematical Specification
This section provides the formal mathematical specification for the variable selection priors available in `mmm_extensions`. For the complete technical document with proofs and implementation details, see `variable_selection_specification.pdf`.

### Causal Constraints

Variable selection priors should **only** be applied to **precision control variables**—variables that affect outcome $Y$ but do *not* affect treatment $X$ (media spending).

**Proposition (Bias from Shrinkage on Confounder Coefficients)**: Consider a confounder $C$ affecting both media $X$ and outcome $Y$:

$$X = \delta C + \nu, \quad Y = \beta X + \gamma C + \epsilon$$

where $\beta$ is the **true causal effect** of media. If we shrink the coefficient on $C$ by factor $s \in [0,1]$ (yielding $\tilde{\gamma} = s \cdot \gamma$), the estimated media effect satisfies:

$$\hat{\beta} \xrightarrow{p} \beta + (1-s) \cdot \gamma \cdot \frac{\text{Cov}(X, C)}{\text{Var}(X)}$$

**Key implications**:
- **Complete shrinkage** ($s=0$): Full omitted variable bias
- **Partial shrinkage** ($s=0.1$, 90% shrinkage): Still leaves 90% of the bias
- **No shrinkage** ($s=1$): Bias eliminated

The bias depends on $\text{Cov}(X,C)/\text{Var}(X)$—the structural correlation between treatment and confounder. This is **invariant to the estimator**. Shrinking $\gamma$ does not shrink this correlation.

**Example**: Distribution (ACV) with $\gamma = 0.1$ (small direct effect) but $\text{Cov}(X,C)/\text{Var}(X) = 2.0$ (high correlation with media). Omitted variable bias = $0.1 \times 2.0 = 0.2$ (20% of media effect). A horseshoe prior seeing only small $\gamma = 0.1$ would shrink it, introducing nearly the full 0.2 bias into $\hat{\beta}$.

---

#### Regularized Horseshoe (Piironen & Vehtari, 2017)

The regularized horseshoe prior is:

$$\beta_j = z_j \cdot \tau \cdot \tilde{\lambda}_j$$

where:
- $z_j \sim \mathcal{N}(0, 1)$ — standardized coefficient
- $\tau \sim \text{Half-}t_{\nu_g}(\tau_0)$ — global shrinkage
- $\lambda_j \sim \text{Half-}t_{\nu_l}(1)$ — local shrinkage
- $\tilde{\lambda}_j = \frac{c \cdot \lambda_j}{\sqrt{c^2 + \tau^2 \lambda_j^2}}$ — regularized local shrinkage
- $c^2 \sim \text{Inv-Gamma}(\nu_s/2, \nu_s s^2/2)$ — slab regularization

The global shrinkage scale is calibrated as:

$$\tau_0 = \frac{D_0}{D - D_0} \cdot \frac{\sigma}{\sqrt{N}}$$

where $D_0$ is the expected number of nonzero coefficients.

#### Spike-and-Slab (Continuous Relaxation)

For NUTS-compatible sampling:

$$\beta_j = \gamma_j \cdot \beta_{\text{slab},j} + (1 - \gamma_j) \cdot \beta_{\text{spike},j}$$

where:
- $\gamma_j = \text{sigmoid}(\text{logit}_{\gamma_j} / T)$ — soft inclusion indicator
- $\text{logit}_{\gamma_j} \sim \mathcal{N}(\text{logit}(\pi), 1)$
- $\beta_{\text{slab},j} \sim \mathcal{N}(0, \sigma_{\text{slab}})$
- $\beta_{\text{spike},j} \sim \mathcal{N}(0, \sigma_{\text{spike}})$ with $\sigma_{\text{spike}} \ll \sigma_{\text{slab}}$
- $T$ is temperature (lower = sharper selection)

### Best Practices

1. **Pre-specify variable classification** before seeing any results
2. **Document rationale** for each variable's classification as confounder vs. precision
3. **Never tune hyperparameters** based on fit metrics (this is specification shopping)
4. **Report inclusion probabilities** alongside point estimates
5. **Show sensitivity** to `expected_nonzero` specification
6. **When uncertain** about causal role, use standard priors (don't apply selection)

---

## Mathematical Specification: Extended Models

This section provides the formal mathematical specification and statistical justification for the nested, multivariate, and combined models in the `mmm_extensions` module.

### Nested/Mediated Models

Nested models estimate causal pathways where media affects intermediate outcomes (mediators) which in turn affect the final outcome. This is essential when the business question is not just "does media work?" but "how does media work?"

#### Core Structure

The nested model specifies a system of equations:

**Stage 1 — Media → Mediator:**

$$M_t = \alpha_M + \sum_{c=1}^{C} \beta^{(M)}_c \cdot f_c(x_{c,t}) + \epsilon^{(M)}_t$$

**Stage 2 — Mediator → Outcome (with direct effects):**

$$y_t = \alpha_y + \gamma \cdot M_t + \sum_{c=1}^{C} \beta^{(D)}_c \cdot f_c(x_{c,t}) + \epsilon^{(y)}_t$$

where:
- $M_t$ is the mediator value at time $t$ (e.g., brand awareness)
- $f_c(x_{c,t})$ is the transformed media input (adstock + saturation) for channel $c$
- $\beta^{(M)}_c$ is the media → mediator effect for channel $c$
- $\gamma$ is the mediator → outcome effect
- $\beta^{(D)}_c$ is the direct media → outcome effect (bypassing the mediator)

#### Effect Decomposition

The total effect of media channel $c$ on the outcome decomposes as:

$$\text{Total Effect}_c = \underbrace{\beta^{(D)}_c}_{\text{Direct}} + \underbrace{\beta^{(M)}_c \cdot \gamma}_{\text{Indirect}}$$

The **proportion mediated** quantifies how much of the total effect flows through the mediator:

$$\text{Proportion Mediated}_c = \frac{\beta^{(M)}_c \cdot \gamma}{\beta^{(D)}_c + \beta^{(M)}_c \cdot \gamma}$$

This decomposition is identified under the standard mediation assumptions (sequential ignorability), which require that there are no unmeasured confounders of (1) media–mediator, (2) mediator–outcome, or (3) media–outcome relationships.

#### Mediator Types and Observation Models

The framework supports three mediator types, each with different observation models that determine how the latent mediator state relates to observed data.

##### FULLY_OBSERVED Mediators

For mediators with complete time series observations (e.g., website traffic, store visits from sensors):

$$M^{obs}_t = M_t + \nu_t, \quad \nu_t \sim \mathcal{N}(0, \sigma^2_\nu)$$

**Likelihood:**

$$M^{obs}_t \sim \mathcal{N}(M_t, \sigma^2_\nu) \quad \forall t$$

The measurement noise $\sigma_\nu$ captures sensor error or sampling variation. With complete observations, the latent mediator trajectory is tightly constrained by the data.

**Use cases:** Daily website sessions, hourly foot traffic counts, real-time social mentions.

##### PARTIALLY_OBSERVED Mediators

For mediators observed only at sparse intervals (e.g., monthly brand tracking surveys in a weekly model):

$$M^{obs}_{t_k} = M_{t_k} + \nu_{t_k}, \quad \nu_{t_k} \sim \mathcal{N}(0, \sigma^2_\nu)$$

where $\{t_1, t_2, \ldots, t_K\} \subset \{1, 2, \ldots, T\}$ are the observation times.

**Likelihood (partial):**

$$M^{obs}_{t_k} \sim \mathcal{N}(M_{t_k}, \sigma^2_\nu) \quad \text{only for } t \in \{t_1, \ldots, t_K\}$$

At non-observed times, the mediator is inferred from:
1. The structural model (media effects)
2. Interpolation/smoothing from observed points
3. The prior distribution

This is a **state-space model** where the observation equation applies only at observed times. The framework handles this by masking the likelihood:

```python
# Only observed times contribute to likelihood
pm.Normal("M_observed", mu=M_latent[mask], sigma=sigma_obs, observed=M_data[mask])
```

**Justification:** Brand metrics like awareness evolve continuously but are measured infrequently via surveys. The structural model (media → awareness) provides information about the trajectory between survey waves, while the observations anchor the estimates at measured points.

**Use cases:** Monthly brand tracking, quarterly NPS surveys, periodic market research.

##### FULLY_LATENT Mediators

For mediators that are never directly observed but are theoretically important:

$$M_t = \alpha_M + \sum_{c} \beta^{(M)}_c \cdot f_c(x_{c,t}) + \epsilon^{(M)}_t$$

**No observation likelihood** — the mediator is identified purely through:
1. Its structural relationship with media inputs
2. Its effect on the observed outcome
3. Prior distributions on parameters

**Identification requirements:** Fully latent mediators require strong assumptions:
- The functional form relating media to the mediator must be correctly specified
- The mediator must have a non-zero effect on the outcome ($\gamma \neq 0$)
- Sufficient variation in media inputs to identify $\beta^{(M)}$

**Bayesian regularization:** Informative priors on $\beta^{(M)}$ and $\gamma$ are critical. Without observed mediator data, the posterior is heavily influenced by priors.

**Use cases:** Latent brand equity, unobserved consideration sets, theoretical constructs without direct measurement.

#### Priors for Nested Models

The framework uses weakly informative priors with optional constraints:

| Parameter | Default Prior | Constraint Options |
|-----------|--------------|-------------------|
| $\beta^{(M)}_c$ (media → mediator) | $\mathcal{N}^+(0, 1)$ | Positive (media builds awareness) |
| $\gamma$ (mediator → outcome) | $\mathcal{N}(0, 1)$ | None (can be negative) |
| $\beta^{(D)}_c$ (direct effect) | $\mathcal{N}(0, 0.5)$ | None |
| $\sigma_\nu$ (observation noise) | $\text{HalfNormal}(0.1)$ | Positive |

The positive constraint on $\beta^{(M)}$ reflects the prior belief that media spending increases (not decreases) brand awareness. This can be relaxed if theoretically justified.

---

### Multivariate Outcome Models

When modeling multiple outcomes simultaneously (e.g., sales of different products), a multivariate model captures correlations and cross-product effects that univariate models miss.

#### Core Structure

For $K$ outcomes $\mathbf{y}_t = (y_{1,t}, \ldots, y_{K,t})'$:

$$\mathbf{y}_t = \boldsymbol{\alpha} + \mathbf{B} \cdot \mathbf{f}(x_t) + \boldsymbol{\Psi} \cdot \mathbf{y}_t + \boldsymbol{\epsilon}_t$$

where:
- $\boldsymbol{\alpha} \in \mathbb{R}^K$ is the intercept vector
- $\mathbf{B} \in \mathbb{R}^{K \times C}$ is the media effect matrix (outcome × channel)
- $\mathbf{f}(x_t) \in \mathbb{R}^C$ is the transformed media vector
- $\boldsymbol{\Psi} \in \mathbb{R}^{K \times K}$ is the cross-effect matrix (diagonal = 0)
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ is the error with covariance $\boldsymbol{\Sigma}$

#### Multivariate Normal Likelihood with LKJ Prior

The error covariance captures residual correlation across outcomes:

$$\boldsymbol{\epsilon}_t \sim \mathcal{N}_K(\mathbf{0}, \boldsymbol{\Sigma})$$

We decompose $\boldsymbol{\Sigma} = \mathbf{D} \mathbf{R} \mathbf{D}$ where:
- $\mathbf{D} = \text{diag}(\sigma_1, \ldots, \sigma_K)$ contains outcome-specific scales
- $\mathbf{R}$ is the correlation matrix

**LKJ Prior on Correlations:**

$$\mathbf{R} \sim \text{LKJCorr}(\eta)$$

The LKJ distribution is the standard prior for correlation matrices:
- $\eta = 1$: Uniform over valid correlation matrices
- $\eta = 2$: Mild shrinkage toward independence (recommended default)
- $\eta > 2$: Stronger shrinkage toward $\mathbf{R} = \mathbf{I}$

**Justification:** Outcomes like product sales are correlated due to shared drivers (weather, holidays, economic conditions) not captured by the model. The multivariate likelihood:
1. Improves efficiency by borrowing strength across outcomes
2. Provides valid inference that accounts for correlation
3. Enables testing hypotheses about outcome relationships

---

### Cross-Effects

Cross-effects model how one outcome causally affects another, beyond shared correlation.

#### Cannibalization

When product $j$'s sales reduce product $k$'s sales (substitution):

$$y_{k,t} = \ldots + \psi_{jk} \cdot y_{j,t} + \ldots, \quad \psi_{jk} < 0$$

**Promotion-modulated cannibalization:**

$$y_{k,t} = \ldots + \psi_{jk} \cdot P_{j,t} \cdot y_{j,t} + \ldots$$

where $P_{j,t} \in \{0, 1\}$ indicates whether product $j$ is on promotion. This captures the intuition that cannibalization is strongest during promotions.

**Prior:** $\psi_{jk} \sim \mathcal{N}^-(0, 0.3)$ (half-normal, negative)

#### Halo Effects

When product $j$'s sales increase product $k$'s sales (complementarity):

$$y_{k,t} = \ldots + \psi_{jk} \cdot y_{j,t} + \ldots, \quad \psi_{jk} > 0$$

**Prior:** $\psi_{jk} \sim \mathcal{N}^+(0, 0.3)$ (half-normal, positive)

**Use cases:** Premium brand lifts value brand, flagship product drives accessory sales.

#### Lagged Cross-Effects

For effects that manifest with delay:

$$y_{k,t} = \ldots + \psi_{jk} \cdot y_{j,t-1} + \ldots$$

This avoids simultaneity issues and may better reflect consumer behavior (e.g., stockpiling from a promotion reduces next-period purchases).

#### Identification of Cross-Effects

Cross-effects face identification challenges:
1. **Simultaneity:** $y_j$ and $y_k$ are jointly determined
2. **Confounding:** Shared drivers affect both outcomes

**Strategies implemented:**
- Lagged effects break simultaneity
- Promotion modulation provides exogenous variation
- Informative priors regularize toward zero
- The multivariate error structure captures residual correlation separately from causal effects

---

### Combined Models

The `CombinedMMM` integrates nested pathways and multivariate outcomes into a unified framework.

#### Full Specification

**Mediator equations** (for each mediator $m$):

$$M_{m,t} = \alpha_m + \sum_{c \in \mathcal{C}_m} \beta^{(M)}_{mc} \cdot f_c(x_{c,t}) + \epsilon^{(M)}_{m,t}$$

where $\mathcal{C}_m$ is the set of channels affecting mediator $m$.

**Outcome equations** (for each outcome $k$):

$$y_{k,t} = \alpha_k + \underbrace{\sum_{c=1}^{C} \beta^{(D)}_{kc} \cdot f_c(x_{c,t})}_{\text{Direct media effects}} + \underbrace{\sum_{m \in \mathcal{M}_k} \gamma_{km} \cdot M_{m,t}}_{\text{Mediator effects}} + \underbrace{\sum_{j \neq k} \psi_{jk} \cdot y_{j,t}}_{\text{Cross-effects}} + \epsilon_{k,t}$$

where $\mathcal{M}_k$ is the set of mediators affecting outcome $k$.

**Joint error distribution:**

$$\boldsymbol{\epsilon}_t = (\epsilon_{1,t}, \ldots, \epsilon_{K,t})' \sim \mathcal{N}_K(\mathbf{0}, \boldsymbol{\Sigma})$$

#### Effect Decomposition in Combined Models

For channel $c$ affecting outcome $k$, the total effect decomposes as:

$$\text{Total}_{ck} = \underbrace{\beta^{(D)}_{kc}}_{\text{Direct}} + \underbrace{\sum_{m \in \mathcal{M}_k} \beta^{(M)}_{mc} \cdot \gamma_{km}}_{\text{Indirect via mediators}} + \underbrace{\sum_{j \neq k} \psi_{jk} \cdot \text{Total}_{cj}}_{\text{Cross-outcome spillover}}$$

The last term captures how media affecting outcome $j$ spills over to outcome $k$ through cross-effects. This requires solving a system of equations when cross-effects are bidirectional.

#### DAG Representation

The combined model implies a directed acyclic graph (DAG):

```
                    ┌─────────────┐
                    │   Media     │
                    │  Channels   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │Mediator │  │Mediator │  │ Direct  │
        │    1    │  │    2    │  │ Effects │
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             └─────┬──────┴────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │      Outcomes        │
        │  ┌────┐    ┌────┐    │
        │  │ Y₁ │◄──►│ Y₂ │    │  ← Cross-effects
        │  └────┘    └────┘    │
        └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Correlated Errors   │
        │     Σ (LKJ prior)    │
        └──────────────────────┘
```

#### When to Use Combined Models

| Scenario | Model Choice |
|----------|--------------|
| Single outcome, no mediators | Standard `BayesianMMM` |
| Single outcome, mediators present | `NestedMMM` |
| Multiple outcomes, no mediators | `MultivariateMMM` |
| Multiple outcomes with mediators | `CombinedMMM` |
| Products with cross-effects | `MultivariateMMM` or `CombinedMMM` |
| Full brand funnel (awareness → consideration → purchase) | `CombinedMMM` with cascading mediators |

---

### Computational Considerations

#### Identifiability

Extended models have more parameters and thus higher risk of weak identification:

| Model | Key Identification Requirements |
|-------|--------------------------------|
| Nested (fully observed) | Variation in media, complete mediator data |
| Nested (partially observed) | Sufficient survey observations, informative priors |
| Nested (fully latent) | Strong priors, non-zero mediator effect |
| Multivariate | Independent variation across outcomes |
| Cross-effects | Exogenous variation (promotions), lagged structure |

**Diagnostics:** Always check:
- $\hat{R}$ (convergence): Should be < 1.01 for all parameters
- ESS (effective sample size): Should be > 400 for reliable inference
- Prior-posterior overlap: Wide overlap suggests weak identification

#### Scaling

| Model Component | Computational Cost |
|-----------------|-------------------|
| Additional mediator | ~1.3x per mediator |
| Additional outcome | ~1.5x per outcome |
| Cross-effects | ~1.1x per effect |
| Partial observation | ~1.2x (masking overhead) |

For complex models, use `nuts_sampler="numpyro"` for 4-10x speedup.

## Mathematical Specification: Variable Selection Priors

This section provides the formal mathematical specification for the variable selection priors available in `mmm_extensions`. For the complete technical document with proofs and implementation details, see `variable_selection_specification.pdf`.

### Causal Constraints

Variable selection priors should **only** be applied to **precision control variables**—variables that affect outcome $Y$ but do *not* affect treatment $X$ (media spending).

**Bias from Confounder Exclusion**: For a confounder $C$ affecting both media $X$ and outcome $Y$:

$$\hat{\beta}_{OLS} \xrightarrow{p} \beta + \gamma \cdot \frac{\text{Cov}(X, C)}{\text{Var}(X)}$$

The bias depends on the *correlation* between treatment and confounder, not just the confounder's effect magnitude. Shrinking $\gamma$ toward zero does not remove this bias.

---

### Regularized Horseshoe Prior

The regularized horseshoe (Piironen & Vehtari, 2017) provides adaptive shrinkage with a regularized slab.

**Model Specification**:

$$\gamma_j = z_j \cdot \tau \cdot \tilde{\lambda}_j$$

where:
- $z_j \sim \mathcal{N}(0, 1)$ — standardized coefficient
- $\tau \sim \text{Half-}t_{\nu_\tau}(0, \tau_0)$ — global shrinkage
- $\lambda_j \sim \text{Half-}t_{\nu_\lambda}(0, 1)$ — local shrinkage
- $c^2 \sim \text{Inv-Gamma}(\nu_s/2, \nu_s s^2/2)$ — slab variance

**Regularized Local Shrinkage**:

$$\tilde{\lambda}_j = \frac{c \cdot \lambda_j}{\sqrt{c^2 + \tau^2 \lambda_j^2}}$$

**Global Shrinkage Calibration**:

$$\tau_0 = \frac{D_0}{D - D_0} \cdot \frac{\sigma}{\sqrt{N}}$$

where $D_0$ is the expected number of nonzero coefficients.

**Shrinkage Factor**:

$$\kappa_j = \frac{1}{1 + \tau^2 \lambda_j^2}$$

- $\kappa_j \approx 1$: Strong shrinkage (coefficient → 0)
- $\kappa_j \approx 0$: Minimal shrinkage (coefficient preserved)

**Effective Number of Nonzero**:

$$m_{\text{eff}} = \sum_{j=1}^{D} (1 - \kappa_j)$$

---

### Spike-and-Slab Prior

The spike-and-slab provides explicit posterior inclusion probabilities.

**Continuous Relaxation** (for NUTS compatibility):

$$\gamma_j = \eta_j \cdot \beta_{\text{slab},j} + (1 - \eta_j) \cdot \beta_{\text{spike},j}$$

where:
- $\eta_j = \text{sigmoid}(\omega_j / T)$ — soft inclusion indicator
- $\omega_j \sim \mathcal{N}(\text{logit}(\pi), 1)$
- $\beta_{\text{slab},j} \sim \mathcal{N}(0, \sigma_{\text{slab}}^2)$
- $\beta_{\text{spike},j} \sim \mathcal{N}(0, \sigma_{\text{spike}}^2)$ with $\sigma_{\text{spike}} \ll \sigma_{\text{slab}}$
- $T$ = temperature (lower = sharper selection)

**Posterior Inclusion Probability**:

$$\Pr(\text{included}_j | \mathbf{y}) = \mathbb{E}[\eta_j | \mathbf{y}]$$

---

### Bayesian LASSO

The Bayesian LASSO places Laplace priors on coefficients via a scale mixture representation.

**Scale Mixture Representation**:

$$\gamma_j | \tau_j \sim \mathcal{N}(0, \tau_j), \quad \tau_j \sim \text{Exponential}\left(\frac{\lambda^2}{2}\right)$$

This is equivalent to:

$$\gamma_j \sim \text{Laplace}\left(0, \frac{1}{\lambda}\right)$$

**Shrinkage Properties**: Unlike the horseshoe, LASSO provides *uniform* shrinkage—all coefficients are shrunk by similar proportions.

---

### Method Selection Guide

| Scenario | Recommended Prior | Reason |
|----------|-------------------|--------|
| Few large effects, many zeros | Regularized Horseshoe | Adaptive shrinkage preserves signals |
| Many small effects | Bayesian LASSO | Uniform shrinkage appropriate |
| Need inclusion probabilities | Spike-and-Slab | Direct interpretation |
| Unknown sparsity structure | Regularized Horseshoe | Most robust |

---

### Hyperparameter Guidance

| Parameter | Symbol | Default | Selection Guidance |
|-----------|--------|---------|-------------------|
| Expected nonzero | $D_0$ | 3 | Domain knowledge; err toward larger |
| Slab scale | $s$ | 2.0 | Max plausible effect in std units |
| Slab df | $\nu_s$ | 4.0 | Lower = heavier tails |
| Local df | $\nu_\lambda$ | 5.0 | Lower = heavier tails |
| Prior inclusion | $\pi$ | 0.5 | 0.5 = maximum uncertainty |
| Temperature | $T$ | 0.1 | Lower = sharper selection |
| LASSO penalty | $\lambda$ | 1.0 | Higher = more shrinkage |

---

### Diagnostics

**Inclusion Probability** (Horseshoe):
$$\Pr(\text{included}_j | \mathbf{y}) \approx \mathbb{E}[1 - \kappa_j | \mathbf{y}]$$

**Inclusion Probability** (Spike-Slab):
$$\Pr(\text{included}_j | \mathbf{y}) = \mathbb{E}[\eta_j | \mathbf{y}]$$

**Report for each analysis**:
1. Variable classification (confounder vs precision)
2. Selection method and all hyperparameters  
3. Posterior inclusion probabilities
4. Effective number of nonzero ($m_{\text{eff}}$)
5. Sensitivity to hyperparameter choices

## Data Format

The framework expects data in **Master Flat File (MFF) format**—a fully normalized long-format structure with 8 columns:

| Column | Description |
|--------|-------------|
| `Period` | Time period identifier (date or week number) |
| `Geography` | Geographic unit (DMA, region, store, etc.) |
| `Product` | Product or brand identifier |
| `Campaign` | Campaign or flight identifier |
| `Outlet` | Media outlet or channel |
| `Creative` | Creative execution identifier |
| `VariableName` | Name of the metric (e.g., "Sales", "TV_Spend", "Price") |
| `VariableValue` | Numeric value for that metric |

### Example MFF Data

| Period | Geography | Product | Campaign | Outlet | Creative | VariableName | VariableValue |
|--------|-----------|---------|----------|--------|----------|--------------|---------------|
| 2023-01-01 | DMA_001 | SKU_A | Q1_Brand | TV | Hero_30s | TV_Spend | 50000 |
| 2023-01-01 | DMA_001 | SKU_A | Q1_Brand | TV | Hero_30s | TV_GRPs | 125 |
| 2023-01-01 | DMA_001 | SKU_A | Q1_Brand | Digital | Banner_A | Digital_Spend | 25000 |
| 2023-01-01 | DMA_001 | SKU_A | — | — | — | Sales | 12500 |
| 2023-01-01 | DMA_001 | SKU_A | — | — | — | Price | 4.99 |
| 2023-01-01 | DMA_001 | SKU_A | — | — | — | Distribution | 0.85 |
| 2023-01-01 | DMA_002 | SKU_A | Q1_Brand | TV | Hero_30s | TV_Spend | 30000 |
| ... | ... | ... | ... | ... | ... | ... | ... |

This normalized structure supports:

- **Multiple granularities** — National media (Geography = "National") alongside geo-specific data
- **Campaign attribution** — Track spend and response by campaign/flight
- **Creative-level analysis** — Compare performance across creative executions
- **Flexible aggregation** — Roll up from creative → outlet → campaign as needed

### Configuration for MFF Structure

```python
from mmm_framework import MFFConfigBuilder

mff_config = (
    MFFConfigBuilder()
    .with_date_column("Period", format="%Y-%m-%d")
    .with_dimension("Geography", type="geo")
    .with_dimension("Product", type="product")
    .with_dimension("Campaign", type="campaign")
    .with_dimension("Outlet", type="outlet")
    .with_dimension("Creative", type="creative")
    .with_variable_column("VariableName")
    .with_value_column("VariableValue")
    .with_kpi("Sales")
    .with_media_variables(["TV_Spend", "Digital_Spend", "Social_Spend"])
    .with_control_variables(["Price", "Distribution"])
    .build()
)
```

## Model Specification

### Additive Model

The default additive specification:

$$y_{jt} = \alpha_j + \sum_{m=1}^{M}\beta_m f_m(x_{m,jt}) + \sum_{c=1}^{C}\gamma_c z_{c,jt} + \text{Trend}_t + \text{Seasonality}_t + \epsilon_{jt}$$

where $f_m(x)$ composes adstock and saturation transformations.

### Multiplicative Model

For elasticity interpretation:

$$\log(y_{jt}) = \log(\beta_0) + \sum_m \beta_m \log(f_m(x_{m,jt})) + \gamma' Z_{jt} + \epsilon_{jt}$$

Coefficients represent elasticities: percent change in sales per percent change in media.

### Saturation Functions

**Logistic Saturation** (recommended for numerical stability):

$$f(x) = \frac{1 - e^{-\lambda x}}{1 + e^{-\lambda x}}$$

**Hill Saturation**:

$$f(x) = \frac{x^s}{x^s + K^s}$$

### Adstock (Carryover Effects)

**Geometric Adstock**:

$$A_t = x_t + \alpha A_{t-1}$$

where $\alpha \in [0, 1)$ controls decay rate.

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with Redis and worker status |
| `POST` | `/data/upload` | Upload MFF data file |
| `GET` | `/data` | List uploaded datasets |
| `GET` | `/data/{id}` | Get dataset details |
| `POST` | `/configs` | Create model configuration |
| `GET` | `/configs` | List configurations |
| `GET` | `/configs/{id}` | Get configuration details |
| `POST` | `/models/fit` | Start async model fitting |
| `GET` | `/models/{id}/status` | Get fitting progress |
| `GET` | `/models/{id}/results` | Get fitted model results |
| `GET` | `/models/{id}/contributions` | Get channel contributions |
| `POST` | `/models/{id}/predict` | Generate predictions |

### Example: Fit a Model via API

```bash
# Upload data
curl -X POST "http://localhost:8000/data/upload" \
  -F "file=@data.csv"

# Create configuration
curl -X POST "http://localhost:8000/configs" \
  -H "Content-Type: application/json" \
  -d @config.json

# Start fitting
curl -X POST "http://localhost:8000/models/fit" \
  -H "Content-Type: application/json" \
  -d '{"data_id": "abc123", "config_id": "xyz789"}'

# Check status
curl "http://localhost:8000/models/{model_id}/status"

# Get results
curl "http://localhost:8000/models/{model_id}/results"
```

## Methodological Foundation

This framework is built on established statistical methodology and addresses known problems in marketing mix modeling practice.

### The Problem with Specification Shopping

When analysts test multiple model specifications and select based on results (statistical significance, expected signs, "reasonable" ROIs), they invalidate standard statistical inference:

- Testing 20 specifications at α=0.05 yields >64% probability of at least one false positive
- Coefficients selected for significance are systematically biased upward (winner's curse)
- Confidence intervals no longer have nominal coverage

### Bayesian Approach Benefits

1. **Genuine uncertainty quantification** — Posterior distributions reflect actual uncertainty from data limitations
2. **Prior incorporation** — External evidence from experiments or meta-analyses can be formally included
3. **Hierarchical modeling** — Partial pooling across sparse groups improves estimation
4. **Decision-theoretic integration** — Posteriors integrate naturally with business decision analysis

### Identification Considerations

The framework explicitly addresses common identification problems:

- **National media with geo-level data** — Random effects on national media cannot be interpreted as differential causal response since national media provides no geo-level exposure variation
- **Saturation-scaling confounding** — When saturation functions are applied, geo-level exposure scaling becomes unidentified due to confounding between scaling and saturation parameters

### Recommended Workflow

1. **Pre-specify the model** before looking at results
2. **Use prior predictive checks** to validate priors make sense
3. **Fit with full uncertainty** using Bayesian inference
4. **Validate against experiments** where feasible
5. **Report credible intervals**, not just point estimates

## Performance

### Inference Speed Comparison

| Method | Time (100 obs) | Uncertainty |
|--------|---------------|-------------|
| Ridge/NNLS | <10 ms | Bootstrap only |
| CVXPY constrained | 10-100 ms | Bootstrap only |
| PyMC ADVI | 10-30 sec | Approximate posterior |
| PyMC + NumPyro | 30 sec - 2 min | Full posterior |
| PyMC CPU | 2-20 min | Full posterior |

### Recommendations

- **Development iteration**: Use `Ridge(positive=True)` with differential evolution for transformation search
- **Production models**: Use `nuts_sampler="numpyro"` for 4-10x speedup over CPU PyMC
- **GPU acceleration**: Additional 4x gains available with JAX on GPU

## Project Structure

```
mmm-framework/
├── src/mmm_framework/          # Core modeling library
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration enums and dataclasses
│   ├── builders.py             # Fluent configuration builders
│   ├── data_loader.py          # MFF parsing and validation
│   ├── model.py                # BayesianMMM implementation
│   ├── jobs.py                 # Async job management
│   └── mmm_extensions/         # Extended model capabilities
│       ├── __init__.py         # Lazy imports for heavy dependencies
│       ├── config.py           # Mediator, Outcome, CrossEffect configs
│       ├── builders.py         # Fluent builders + factory functions
│       ├── components.py       # PyMC/PyTensor building blocks
│       └── models.py           # NestedMMM, MultivariateMMM, CombinedMMM
├── api/                        # FastAPI backend
│   ├── main.py                 # Application factory
│   ├── routes/                 # API route handlers
│   ├── schemas.py              # Pydantic models
│   ├── redis_service.py        # Redis connection management
│   └── worker.py               # ARQ worker settings
├── app/                        # Streamlit frontend
│   ├── Home.py                 # Main entry point
│   ├── pages/                  # Multipage app pages
│   ├── api_client.py           # HTTP client for backend
│   └── components/             # Reusable UI components
├── examples/                   # Usage examples
│   └── ex_extensions.py        # Extended model examples
├── tests/                      # Test suite
│   └── mmm_extensions/         # Extension module tests
├── pyproject.toml              # Project configuration
└── README.md
```

## Dependencies

### Core

- `pymc>=5.26` — Probabilistic programming
- `pymc-marketing` — MMM components (saturation, adstock)
- `numpyro>=0.19` — JAX-based NUTS sampler
- `nutpie>=0.16` — Fast NUTS implementation
- `pandas>=2.3` — Data manipulation
- `numpy>=2.3` — Numerical computing

### Backend

- `fastapi>=0.124` — API framework
- `redis>=7.1` — Queue backend
- `arq>=0.25` — Async job queue
- `pydantic>=2.12` — Data validation
- `uvicorn>=0.38` — ASGI server

### Frontend

- `streamlit>=1.52` — Web application
- `plotly>=6.5` — Interactive visualization
- `httpx>=0.28` — HTTP client

## References

### Bayesian Methods

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.)
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.)

### Marketing Mix Modeling

- Jin, Y., et al. (2017). Bayesian methods for media mix modeling with carryover and shape effects. *Google Research*.
- Chan, D., & Perry, M. (2017). Challenges and opportunities in media mix modeling. *Google Research*.

### Specification Shopping & Replication

- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology. *Psychological Science*, 22(11), 1359-1366.
- Silberzahn, R., et al. (2018). Many analysts, one data set. *Advances in Methods and Practices in Psychological Science*, 1(3), 337-356.
- Camerer, C. F., et al. (2016). Evaluating replicability of laboratory experiments in economics. *Science*, 351(6280), 1433-1436.
  
### Variable Selection

- Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018-5051.
- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2), 465-480.
- George, E. I., & McCulloch, R. E. (1993). Variable selection via Gibbs sampling. *JASA*, 88(423), 881-889.
- Park, T., & Casella, G. (2008). The Bayesian Lasso. *JASA*, 103(482), 681-686.
- Piironen, J., & Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018-5051.
- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2), 465-480.
- George, E. I., & McCulloch, R. E. (1993). Variable selection via Gibbs sampling. *JASA*, 88(423), 881-889.
- Park, T., & Casella, G. (2008). The Bayesian Lasso. *JASA*, 103(482), 681-686.

## Contributing

Contributions are welcome. Please ensure:

1. All new features include tests
2. Code follows the existing style (run `ruff check` and `ruff format`)
3. Documentation is updated for API changes
4. Commit messages are descriptive


## Author

Matthew Reda (m.reda94@gmail.com)