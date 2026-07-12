# StructuralNestedMMM — a multi-mediator structural MMM

**Status:** design spec (2026-07-09) → implemented alongside this document.
**Where:** `mmm_extensions/models/structural.py` + `mmm_extensions/components/latent_states.py`
+ new config classes in `mmm_extensions/config.py`.
**Motivating example:** TV → *awareness* (binary survey, weekly varying sample size, strong
week-to-week persistence); Display + Social + awareness + price + a latent demand trend →
*consideration* (Likert survey); consideration + demand + direct-response channels → *sales*.

## 1. Why the existing NestedMMM is not enough

The current `NestedMMM` (`mmm_extensions/models/nested.py`) hardcodes exactly one structural
shape: `mediator = intercept + Σ β·sat(media)`, Gaussian point observation (or a survey path
that clips an *unbounded linear* latent into [0,1] as a binomial probability — the documented
scale-mismatch root cause of the PyMC-6 mediation under-recovery, see
`technical-docs/nested-recovery-search.md`). It cannot express:

1. **Per-mediator observation families.** A brand-awareness tracker is a *binary* question
   ("have you seen this brand?") asked of `n_t` respondents per week, with `n_t` varying — the
   weekly sampling variance is `p(1-p)/n_t`, not a constant Gaussian noise. Consideration is
   often a *Likert* item (ordered categories). Neither has a home today.
2. **Latent mediator dynamics.** Population awareness *carries over*: most people who were
   aware last week are aware this week. That is an AR(1) latent state with high persistence,
   not an adstock on the media input (though the two interact — see §4.3).
3. **Mediator → mediator edges.** Consideration is a function of awareness.
4. **Controls in mediator equations.** Consideration responds to price.
5. **Shared latent factors.** A latent demand trend moves consideration *and* sales; leaving
   it out confounds the mediated path (demand → consideration and demand → sales is a
   back-door through the mediator).

`StructuralNestedMMM` generalizes the mediator block into a small **structural equation
system over a DAG of latent states**, each with its own dynamics and measurement model,
while keeping the `BaseExtendedMMM` contract (arrays in, `ModelResults` out, experiment
calibration, approximate fits, extended-report duck-typing).

## 2. Config layer

New frozen dataclasses in `mmm_extensions/config.py` (sibling system to `MediatorConfig` —
the old classes are untouched; `NestedMMM` keeps its byte-identical graph).

```python
class MediatorDynamics(str, Enum):
    STATIC = "static"          # z_t = level + drivers_t  (no state carryover)
    AR1 = "ar1"                # z_t = level + Σ_{s≤t} ρ^{t-s}(drivers_s + σ·ε_s)
    RANDOM_WALK = "random_walk"  # ρ = 1 (accumulating; use for slow-moving stocks)

class MediatorLikelihood(str, Enum):
    GAUSSIAN = "gaussian"   # continuous index, standardized, masked Normal
    BINOMIAL = "binomial"   # per-period success counts + per-period trials, logit link
    ORDERED = "ordered"     # per-period Likert category counts, cumulative-logit Multinomial
    LATENT = "latent"       # never observed

@dataclass(frozen=True)
class MediatorMeasurement:
    likelihood: MediatorLikelihood = GAUSSIAN
    noise_sigma: float = 0.3        # gaussian: HalfNormal prior scale on obs noise (std-scale)
    design_effect: float = 1.0      # binomial/ordered: n_eff = n / design_effect
    n_categories: int | None = None # ordered: K (Likert points)
    cutpoint_prior_sigma: float = 2.0

@dataclass(frozen=True)
class MediatorSpec:
    name: str
    channels: tuple[str, ...] = ()        # media channels entering this equation
    parents: tuple[str, ...] = ()         # upstream mediators (must form a DAG)
    controls: tuple[str, ...] = ()        # control columns entering this equation
    latent_factors: tuple[str, ...] = ()  # latent factor names entering this equation
    dynamics: MediatorDynamics = STATIC
    rho_prior_alpha: float = 6.0          # Beta prior on ρ (AR1) — default favors persistence
    rho_prior_beta: float = 2.0
    innovation_sigma: float = 0.3         # HalfNormal prior scale on state innovations (AR1/RW)
    measurement: MediatorMeasurement = MediatorMeasurement()
    media_effect: EffectPriorConfig = EffectPriorConfig(POSITIVE, sigma=1.0)
    parent_effect: EffectPriorConfig = EffectPriorConfig(POSITIVE, sigma=1.0)
    control_effect: EffectPriorConfig = EffectPriorConfig(NONE, sigma=1.0)
    factor_effect_sigma: float = 1.0      # loading prior scale for latent factors
    outcome_effect: EffectPriorConfig = EffectPriorConfig(POSITIVE, sigma=1.0)  # γ_m
    affects_outcome: bool = True          # False → mediator only feeds other mediators
    allow_direct_effect: bool = True      # its channels get a (tight) direct δ path to y
    direct_effect: EffectPriorConfig = EffectPriorConfig(NONE, sigma=0.3)  # recovery lesson 4
    apply_adstock: bool | None = None     # None → dynamics-resolved: STATIC on, AR1/RW off
    state_parameterization: str = "auto"  # auto|centered|non_centered (see §4.3)

@dataclass(frozen=True)
class LatentFactorSpec:
    name: str
    dynamics: MediatorDynamics = AR1      # STATIC allowed; RW allowed
    rho_prior_alpha: float = 8.0
    rho_prior_beta: float = 2.0
    affects_outcome: bool = True
    outcome_effect_sigma: float = 1.0
    mediator_effect_sigma: float = 1.0
    anchor: str = "auto"                  # where the sign is pinned (HalfNormal loading):
    # auto = first MEASURED mediator consumer, falling back to the outcome. The anchor
    # must be a loading the data holds materially nonzero: a reflected factor mode
    # escapes a HalfNormal anchor whose true loading is small by sitting at its
    # cost-free zero mode — observed as split-chain R-hat ≈ 1.75 in the brand-funnel
    # recovery when the anchor was the (small) outcome loading; re-anchoring at the
    # densely-measured consideration loading dropped R-hat to 1.08 and locked the
    # factor onto the true demand series (corr 0.95).

@dataclass(frozen=True)
class StructuralNestedConfig:
    mediators: tuple[MediatorSpec, ...]
    latent_factors: tuple[LatentFactorSpec, ...] = ()
    outcome_controls: tuple[str, ...] | None = None  # None → all provided controls
    nonmediated_effect: EffectPriorConfig = EffectPriorConfig(NONE, sigma=1.0)
    # channels not routed to any mediator get a plain direct beta with this prior
```

`StructuralNestedConfig.__post_init__` validates: non-empty mediators, unique
mediator/factor names, `parents` reference existing mediators, the parent graph is
**acyclic** (Kahn topological sort — the sorted order is reused at build time),
`latent_factors` reference declared factors, `ORDERED` measurement has `n_categories ≥ 3`,
`design_effect ≥ 1` (it deflates information; <1 would claim more precision than the
sample), a fully-latent mediator has at least one driver, and **every latent factor enters
at least one MEASURED mediator equation** (review finding: a factor identified only through
the outcome residual is an unidentified noise absorber — only `w² + σ_y²` would be
pinned). Channel/control *column* existence and mediator/factor-vs-channel/control name
collisions are validated at model construction (the config does not know the data).

## 3. Constructor / data contract

```python
StructuralNestedMMM(
    X_media, y, channel_names, config,             # BaseExtendedMMM contract
    mediator_data: dict[str, np.ndarray] | None,   # per-mediator observations
    mediator_trials: dict[str, np.ndarray] | None, # binomial: per-period sample sizes n_t
    X_controls: np.ndarray | None, control_names: list[str] | None,
    index=None, model_config=None, trend_config=None,
)
```

- `mediator_data[name]`:
  - `GAUSSIAN` → `(n_obs,)` float, `NaN` = unobserved week.
  - `BINOMIAL` → `(n_obs,)` success counts (float, `NaN` = no survey that week).
  - `ORDERED` → `(n_obs, K)` per-category counts; a `NaN` (or all-zero) row = unobserved.
  - `LATENT` → key must be absent.
- `mediator_trials[name]` (`BINOMIAL` only): `(n_obs,)` respondents per week; weeks with
  `NaN`/`0` trials are treated as unobserved. **This is how weekly varying survey volume
  enters**: the binomial variance `n_t·p(1-p)` re-weights each week's evidence automatically.
- Controls are z-scored internally (per column, guard `+1e-8`); mediator and outcome
  equations consume the standardized values, matching the recovery lesson that inline
  `Normal(0,1)` control betas on standardized controls avoid over-absorption.
- Validation at init: every configured channel/control/mediator reference resolves;
  binomial counts ≤ trials on observed weeks; ordered `K` matches the data's column count;
  ≥ 3 observed points per measured mediator (soft floor, warn below 8).

## 4. Model structure

All equations are built inside one `pm.Model` on the standardized-outcome / link scales.
Time axis = `"obs"` (extension models are single national series).

### 4.1 Media transforms

Reuses `BaseExtendedMMM._media_transform_apply` per channel — `alpha_<ch> ~ Beta(2,2)`,
`lambda_<ch> ~ Gamma(3,1)`, normalize-by-max → geometric adstock (L=8, normalized) →
logistic saturation. One shared transform per channel (shared across every consuming
equation, like `share_adstock_across_mediators=True` today).

**Adstock vs AR(1) double-counting (§1.2):** when a mediator has `AR1`/`RW` dynamics its
channels enter as `saturation(normalized spend)` with **no adstock** — the AR(1) state
supplies all carryover (media this week lifts awareness, which persists via ρ).
`apply_adstock=None` (the default) resolves by dynamics (STATIC → on, AR1/RW → off);
forcing it on for a dynamic equation warns (α↔ρ ridge). A channel consumed by both an
adstocked and a non-adstocked equation gets both variants built from the *same* `lambda`
RV (saturation shape is a channel property; carryover routing is an equation property).
The adstock `alpha_<ch>` RV is built **only when some equation consumes the adstocked
series** (an adstock-enabled mediator or the channel's direct outcome path) — otherwise it
would be a dead free RV sampling its prior (confirmed code-review finding).

### 4.2 Latent factors (e.g. demand)

For each `LatentFactorSpec`, on the obs axis:

```
ε_f ~ Normal(0, 1, dims="obs")                       # non-centered, unit innovations
ρ_f ~ Beta(rho_prior_alpha, rho_prior_beta)          # AR1 only
F_raw = decay(ρ_f) @ ε_f                             # scan-free lower-triangular Toeplitz
F = (F_raw - mean(F_raw)) / (std(F_raw) + 1e-6)      # in-graph unit-variance standardization
pm.Deterministic(f"factor_{name}", F, dims="obs")
```

The in-graph standardization pins the factor's scale (the `latent_factor_mmm` lesson: the
AR(1) marginal variance `1/(1-ρ²)` otherwise trades off against the loadings and the
sampler collapses them). **Sign anchoring:** the factor's *outcome* loading is
`HalfNormal(outcome_effect_sigma)` when `affects_outcome=True`, otherwise the loading on its
*first consuming mediator* is HalfNormal; all other loadings are free-sign
`Normal(0, mediator_effect_sigma)`. A factor nobody consumes is a config error.

### 4.3 Mediator equations (topologically ordered)

For mediator `m` with driver sum

```
drivers_t = Σ_c β_{c→m}·sat_c(t) + Σ_p λ_{p→m}·signal_p°(t) + Σ_k φ_{k→m}·x_k(t) + Σ_f w_{f→m}·F_f(t)
```

(`signal_p°` = the parent's centered natural-scale signal, §4.5), the latent state is:

- `STATIC`: `z = level_m + drivers` (+ iid noise when measured non-Gaussian, below)
- `AR1`: `z = level_m + decay(ρ_m) @ (drivers + σ_m·ε_m)`, `ε_m ~ N(0,1)` non-centered,
  `σ_m ~ HalfNormal(innovation_sigma)`, `ρ_m ~ Beta(rho_prior_alpha, rho_prior_beta)`.
  The `level_m` intercept sits **outside** the recursion (the state is
  `z_t − level = ρ(z_{t-1} − level) + drivers_t + σε_t`), so the baseline does not ramp up
  from zero over a burn-in window; media impulses correctly *accumulate* through the state
  (sustained unit driver → steady-state lift `1/(1-ρ)` — the "large latent autoregressive
  effect").
- `RANDOM_WALK`: as AR1 with ρ ≡ 1 (cumsum; no ρ RV) and **no level** — the walk's start
  and a free level are an exact ridge (`level ↔ ε₀`), so location is carried by the state
  itself.

`level_m ~ Normal(0, 2)` — **except** `ORDERED` mediators (cutpoints absorb location; a free
intercept would be exactly confounded) and `RANDOM_WALK` states (above).

**Process-noise identification rule (review-hardened):** the innovation term `σ_m·ε_m`
exists when the mediator is *measured* AND (dynamics ≠ STATIC OR the measurement is
non-Gaussian). A `LATENT` mediator's state is deterministic given its drivers (`σ ≡ 0`) —
n_obs unidentified latents with no measurement would just absorb outcome residual. A
STATIC + GAUSSIAN mediator has no separate process noise (not separately identified from
measurement noise without dynamics). STATIC + BINOMIAL/ORDERED keeps an iid noise term as
**overdispersion slack**: a large-n_t tracker's binomial variance is far tighter than real
week-to-week methodology wobble, and without slack the misfit corrupts β (review finding).

**Adstock × AR warning:** building an AR1/RW mediator with `apply_adstock=True` on its
channels warns — two nearly-interchangeable geometric carryovers create an α↔ρ ridge. The
`binary_survey_mediator` factory sets `apply_adstock=False`.

### 4.4 Measurement models (`components/latent_states.py`)

Each observed mediator pins its latent's scale through its measurement — the load-bearing
generalization of recovery lesson 1 ("standardize the survey"):

- **GAUSSIAN**: observed series z-scored over its *observed* entries at data-prep;
  `obs_std[mask] ~ Normal(z[mask], σ_obs)`, `σ_obs ~ HalfNormal(noise_sigma)`. The latent is
  thereby *defined* on the standardized survey scale (loading fixed at 1): the survey pins
  media→mediator, which is what restored mediation recovery.
- **BINOMIAL**: `p = sigmoid(z)`; `counts[mask] ~ Binomial(n_eff[mask], p[mask])` with
  `n_eff = round(n_t / design_effect)` (counts rescaled by the same factor when
  `design_effect > 1`). The logit link replaces the old `pt.clip` — no dead gradients, and
  the probability scale pins both location and scale of `z` *absolutely* (a stronger anchor
  than standardization). Weekly varying `n_t` gives exactly the finite-sample weekly variance
  the user's tracker has.
- **ORDERED**: cutpoints are **generatively ordered** — anchor `c₀ ~ Normal(−(K−2)/2, σ_c)`
  plus `K−2` positive gaps `~ Gamma(2, 2)`, `c = c₀ + cumsum([0, gaps])`, registered as the
  `<m>_cutpoints` Deterministic. NOT `transform=ordered`: transforms only affect logp
  space, so prior-predictive forward draws would be unordered (negative cell probabilities
  silently laundered by Multinomial — verified in review), breaking
  `sample_prior_predictive` / parameter-learning diagnostics. Cumulative-logit cells
  `P(Y=k) = σ(c_k − z) − σ(c_{k-1} − z)` (clipped + renormalized against tail underflow);
  `counts_t ~ Multinomial(n_t, p_t)` on observed rows with `n_t` = the row sum of the
  (design-effect-deflated, rounded) cells — the row-sum invariant holds by construction.
  Scale is identified against the implicit unit-logistic response noise; location lives in
  the cutpoints (mediator level fixed at 0, §4.3).
- **LATENT**: no measurement, and deliberately **no in-graph standardization** (review
  finding: media-dependent standardization constants would be recomputed under a
  counterfactual `set_data` swap, contaminating every contrast, and a common rescale of
  the driver βs would be an exactly flat direction). The β·γ path products are identified
  through the priors; individual edges are not — a STATIC LATENT mediator is
  observationally a reparameterized direct effect.

### 4.5 Downstream signal convention

Downstream consumers (child mediators, the outcome) read a mediator's **uncentered
natural-scale signal**: `p_m = sigmoid(z_m)` for BINOMIAL (interpretable as population
share aware), `z_m` otherwise. **No in-graph centering** — the design originally centered
at consumption, but the review found this makes the summed counterfactual mediated
contrast *identically zero* (the mean is a graph op, recomputed under intervention:
`Σ_t γ(s_t − s̄) ≡ 0` in both arms). The consumer's intercept absorbs the signal's level
instead, exactly as in `NestedMMM`. Registered deterministics: `<m>_latent` (z,
dims="obs"), `<m>_probability` (BINOMIAL only).

### 4.6 Outcome equation

On the standardized-y scale:

```
μ_std = α_y + Σ_m 1[affects_outcome] γ_m·signal_m° + Σ_c∈direct δ_c·sat_c
        + Σ_c∈nonmediated β_c·sat_c + Σ_k β_ctrl_k·x_k + Σ_f w_f·F_f
        + trend + seasonality
```

- `γ_m` from `outcome_effect` (default positive HalfNormal).
- **Direct effects** `δ_c` (`delta_direct_<ch>`): channels that feed ≥1 mediator whose
  `allow_direct_effect=True`, with the granting mediator's `direct_effect` prior — default
  `Normal(0, 0.3)` **tight** (recovery lesson 4: an over-wide direct path steals the
  mediated signal).
- **Non-mediated channels** (feeding no mediator, e.g. Search) get a plain full-strength
  `beta_<ch>` with `nonmediated_effect` prior.
- Controls: `outcome_controls` (default all), inline `Normal(0,1)` betas on standardized
  values (recovery lesson 5).
- Trend/seasonality/likelihood: the existing `build_trend_contribution` /
  `build_seasonality_contribution` / `build_outcome_likelihood` components — `None` configs ⇒
  no RVs, and spec-driven configs mean the same thing as a plain MMM's (recovery lesson 3
  is the *user's* responsibility: prefer a modest trend over a kitchen-sink baseline).
- Report-contract deterministics (extended-extractor vocabulary): `mu` (original units),
  `effect_<m>_on_y`, `direct_effect_<ch>`, `trend_component`, `seasonality_component`
  (all × `y_std`).

### 4.7 Experiment calibration

Total-effect handles are **nonlinear** here (a channel's effect routes through sigmoid
states and AR gains), so V1 attaches calibration handles **only for channels whose every
reachable mediator is STATIC and non-BINOMIAL** — there the coefficient product is the
exact derivative. Channels routed through AR dynamics or a sigmoid link are **skipped with
a warning** (review finding: a steady-state-gain linearization overstates a finite test
window ~3× at ρ=0.9/W=4, and is undefined for RANDOM_WALK — refusing beats attaching the
likelihood to a mis-scaled estimand). Window-aware nonlinear calibration is deferred (§7);
exact counterfactual estimands stay available post-fit (§5).

## 5. Effects & decomposition (post-fit)

The exact total effect of a channel is a **counterfactual contrast**, not a coefficient
product, because of the sigmoid links and AR dynamics:

- `_counterfactual_mu(multipliers)`: `pm.set_data({"X_media": X·mult})` +
  `sample_posterior_predictive(var_names=["mu"])` — deterministic recompute under posterior
  draws with the AR innovations *held fixed* (structure-preserving intervention; the same
  demand shocks, different media). Restores the original data afterwards.
- `get_mediation_effects()` → per channel: `total = Σ_t(mu_base − mu_{c=0})` (posterior
  mean + sd), `direct = Σ_t δ_c·sat_c·y_std` (exact in-graph), `mediated = total − direct`,
  `proportion_mediated`. Matches the `NestedMMM`/garden-model result vocabulary.
- `get_channel_roas(spend=None)` → `total_contribution / spend_sum` per channel (spend
  defaults to the raw `X_media` sums; pass real dollars when the modeled variable is not
  spend).
- `get_pathway_effects()` → the *linearized* per-path table (c → m₁ → … → y strength via
  coefficient products × mean sigmoid slopes × AR gains), labeled approximate — for "how
  much of TV flows via awareness vs directly", complementing the exact channel totals.

## 6. Identification notes (design rationale)

1. **Every measured mediator pins its own scale** (standardized Gaussian scale, absolute
   probability scale, or cutpoint-anchored ordered scale). This is the generalization of
   the single load-bearing fix from the recovery search — media→mediator paths are
   identified by the mediator data, not from the outcome residual.
2. **Latent scale rules**: unmeasured states are unit-standardized in-graph; AR innovation
   noise exists only under a measurement; ordered states have no free level. Each rule
   removes one exact non-identifiability.
3. **Funnel chains** (awareness → consideration → sales with `affects_outcome=False`
   upstream) are identified when each stage is measured; a fully-latent middle stage
   reduces to a rescaled composite edge (documented, allowed).
4. **The latent demand factor is deliberately shared** (consideration *and* sales): a
   common-cause trend entering only the outcome would leave the mediator equation
   confounded. Sign is pinned at the outcome loading.
5. **Known residual risk**: with few survey waves and high ρ, the AR(1) state can absorb
   slow media effects (awareness SNR limits from the recovery memo apply). The tight direct
   prior + measured mediators keep the decomposition stable; the recovery test asserts
   structure (signs, shares, ρ, factor correlation), not point magnitudes.

## 7a. Fit-path wiring (SHIPPED, 2026-07-09 follow-up)

The agent fit path reaches `StructuralNestedMMM`:

- **Resolution** (`dag_model_builder/model_type_resolver.py`):
  `ModelType.STRUCTURAL_NESTED_MMM = "structural_nested_mmm"`. A mediator DAG upgrades
  from NESTED when it uses features only the structural model can express — a
  **mediator→mediator** or **control→mediator** edge (both now legal in
  `validation.py`; the resolver upgrade guarantees such DAGs never route to plain
  nested, which would silently drop the edges), any **structural node-config key**
  (`STRUCTURAL_MEDIATOR_KEYS` in `node_configs.py`), a
  `dag.metadata["latent_factors"]` declaration, or an explicit
  `dag.metadata["model_type"]` override. Structural + multiple outcomes/cross-effects
  raises (single-outcome only). `build_model_from_dag` stamps the new type
  automatically.
- **Node configs** (`node_configs.py`): `MediatorNodeConfig` gained optional structural
  fields — `dynamics`, `likelihood`, `trials_variable` (binomial weekly-N MFF column),
  `category_variables` (ordered count columns, low→high), `design_effect`,
  `cutpoint_prior_sigma`, `rho_prior_alpha/beta`, `innovation_sigma`,
  `state_parameterization`, `affects_outcome`, `parent_effect_sigma`,
  `control_effect_sigma`, `latent_factors` — all `None`/neutral defaults so existing
  plain-nested configs are untouched.
- **Translation** (`config_translator.py::dag_to_structural_config`): channels from
  MEDIA→mediator edges, parents from mediator→mediator edges, controls from
  CONTROL→mediator edges; maps ONLY keys present in the RAW node-config dict so
  `MediatorSpec` defaults (tight direct prior, dynamics-resolved adstock) hold when
  unset; latent factors from `dag.metadata["latent_factors"]`
  (LatentFactorSpec-shaped dicts, strict key validation); `outcome_controls` = the
  CONTROL→KPI edge set (a control driving only a mediator stays out of the outcome
  equation).
- **Build** (`builder.py`): the STRUCTURAL branch pulls survey series via
  `_national_series(..., allow_missing=True)` — **NaN = no survey that week** (the
  strict-coverage rule stays for plain nested/multivariate); binomial trials from
  `trials_variable`, ordered `(n_obs, K)` matrices column-stacked from
  `category_variables`; forwards `X_controls`/`control_names` (extension branches
  previously discarded controls).
- **Agent registry** (`agents/fitting.py`): `structural_nested_mmm` joined
  `_EXTENSION_DAG_TYPES`; `priors.mediator.<name>.*` accepts
  `_STRUCTURAL_MEDIATOR_PRIOR_KEYS` (the plain keys + rho/innovation/design-effect/
  cutpoint/parent/control/state-parameterization knobs) for structural specs only;
  `spec["latent_factors"]` is a validated top-level key (structural specs only);
  `_build_extension_model` stamps `metadata["model_type"]` + copies
  `spec["latent_factors"]` into the DAG metadata before validation.
- **Data** (`src/mmm_framework/synth/mff.py::brand_funnel_mff`): the brand-funnel world as an MFF long
  table INCLUDING `awareness_count`/`awareness_trials` + `consideration_cat_1..5`
  blocks (unobserved weeks omitted → NaN on load), + answer key with the survey
  variable names. Tests: `tests/test_structural_fit_path.py` (resolver rules,
  E2E build_model → StructuralNestedMMM, sparse-survey masks, MAP fit, prior
  injection, registry acceptance/rejection).
- **Wiring review outcomes** (2-lens adversarial + verify, all fixed):
  `describe_model_type` KeyError for the new type (blocker — crashed
  `build_model_from_dag` for every structural DAG; now has an entry + a
  defensive `.get` default); `affects_outcome` now **defaults from the DAG's
  mediator→outcome edges** (a mediator without an edge into the KPI must not
  get a γ path the DAG doesn't draw; explicit config key still overrides);
  the FE extension gate (`WorkspaceTabs`) + widget label know the structural
  type (prevents the plain prior editor wiping `priors.mediator.*`);
  `latent_factors` is writable through `update_model_setting` via per-factor
  sub-paths (`latent_factors.<name>.<key>` with upsert semantics — the
  list-of-dicts shape is FE/programmatic-only), an explicit empty list CLEARS
  DAG-declared factors, survey columns without a matching `likelihood` raise
  at translation, duplicate driver edges are deduped, and partial ordered
  rows (some categories missing) warn instead of silently dropping.
- ~~Extension auto-save gap~~ **FIXED** (same-day follow-up): `MMMSerializer`
  gained an **extended flavor** for the whole `BaseExtendedMMM` family —
  `_save_extended` writes the pickled instance (arrays + configs; graph/trace
  stripped) with the platform-legible `metadata.json`
  (`model_flavor: "extended"`, class qualname, fit provenance from the new
  `_fit_diagnostics` stamp, experiments) / `configs.json` sidecars and the
  gzipped trace, through the same atomic swap; `load()` detects the flavor
  from metadata and loads **panel-free** (`panel` is now optional — core
  saves without one raise a clear TypeError). `load_model_core` gets an
  extended fast path (no dataset requirement), the cold-kernel rehydrator
  flavor-checks before its panel rebuild, and `BaseExtendedMMM.load` also
  reads the serializer's `trace.nc.gz` so cross-flavor loads never silently
  drop the posterior. `build_and_fit` auto-save now succeeds for every
  extension fit (E2E-tested). Tests: `tests/test_serialization_extended.py`.
- Still deferred: FE polish beyond the gate fix (the widget doesn't surface
  the structural-only keys; the DAG canvas can't draw mediator→mediator
  edges).

## 7. Deferred (documented non-goals for V1)

- Geo panels (extension models are national single-series today).
- ~~Centered state parameterization~~ **SHIPPED** (the review's predicted failure regime
  showed up immediately in the brand-funnel recovery: non-centered innovations under the
  dense weekly tracker → R-hat 1.76): `MediatorSpec.state_parameterization ∈
  {auto, centered, non_centered}`, default `auto` = centered when the mediator is measured
  on ≥ 50% of weeks (a dense tracker pins `z_t`, and the non-centered form funnels sigma
  against every innovation), non-centered when sparse. Centered samples the AR noise
  directly (`pm.AR` / `pm.GaussianRandomWalk`, zero-history init `N(0, σ)`) and adds the
  deterministic `D(ρ)@drivers` accumulation on top — the same model, different geometry.
- **Steady-state AR initialization** (review): the state starts its media stock at zero, a
  ~1/(1−ρ)-week burn-in that under-attributes early media; windowed summaries of the first
  ⌈log(0.05)/log(ρ)⌉ weeks should be treated as burn-in until a stationary-init option
  lands.
- **Measurement-density-scaled innovations** (review): a sparsely observed mediator (e.g.
  quarterly Likert, 8 points over 104 weeks) still buys n_obs weekly innovations; the
  recovery suite should watch per-mediator prior→posterior contraction on σ_m.
- Window-aware nonlinear experiment calibration ((1−ρ^W)/(1−ρ) gains + sigmoid slopes as a
  handle-term list, or an exact counterfactual estimand node).
- Beta-binomial / Dirichlet-multinomial overdispersion options (the iid logit-noise slack
  in §4.3 covers the first-order need); per-channel/per-edge saturation configs;
  time-varying design effects.
- Reporting: routed to `ExtendedMMMExtractor` (routing added in
  `reporting/extractors/__init__.py`); a bespoke funnel section is future work.
- Serialization is inherited from `BaseExtendedMMM.save/load` (cloudpickle + NetCDF).
