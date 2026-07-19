# LTV / CLV Modeling — Bayesian BG/NBD + Gamma-Gamma (Phases 4–5)

> Status: implementation spec. LTV/CLV is greenfield (confirmed: zero existing
> code; every "retention" in the repo is media adstock, not customer retention).
> It lands as a **non-MMM garden family** (`__garden_model_kind__ = "clv"`),
> exactly like the existing CFA/LCA examples, inheriting `fit` / serialization /
> the estimand engine / report routing unchanged.
>
> **Phase 4** = the standalone transaction-level LTV model. **Phase 5** = wiring
> LTV-as-outcome into the experiment/reallocation loop.

## 0. Model choice

The user has **transaction-level (per-customer) data** → the canonical
non-contractual "buy-till-you-die" pair:

- **BG/NBD** (Fader–Hardie–Lee 2005) — models purchase *frequency* and *dropout*:
  each customer has a latent purchase rate `λ` (Gamma-distributed across the
  population, shape `r`, rate `α`) and a latent dropout probability `p`
  (Beta-distributed, `a`, `b`). Yields `E[# transactions | horizon]` and
  `P(alive)`.
- **Gamma-Gamma** (Fader–Hardie 2013) — models *monetary value*: per-customer
  average spend `E[M]` from observed frequency + monetary mean, independent of
  frequency (a checkable assumption).
- **CLV** = discounted `E[transactions] · E[monetary value]` over a horizon.

Bayesian (not MLE like `lifetimes`) is the right call here — it gives full
posterior uncertainty on CLV that flows into the estimand engine, reports, and
(Phase 5) the experiment economics, consistent with the rest of the framework.
BG/NBD's likelihood is a **closed form**, so it enters PyMC via `pm.Potential`
(or `pm.CustomDist`) with no discrete latents → NUTS-able.

## 1. Data contract — RFM

Transaction data must be reduced to **per-customer RFM summaries** before the
model. New preprocessing module `src/mmm_framework/ltv/preprocess.py`:

```python
def transactions_to_rfm(
    df, *, customer_col, date_col, value_col=None,
    observation_end=None, freq="W",
) -> pd.DataFrame:
    """Collapse a transaction log to one row per customer with:
      frequency  = # repeat purchases (transactions − 1, first is acquisition)
      recency    = time of last purchase − time of first purchase (in `freq` units)
      T          = observation_end − time of first purchase (customer age)
      monetary   = mean value of REPEAT transactions (value_col; Gamma-Gamma input)
      n_txn      = raw transaction count
    """
```

RFM is a **per-entity wide table** (one row = one customer), not the MFF time
panel. The LTV garden model reads it through a role-tagged path:

- MVP: `model_params.rfm_columns = {"frequency": ..., "recency": ..., "T": ...,
  "monetary": ...}` (by-name mapping, like `latent_factor_mmm`'s
  `indicator_columns` fallback). `_prepare_data` pulls those arrays off
  `self.dataset.observed()` / the raw frame and calls `_set_non_mmm_defaults()`
  after setting `self.n_obs = n_customers`.
- The dataset reaches the constructor as the standard `panel`; for LTV each
  "period" is a customer. `has_geo/has_product` = False. No y-standardization
  (counts + monetary stay natural scale — like the non-Gaussian likelihood path).

Optional grouping columns (acquisition **channel**, **cohort** month) are carried
as per-customer categorical codes for Phase-5 segment CLV.

## 2. The model — `examples/garden_models/bayesian_clv.py`

Mirrors `bayesian_cfa.py` / `bayesian_lca.py`. Overrides only `_prepare_data`
and `_build_model`.

```python
class BayesianCLV(CustomMMM):
    __garden_model_kind__ = "clv"
    CONFIG_SCHEMA = CLVConfig          # horizon_periods, discount_rate, priors,
                                       # monetary_model on/off, freq
    DEFAULT_ESTIMANDS = [
        latent_scalar("mean_clv", var="mean_clv"),
        latent_scalar("total_clv", var="total_clv"),
        latent_scalar("mean_expected_purchases", var="mean_expected_purchases"),
        latent_scalar("mean_p_alive", var="mean_p_alive"),
    ]

    def _prepare_data(self): ...       # RFM arrays + _set_non_mmm_defaults()
    def _build_model(self): ...        # BG/NBD Potential + Gamma-Gamma + deterministics
    def customer_value_summary(self): ...  # per-customer / per-segment table (report)
```

### 2.1 BG/NBD block

Population heterogeneity priors:
- `r ~ HalfNormal`/`Weibull`, `alpha ~ HalfNormal` — Gamma purchase-rate mixing.
- `a, b ~ HalfNormal` — Beta dropout mixing.

Likelihood via `pm.Potential("bgnbd_ll", bgnbd_loglik(r, alpha, a, b, x=freq,
t_x=recency, T=T))` where `bgnbd_loglik` is the closed-form BG/NBD individual
log-likelihood (pytensor ops: `pt.gammaln`, `pt.log`, `logaddexp` for the
two-term dropout mixture — numerically stabilized). This is the standard FHL
expression; unit-tested against a reference implementation (`lifetimes`) at fixed
params.

Per-customer deterministics (the quantities of interest):
- `expected_purchases` = `E[# transactions in (0, horizon] | x, t_x, T]` (the FHL
  conditional-expectation formula).
- `p_alive` = `P(alive | x, t_x, T)`.

### 2.2 Gamma-Gamma block (monetary)

- Priors `p, q, gamma ~ HalfNormal`.
- Likelihood on repeat-buyers' observed `monetary` given `frequency` via the
  Gamma-Gamma `pm.Potential`.
- `expected_avg_value` = conditional expected monetary value per customer.

### 2.3 CLV + aggregate deterministics

- `clv_per_customer = Σ_{t<horizon} discount^t · (expected_purchases_rate_t ·
  expected_avg_value)` — the discounted expected value; MVP uses the
  horizon-total `expected_purchases · expected_avg_value / (1+discount)^{h/2}`
  approximation, with a per-period refinement flag.
- Aggregate scalars the estimands read: `mean_clv`, `total_clv`,
  `mean_expected_purchases`, `mean_p_alive` — `pm.Deterministic` reductions over
  the customer axis (so `latent_scalar` summarizes them as mean+HDI, the
  established non-MMM estimand path).

## 3. Report

Rides the family-agnostic latent-structure report path: implement
`customer_value_summary()` (per-segment CLV table: count, mean CLV + HDI, mean
P(alive), expected purchases) so `has_latent_structure()` fires and the
`FactorAnalysisSection`/`FactorAnalysisExtractor` render it (they already
generalize to CFA loadings / LCA class profiles via
`bundle.latent_table_title`). Set the LTV titles ("Customer value",
"Segment CLV", "Value estimands"). No new report section needed.

## 4. Synthetic ground truth — `synth/dgp_clv.py`

Simulate a customer population with **known** `(r, alpha, a, b, p, q, gamma)`:
draw per-customer `λ`, `p`, spend params; simulate a transaction stream over a
calendar; split at an `observation_end` into calibration (fit) + holdout
(validation). Emit the transaction log + a JSON answer key (true population
params, true holdout transactions, true CLV). Recovery test asserts posterior
covers the planted params and that predicted holdout transactions track actual
(the standard BG/NBD calibration-vs-holdout validation).

## 5. Fit + serialize + agent (Phase 4 wiring)

- **Fit**: inherited. NUTS by default; MAP/ADVI available. BG/NBD posteriors can
  be ridge-y — the `sampling-failure-playbook` applies; note reparameterization
  (non-centered mixing) if divergences appear.
- **Serialize**: inherited (records `model_kind="clv"`).
- **Garden registration**: register `bayesian_clv` like the other examples;
  `seed_atelier_demo.py` can seed a CLV demo session (synthetic transactions →
  RFM → fit → CLV estimands + segment table).
- **Agent**: reachable through the existing garden flow (`list_garden_models` →
  `load_garden_model` → `fit_mmm_model` → `get_estimands`). A small
  `build_rfm_from_transactions` tool wraps `transactions_to_rfm` so a user can go
  from a raw transaction upload to a fitted LTV model in chat / Data Studio.

## 6. Phase 5 — LTV as an experiment/MMM outcome

> STATUS: SHIPPED (2026-07-19), with one deviation: `clv_to_cac` is a pure
> helper (`ltv/kpi.py`) + a `clv_value` model-op feature rather than a registry
> estimand — CAC is external data (spend ÷ new customers) the CLV model never
> sees, so an in-model estimand would have smuggled a constant into the
> posterior. Implemented surface: `CLVConfig.segment_column/segment_labels` →
> per-segment `segment_clv_<label>`/`segment_p_alive_<label>` deterministics +
> dynamic estimands + `segment_clv_means()`; `transactions_to_rfm(segment_col=…)`
> carry-through; `make_clv_world(channels=…)` planted heterogeneity;
> `ltv/kpi.py::new_customer_clv_series` (cohort CLV KPI) + `clv_to_cac`;
> model-op `clv_value` (+ optional `segment`/`cac`); agent tools `get_clv_value`
> and `ghost_ads_power_calc(value_from_clv=True, clv_segment=…)`. For net
> economics, an acquisition experiment passes the CLV as `margin_per_kpi` with
> KPI = conversions (a usage pattern, no code change). Tests:
> `tests/test_ltv_loop.py`.

Once the standalone model exists, wire long-term value into the loop so
acquisition experiments and reallocation are valued on **CLV, not first
purchase**:

1. **Per-channel / per-cohort CLV** — the model already carries per-customer
   acquisition-channel codes (§1); add `segment_clv` deterministics (mean CLV by
   acquisition channel) → posterior CLV *per acquisition channel*.
2. **Ghost ads / experiments valued on CLV** — feed `segment_clv` posterior as
   `value_per_conversion` in the Phase-2 ghost-ads calculator and as the
   `value_per_kpi` in Phase-3 net economics, so an acquisition test's
   reallocation gain reflects lifetime value. A channel that acquires
   low-frequency one-time buyers is correctly discounted vs one acquiring
   high-CLV customers even at equal CPA.
3. **CLV as an MMM KPI** — a cohort-level CLV series (new-customer CLV by week)
   can be the KPI an MMM/experiment targets; the measurement resolver
   (`reporting/helpers/measurement.py`) treats it as a monetary KPI with
   `margin_per_kpi=1`. This closes "spend → acquisitions → lifetime value".
4. **Blended acquisition economics** — reallocation optimizer weighs channels by
   `CLV − CAC` (from `segment_clv` − channel cost) instead of short-term ROAS,
   surfaced as a new estimand `clv_to_cac`.

Phase 5 is deliberately staged after Phase 4 (the user chose "both, phased"): the
standalone model is validated on known ground truth first, then the connective
estimands + economics inputs are added.

## 7. Wiring summary

- **`src/mmm_framework/ltv/`** — new: `preprocess.py` (`transactions_to_rfm`),
  `likelihood.py` (`bgnbd_loglik`, `gamma_gamma_loglik` pytensor closed forms),
  `__init__.py`.
- **`examples/garden_models/bayesian_clv.py`** — the model + `CLVConfig` +
  `customer_value_summary`.
- **`synth/dgp_clv.py`** — known-truth transaction simulator + answer key.
- **`agents/tools.py`** — `build_rfm_from_transactions` tool.
- **Phase 5**: `segment_clv` deterministics; `clv_to_cac` estimand
  (`estimands/registry.py`); `value_per_conversion` hookup in ghost-ads +
  net-economics; measurement-resolver CLV KPI.
- **Docs** — a `docs/ltv-modeling.html` guide + link from Model Garden docs.

## 8. Build order (Phase 4)

1. `ltv/preprocess.py` `transactions_to_rfm` + tests (RFM correctness).
2. `ltv/likelihood.py` BG/NBD + Gamma-Gamma closed forms, unit-tested vs a
   reference at fixed params (numerical correctness before any sampling).
3. `synth/dgp_clv.py` known-truth simulator + answer key.
4. `examples/garden_models/bayesian_clv.py` (`_prepare_data`, `_build_model`,
   deterministics, `DEFAULT_ESTIMANDS`, `customer_value_summary`).
5. Recovery test (`tests/test_clv_garden_model.py`): posterior covers planted
   params; holdout transactions tracked; estimands finite; report renders.
6. Garden registration + demo seed + agent RFM tool + docs guide.

## 9. Tests

- `tests/test_ltv_preprocess.py` — RFM matches a hand-computed fixture; edge
  cases (single-purchase customers → frequency 0; `observation_end` handling;
  weekly vs daily `freq`).
- `tests/test_ltv_likelihood.py` — `bgnbd_loglik`/`gamma_gamma_loglik` match a
  reference to tolerance at fixed params; numerically finite at frequency 0 and
  large `T`.
- `tests/test_clv_garden_model.py` — end-to-end recovery on `dgp_clv`; estimand
  round-trip; serialization; `customer_value_summary` shape; report render (the
  family-agnostic latent section).
