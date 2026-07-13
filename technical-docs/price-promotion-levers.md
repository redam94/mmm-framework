# Price & promotion levers — #138

For CPG / retail / DTC, price and promotion are among the largest sales drivers
and are *decisions the client controls* — planners want an **elasticity** and a
**promo ROI**, not a nuisance control coefficient. These levers promote a control
column to a first-class term with its own transform and prior.

## Usage

```python
from mmm_framework.config import PriceConfig, PromoConfig

model_config = (
    ModelConfigBuilder()
    .with_price(PriceConfig(variable="Price", reference="median"))
    .with_promotions(PromoConfig(variable="Promo", adstock_lmax=4))
    .build()
)
```

The named columns are **removed from the linear control block** (so they are not
double-counted) and routed through:

- **Price** — `beta_price · log(price / reference)`, with a **sign guard** so the
  elasticity is `≤ 0` (a price rise cannot raise demand). `reference` is the
  regular price the model responds to the gap from: a float, or `"mean"` /
  `"median"` / `"max"` from the data (median ≈ regular price), so the model reads
  *discount depth*, not absolute price. `None` uses `log(price)` (the constant is
  absorbed by the intercept). Emits `price_elasticity`.
- **Promotion** — `beta_promo · adstock(promo)`, a lift with its **own geometric
  carryover** (learnable `promo_alpha`, `adstock_lmax` weeks; `<= 1` ⇒ no
  carryover), distinct from an instantaneous linear bump. Works for a continuous
  discount % or a 0/1 event flag; `allow_negative` permits a pull-forward that
  later depresses sales. Emits `beta_promo_<var>`.

Off by default (`price=None`, `promotions=[]`) — a price/promo column then enters
(if present) as an ordinary linear control, byte-identical to today.

## Reporting

`price_elasticity` (with HDI), `price_component`, `promo_component` and the
combined `lever_component` are registered deterministics; the decomposition
carries `levers` / `total_levers`, so the waterfall closes and price+promo appear
as a **"Price & Promotion"** line (summary + report component totals), separate
from the media and control blocks.

## Elasticity interpretation

`price_elasticity` is the coefficient on `log(price/reference)`. Under the
**multiplicative** specification (`.multiplicative()`, log KPI) it is a genuine
constant elasticity — a 1% price rise moves sales by `elasticity` %. Under the
default **additive** model it is the coefficient on the standardized KPI (a
semi-elasticity), still correctly signed and correct in the decomposition; pair
the price lever with the multiplicative form for a clean elasticity number.

## Identification assumption — price endogeneity

**Price is very often endogenous.** A retailer cuts price *because* demand is
soft (or raises it into strong demand), and promotions are timed to seasonal
peaks — so `price` / `promo` correlate with the very demand shocks in the error
term, and the naive elasticity is biased (typically toward zero or the wrong
sign). This lever estimates the *conditional* association; it does not
manufacture an instrument. Treat the elasticity as identified only to the extent
that price/promo variation is **exogenous** given the controls, and cross-check
with the endogeneity diagnostics (`diagnostics/endogeneity.py`, #110) and,
ideally, a price/promo experiment. The sign guard (`elasticity ≤ 0`) and the
reference-gap parameterization reduce, but do not remove, the bias.

## Tests

`tests/test_price_promo_levers.py` — off-is-linear-control, on-promotes +
excludes-from-controls, unknown-lever warning, negative-elasticity +
positive-promo recovery, decomposition closure + reporting separation.
