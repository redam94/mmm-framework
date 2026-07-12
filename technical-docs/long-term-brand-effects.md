# Short-term vs long-term (brand) effects

*Issue #106 — the CMO lens, finding C2.*

## The problem

A weekly marketing-mix model captures **activation** well but **under-captures
long-term brand effects**. Left unaddressed, this is the single most
consequential way an MMM misleads a CMO: it over-rotates budget to lower-funnel
performance channels and starves brand-building. A CMO is accountable for brand
equity, not just this quarter's sales — so a model that silently prices brand at
zero beyond a few weeks of adstock is not just incomplete, it is *directionally
biased against brand*.

This document states plainly what the framework's weekly model **does** and
**does not** measure, the method it uses to surface the estimable part of the
split, the honest caveat where it can't, and the data required to do better.

## What a weekly MMM measures

Each channel's fitted effect decomposes into:

1. **Activation (immediate).** The response that lands in the same week the spend
   runs — the week-0 adstock weight.
2. **Carryover.** The adstock tail: the response that persists over the next few
   weeks, `Σ` of the lag weights beyond week 0, capped at the adstock window
   `l_max`.

Both are **short-term to medium-term**. Adstock half-lives in practice are a few
weeks; even a generous `l_max` rarely exceeds a quarter. This is the part the
framework can and does report — see *the estimable split* below.

## What it does **not** measure

**Long-term brand equity.** The multi-quarter to multi-year lift that
brand-building creates through mental availability, distinctiveness, pricing
power, and base-demand growth. This decays on a horizon (12–36+ months) far
longer than any adstock window, and a weekly model fit on ~1–2 years of data has
neither the span nor the functional form to see it. Its contribution is absorbed
into the intercept / trend / baseline, i.e. it is **credited to "base," not to
the channel that built it.**

Consequence: brand-heavy channels (TV, video, sponsorship, OOH) are
systematically **under-credited**, and a naive read of the channel ROIs will
recommend cutting them. **Do not conclude a brand channel is inefficient from
this model alone.**

## The estimable split (what the report shows)

`reporting/helpers/longterm.py::carryover_split` decomposes each channel's
*measured* effect into immediate vs carryover from the fitted adstock decay
weights:

```
immediate_pct = weight[0] / Σ weights
carryover_pct = 1 − immediate_pct
effective_weeks = # lags carrying ≥ 1% of the weight
```

`build_long_term_facts` assembles the per-channel split (and, when per-channel
contributions are available, the KPI-unit split and a portfolio blend) into
`bundle.long_term`. The **`LongTermSection`** renders it under a prominent caveat
that this is a *within-window* split, **not** brand equity.

This is an honest, model-derived number — but it answers "how long does the
*measured* effect keep working?", **not** "how big is the brand effect?".

## The long-term scenario (an assumption, never an estimate)

When `ReportConfig.long_term_multiplier` is set (e.g. `2.0`), the section adds a
clearly-labelled **scenario**:

```
total effect ≈ measured short-term effect × multiplier
```

The default `2.0` is a round midpoint from published brand meta-analyses (Binet
& Field; Nielsen / Analytic Partners long-term studies commonly find total
effects ~1.5–2× the short-term measured effect for brand-heavy channels). It is
an **external assumption the user should replace with their own evidence** — the
report labels it as such and never presents it as a model output.

## The structural-funnel path (partial long-term capture)

When the model is a **survey/brand funnel**
(`mmm_extensions/models/structural.py` — awareness → consideration → sales, fed
by brand-tracker surveys), part of the longer-horizon brand path *is* modelled:
media that moves awareness, which later moves sales, is credited to the channel
rather than to base. The section detects this (`has_structural_funnel`) and
softens the caveat to "treat the split as a lower bound." Effects beyond the
funnel's own measurement window are still unmeasured.

## Data required to measure long-term properly

| Data | What it unlocks |
|---|---|
| **Long history (2–3+ years)** | Separating slow, brand-driven base-demand trends from noise. |
| **Brand-tracker surveys** (awareness / consideration / preference) | The structural funnel — routing media through a *measured* brand path instead of only same-quarter sales. |
| **Long-window experiments** | Geo lift tests read for 2–4 quarters after the flight capture persistence a two-week readout misses; fold them in via experiment calibration. |
| **A brand-equity latent / documented multiplier** | An explicit long-term term calibrated to the above — used as a *stated* assumption, never a silent one. |

## Summary of the framework's position

- The weekly model reports the **immediate vs carryover** split it can estimate,
  and says so.
- It states **plainly** that long-term brand equity beyond the adstock window is
  **not measured**, and that this under-credits brand.
- It offers an **opt-in, clearly-labelled** long-term-multiplier scenario for
  planning conversations.
- It lists the **data** required to measure long-term effects for real.

Honesty over false precision: better to name the gap and its direction than to
let a silent zero masquerade as "brand doesn't work."

*Code:* `reporting/helpers/longterm.py`, `reporting/sections.py::LongTermSection`.
*Tests:* `tests/reporting/test_long_term_section.py`.
