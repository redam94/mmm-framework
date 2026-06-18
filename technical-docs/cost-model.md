# Per-Fit Cost Model & Margin Guardrails — MMM Framework

> **READ THIS FIRST — what is fact and what is a knob.**
>
> - **Measured (hard facts).** Fit *times* in this doc are measured on real runs:
>   a national NUTS fit completes in **~17 seconds**, and **NumPyro is 6.0× faster
>   than stock PyMC at equal draws**. These are the only benchmark figures used;
>   nothing here invents a benchmark.
> - **Assumption (replace before you bill anyone).** Every **dollar instance
>   price** is a clearly-labeled *representative* assumption — `~$0.10–$0.50 per
>   vCPU-hour depending on instance type, region, and commitment`. The deployer
>   **must** substitute their real cloud rates. The arithmetic is what matters;
>   the inputs are yours to set.
> - **Proposed / illustrative.** Every **subscription price floor** below is a
>   *proposed* number derived from the cost model — not a final price. The founder
>   sets finals. Treat these as the *lower bound that protects margin*, not a price
>   sheet.
> - **Source of truth for limits/features.** Tier seat/project/fit quotas and
>   feature gates come from `src/mmm_framework/auth/plans.py` (the enforceable
>   contract). This doc must not contradict it; if they diverge, `plans.py` wins.

This is the Track-3 "Per-fit cost model (PyMC compute is the COGS driver) → margin
guardrails" deliverable from `technical-docs/business-readiness-p0.md`.

---

## 0. The deployment model sets the frame

The platform is **open-source and self-hosted by default.** In the single-tenant
self-hosted posture (auth off) there are **no plan limits at all** — that is the
"open" in open-core, and it has **no COGS to us** because the operator runs their
own compute. See `plans.py:50-62`.

The cost model in this doc applies only to the **hosted, multi-tenant managed
service** (`MMM_AUTH_ENABLED=1` / `MMM_AGENT_HOSTED=1`), where *we* run the
compute and **hosted compute is the COGS driver.** Plan limits (seats, projects,
fits/month) are enforced only in that mode.

---

## 1. Cost of a single national fit

### 1.1 Inputs

| Input | Value | Status |
|---|---|---|
| National NUTS fit wall-clock | **~17 s** | **Measured** |
| NumPyro speedup vs stock PyMC (equal draws) | **6.0×** | **Measured** |
| vCPUs per fit (4 NUTS chains in parallel) | **4 vCPU** | Assumption (representative) |
| Cloud compute price | **~$0.10–$0.50 / vCPU-hr** | **Assumption — replace with your rate** |
| Working midpoint used below | **$0.30 / vCPU-hr** | Assumption |
| Job-envelope overhead multiplier | **2×** | Assumption (see §1.3) |

> The **17 s figure already assumes the fast sampler path.** Stock PyMC at equal
> draws would be ~6.0× slower, i.e. on the order of ~100 s — which is exactly why
> the managed service should pin the NumPyro/NutPie path. Pricing a tier off the
> stock-PyMC time would inflate COGS ~6× for no benefit. **Use the 17 s number.**

### 1.2 Raw compute cost per fit

```
vCPU-hours per fit = 17 s × 4 vCPU ÷ 3600 = 0.0189 vCPU-hr
```

| Cloud price (assumed) | Raw compute / fit |
|---|---|
| $0.10 / vCPU-hr (low)  | **$0.0019** |
| $0.30 / vCPU-hr (mid)  | **$0.0057** |
| $0.50 / vCPU-hr (high) | **$0.0094** |

A bare national fit costs **fractions of a cent.** Compute alone is not what makes
or breaks a tier — the *envelope around the fit* and the *fixed overhead* do.

### 1.3 The job envelope (why we apply a 2× multiplier)

A billed "fit" is rarely just the sampler. The measured robustness profile
(`technical-docs/mmm-robustness-report.md`) shows the work *around* a fit often
costs **more than the fit itself**: a scored scenario is ≈ fit (17 s) + causal
refutation (~25 s, 4 refits) + post-fit ops (~2 s). We do **not** assume every
billed fit runs the full refutation suite, but we do budget for warm-up,
posterior write/serialization, typical post-fit ops, and warm-pool idle. We fold
that into a **2× envelope multiplier** on raw compute:

| Cloud price (assumed) | Envelope cost / fit (2×) |
|---|---|
| $0.10 / vCPU-hr | **$0.0038** |
| $0.30 / vCPU-hr | **$0.0113** |
| $0.50 / vCPU-hr | **$0.0189** |

**Working per-fit COGS (mid): ≈ $0.011.** This is the number that drives the
per-tier tables. If your real workload skips refutation, your COGS is lower; if it
runs the full suite on every fit, raise the multiplier (see §3).

---

## 2. Per-tier monthly COGS & price floor

Fit quotas and seats are taken **verbatim from `plans.py`** (the enforceable
contract). Fixed overhead (object storage for saved posteriors/models + Redis/ARQ
job queue + metering) is a **representative assumption** that scales loosely with
tier size — replace with your real platform bill.

**Target gross margin: 80%.** The price floor that yields ≥80% margin is
`floor = COGS ÷ (1 − 0.80) = COGS × 5`.

### 2.1 At the mid assumption ($0.30/vCPU-hr, 2× envelope, $0.011/fit)

| Tier (`plans.py`) | Fits/mo | Compute COGS | Storage+Redis (assumed) | **Total COGS/mo** | **80%-margin price floor/mo** | Per-seat floor |
|---|---:|---:|---:|---:|---:|---:|
| **Free** (2 seats, 3 proj)      | 20   | $0.23  | $0.50  | **$0.73**  | **$3.63**   | $1.82 (÷2) |
| **Team** (5 seats, 10 proj)     | 200  | $2.27  | $3.00  | **$5.27**  | **$26.33**  | $5.27 (÷5) |
| **Business** (25 seats, 50 proj)| 1000 | $11.33 | $12.00 | **$23.33** | **$116.67** | $4.67 (÷25) |
| **Enterprise** (unlimited)      | —    | metered | custom | **per contract** | **custom (cost-plus / committed-use)** | — |

### 2.2 At the high assumption ($0.50/vCPU-hr, 2× envelope, $0.019/fit)

This is the conservative ceiling — what your COGS looks like on a pricier
instance/region. **Price floors should clear *this* line, not the mid line.**

| Tier | Fits/mo | **Total COGS/mo (high)** | **80%-margin price floor/mo** |
|---|---:|---:|---:|
| **Free**     | 20   | **$0.88**  | **$4.39**   |
| **Team**     | 200  | **$6.78**  | **$33.89**  |
| **Business** | 1000 | **$30.89** | **$154.44** |

> **Reading the table.** *Free* is essentially a loss-leader funded by overhead,
> not compute — its real cost is the storage/Redis floor and a seat of support
> attention, not the 20 fits (~$0.23 of compute). *Business* at full quota is the
> only tier where compute is a material line item, and even there it is ~$11–31/mo.
> **The dominant COGS at small scale is fixed platform overhead, not the sampler.**

---

## 3. Sensitivity — what makes a fit cost N× more

The 17 s anchor is a **national** base model. Real COGS varies with model class and
data shape. These multipliers are directional (from measured scaling in
`mmm-robustness-report.md` where noted) and should be confirmed against your own
telemetry:

| Driver | Effect on per-fit cost | Basis |
|---|---|---|
| **Channel count** | **Super-linear.** 4 ch → ~17 s, 8 ch → ~45 s (**≈2.6×**). | Measured (`mmm-robustness-report.md:25`) |
| **Time length** | **~Linear, gentle.** 104 wk → ~12 s, 260 wk → ~21 s. | Measured (same) |
| **Geo / geo×product panels** | Scales with **weeks × geos** — a multi-geo panel is the biggest single cost multiplier; budget **several× to 10×+** a national fit. | Measured scaling shape |
| **Extension models** (Nested / Multivariate / Combined) | **Materially more** per fit — more parameters, more graph, more sampling. Treat as **2–N×** and meter separately. | Architectural (`mmm_extensions/`) |
| **Full causal refutation suite** | **+~25 s** per scored fit (4 refits) — **more than the fit itself.** | Measured (`:117`) |
| **Per-channel counterfactual/marginal ops** | `O(n_channels)` posterior-predictive passes — cheap per pass, expensive on wide channel sets. | Measured (`:117`) |
| **Warm-pool vs cold-start** | Warm pool removes interpreter/import/compile startup but adds **idle vCPU-hr** you pay for between fits; cold-start pays import+compile **per fit** (seconds of latency) but **zero idle**. Low-utilization tiers (Free) favor cold-start economics; high-throughput tiers (Business) favor a warm pool. | Operational |

**Implication for pricing:** the quotas in `plans.py` count *fits*, not *fit-cost*.
A Business org running 1000 **geo-panel + extension** fits costs far more than 1000
national fits. Two ways to keep margin honest:

1. **Weight the meter.** Charge geo/extension fits as multiple "fit units" against
   the quota (the meter already records `run_metrics` rows per fit; attach a
   cost-weight). This keeps the published quota simple while protecting COGS.
2. **Cap heavy work.** Gate the full refutation suite / wide counterfactual sweeps
   behind a higher tier or an explicit opt-in, so they don't silently 10× a fit.

---

## 4. The margin guardrail

> **Guardrail (proposed, founder sets finals): do not price a tier below its
> 80%-margin floor at the high cost assumption.**
>
> | Tier | **Floor — don't price below** | At quota |
> |---|---|---|
> | **Free** | $0/mo is fine *only* because overhead is small | 20 fits/mo |
> | **Team** | **≥ ~$34/mo** (covers $30/mo round number comfortably; price it at $49–99 for real margin + headroom) | 200 fits/mo |
> | **Business** | **≥ ~$155/mo** (price it at $299–499+ for margin + the SSO/audit/support value, which is worth far more than its compute) | 1000 fits/mo |
> | **Enterprise** | Cost-plus / committed-use; unlimited fits ⇒ **must** meter and cost-weight (§3). Never offer flat-rate unlimited geo/extension fits. | unlimited |

Three hard rules behind the numbers:

1. **Free is funded by overhead, not compute.** It can be $0 because its 20 fits
   cost ~$0.23. The risk on Free is **seats × support attention** and **storage
   accumulation**, not the sampler. Keep the 2-seat / 3-project caps from
   `plans.py` — they are the real cost control on Free.
2. **Team and Business margin comes from features, not fits.** Compute is $5–31/mo
   at quota. The price is justified by `multi_tenant`, `hosted_sandbox`, `branding`
   (Team) and `+sso`, `+audit_export`, `+priority_support` (Business) — see
   `plans.py:63-89`. **Never let a feature-rich tier be priced as if compute were
   the only cost**; that leaves money on the table *and* under-prices support.
3. **Unlimited fits is only safe when metered + cost-weighted.** Enterprise's
   `monthly_fit_quota=None` (`plans.py:96`) is fine **only** because Enterprise is
   contracted cost-plus. A flat-rate "unlimited fits" consumer tier is the one way
   to invert margin — a single customer running geo-panel + extension + refutation
   fits in a loop costs 10–100× a national fit. **Don't ship flat-rate unlimited.**

### 4.1 Break-even sanity check

At the **mid** assumption, one Team subscription priced at the **$26/mo floor**
exactly covers its COGS at 80% margin; priced at a realistic **$49/mo** it carries
~89% gross margin even before accounting for the fact that most orgs use a fraction
of their fit quota. The model is structurally **high-margin** because the compute
unit (a 17 s fit) is genuinely cheap — *provided* heavy models are metered (§3) and
fixed overhead is contained.

---

## 5. How to re-run this with your real numbers

Everything above is arithmetic over six inputs. To re-baseline:

1. Replace the **vCPU price** ($0.10–$0.50 assumption) with your committed cloud
   rate.
2. Replace **vCPUs/fit** (4) if you change chain parallelism.
3. Replace the **envelope multiplier** (2×) with your measured fit-to-billed-job
   ratio — i.e. how much refutation/ops you actually run per billed fit.
4. Replace **storage + Redis overhead** with your real platform bill per tier.
5. Keep the **fit-time facts** (17 s, 6.0× NumPyro) unless you re-benchmark — and
   if you do, **measure**, don't estimate.
6. Apply **cost-weights** (§3) for geo-panel and extension fits before reading the
   quota as a cost.

Per-fit COGS = `fit_seconds × vCPUs ÷ 3600 × $/vCPU-hr × envelope_multiplier`.
Per-tier COGS = `per_fit_COGS × monthly_fit_quota + fixed_overhead`.
80%-margin floor = `per_tier_COGS × 5`.

---

*All subscription dollar figures here are **proposed / illustrative** floors derived
from the cost model — the founder sets final prices. All cloud instance prices are
**representative assumptions** to be replaced with the deployer's real rates. The
fit-time benchmarks (17 s, 6.0× NumPyro) are measured. Tier limits and feature
gates are authoritative in `src/mmm_framework/auth/plans.py`.*
