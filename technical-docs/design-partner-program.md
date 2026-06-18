# Design Partner Program — MMM Framework

> **Status: FOUNDER-SENDABLE OFFER (draft you can paste into an email).** This is
> the playbook for landing the first **1–3 reference customers**. It describes a
> real, deliverable engagement built on capabilities that exist in the codebase
> today (the T₀–T₅ measurement loop, experiment economics, the causal refutation
> suite, in-graph calibration, the analyst agent). It also tells the truth about
> what we have and have not proven yet — because for this product **credibility is
> the pitch**, and a design partner who feels oversold churns and tells people.
>
> **HONESTY GUARDRAILS — read before sending anything (§6 is the long version):**
> - **There is no real customer yet.** The case-study artifact in §7 is a clearly
>   marked **TEMPLATE with `[PLACEHOLDERS]`** — never send it as if it describes a
>   real client, and never invent company names or numbers.
> - **The headline 3.0% MAPE backtest is on a SYNTHETIC benchmark world, not a
>   client.** Quote it as a methodology stress-test, not as "results we got for a
>   customer."
> - **Our one genuinely-external real dataset is Lydia Pinkham** — a *causal-anchor
>   win* (ROAS inside the published-literature bracket) but a *forecasting non-win*
>   (no skill over a naive baseline, honestly calibrated). We lead with this on
>   purpose: it's proof we report no-skill honestly. See §6.
> - **SOC 2 is readiness, not certification** (`technical-docs/soc2-readiness.md`).
>   Self-hosted single-tenant is the easy default for a design partner who can't
>   send data anywhere.

---

## 1. Who is an ideal design partner

We want **one to three** partners, chosen for fit, not logo size. The profile:

**Company shape**
- A **mid-market brand** (roughly $5M–$250M revenue) running paid media across **a
  few channels** (e.g., paid search, paid social, some combination of CTV / display
  / affiliate / direct mail), **or**
- A **performance / media agency** that owns a brand's allocation and will bring one
  client account into the engagement.

**Data they already have** (this is the hard gate)
- **≥ ~12 months of weekly data** — the more the better; 18–24 months lets us see
  seasonality cleanly. Weekly cadence is the sweet spot for the framework.
- **Spend by channel** (the same channels every week) **and a KPI** — revenue, units,
  sign-ups, or another outcome they actually optimize toward. The framework is
  KPI-kind-aware (revenue / units / other), so a non-revenue KPI is fine.
- Bonus, not required: **geo-level** spend+KPI (DMA / region / store), which unlocks
  matched-market geo-lift experiments. National-only data is fine — we fall back to
  budget-neutral randomized flighting.

**A real decision on the table**
- A **pending budget-allocation decision** with a deadline: an annual/quarterly plan,
  a "should we keep funding channel X?" question, a "where does the next incremental
  $250k go?" question. The loop is built to *inform a decision*, not to produce a
  vanity dashboard. No pending decision → no urgency → a bad design partner.

**A human who will engage**
- One named **point of contact** (analyst, growth lead, or founder) who will
  (a) get us the data under NDA, (b) take a **~bi-weekly** feedback call, and
  (c) be willing — *if and only if the results land* — to be a reference.

**Disqualifiers (be honest with yourself):**
- < 12 months of data, or channels that change every quarter (nothing stable to fit).
- No decision pending → this is a science project, not a design partner.
- Data they legally cannot share even under NDA and even self-hosted → revisit later.
- Someone who wants a guaranteed lift number up front. We don't sell guaranteed lifts;
  we sell an honest measurement loop (see §6).

---

## 2. What the partner GETS

A **white-glove, founder-led engagement** for **8–12 weeks** at **no cost or a steep
discount** (founder's call per partner; the deal is the data + feedback + reference,
not the dollars). Concretely:

1. **Managed access to the platform for the engagement.** Either we host it for them
   (managed tier capabilities — auth, isolation, branded reports) or we stand it up
   **self-hosted single-tenant** inside their environment so **no data ever leaves
   their network** (the in-process / subprocess posture; the fully-local `lmstudio`
   LLM option means even the agent can run without egress). For a data-sensitive
   partner, self-hosted is the unlock — see `technical-docs/soc2-readiness.md` §0.

2. **White-glove onboarding.** We do the data wiring (Master Flat File ingest),
   the EDA / data-quality pass (the `eda` module flags outliers, level-shifts, and
   noisy-proxy controls *before* we fit), and the first model fit **with them on the
   call**. A national fit completes in **~17 seconds** (measured), so onboarding is
   interactive, not a multi-day batch wait.

3. **A pre-registered measurement plan.** Before we look at any results, we write down
   what we're going to measure and how — the pre-registration is a first-class object
   in the product (`preregister_experiment`), which is what keeps this **causal and
   defensible** rather than a data-dredge. This is the methodological backbone: we
   reduce researcher degrees of freedom on purpose.

4. **EIG/EVOI dollar-priorities.** The fitted model tells us, in **dollars**, which
   uncertainty is worth resolving — Expected Information Gain and Expected Value of
   Information per candidate experiment (`compute_experiment_priorities`). This is how
   we decide *what to test first* instead of guessing.

5. **One experiment, designed and read out.** We design a runnable experiment
   (`design_experiment_plan` / `suggest_experiment`) — a **matched-market geo lift**
   or **matched-market DiD** if they have geo data, or **budget-neutral randomized
   flighting** if they're national-only — with the **experiment-economics** layer
   attached: an **incremental-ROAS estimate with a powered / underpowered / overpowered
   / inconclusive verdict**, the **opportunity cost** of running it (forgone KPI and
   spend-at-risk, with posterior uncertainty), and a **Pareto-front** of designs
   trading MDE × power × short-term cost × duration. We help them run it; we read it
   out.

6. **A calibrated refit.** The experiment readout is folded **back into the model**
   via **in-graph experiment calibration** (`apply_experiment_calibration` →
   `add_experiment_calibration`) — not bolted on as a side-note. The model literally
   updates its beliefs about the tested channel given the measured lift, including the
   **off-panel** case where the experiment ran in a window the model wasn't fit on.

7. **A defensible reallocation recommendation.** The deliverable is a **branded report**
   plus a calibrated recommendation: where to move budget, with **uncertainty bands**,
   anchored to the experiment — and the **4-test causal refutation suite** run against
   it so we can say *why* the recommendation should be trusted (or where it's fragile).
   A genuine no-change recommendation is also a valid, honest deliverable (see §6).

8. **Direct founder access** throughout. You're talking to the person building it.

---

## 3. What we ASK in return

Three things, in plain language:

1. **Data access under NDA.** ≥ 12 months of weekly spend-by-channel + KPI (and geo
   if they have it), shared under a mutual NDA. Self-hosted single-tenant is on the
   table specifically so a sensitive partner can say yes without data leaving their
   walls.

2. **~Bi-weekly feedback.** A standing **30–45 min call every ~2 weeks** for the
   engagement: what's confusing, what's missing, what they'd never use, what would make
   them pay. We want the product critique as much as the modeling outcome.

3. **A logo + a co-authored ROI case study + a reference call — *if the results land*.**
   This is explicitly **conditional on the engagement producing a defensible win they're
   proud of.** If it does: their logo on the site, a **co-authored** case study (they
   approve every number and every word), and willingness to take a **reference call**
   from a future prospect. If the result is "your spend is already near-optimal" or
   "you need a bigger experiment first," we don't manufacture a win — we'll still ask
   for a **process** testimonial ("they told us the truth and saved us from a bad
   reallocation"), which is honestly worth as much (§6).

We are **not** asking for: payment (or only a steep discount), a public commitment
before they've seen results, or an exclusivity / right-to-publish over their data.

---

## 4. The 8–12 week timeline (mapped to T₀–T₅)

The engagement *is* the platform's adaptive measurement loop. Each stage is a real
product capability, not a slide.

| Wk | Stage | What happens | Capability / evidence |
|----|-------|--------------|----------------------|
| **0** | Kickoff | NDA signed; access stood up (managed or self-hosted single-tenant); data handoff; onboarding brief captured (client / goals / KPIs / channels / constraints). | Self-hosted posture (`soc2-readiness.md` §0); project onboarding brief ingested into the project KB. |
| **1–2** | **T₀ — Fit** | MFF ingest + EDA/data-quality pass (outliers, level-shifts, noisy controls flagged *before* fitting); first **BayesianMMM** fit, on the call (**~17s national**, **6.0× NumPyro speedup** — measured). Baseline decomposition + ROI with **uncertainty bands**. | `eda` module; `BayesianMMM`; reporting. |
| **2–3** | **T₁ — Priorities** | **EIG/EVOI** dollar-ranked: which channel's uncertainty is worth the most to resolve. We pick the one experiment to run. | `compute_experiment_priorities` (`planning/eig.py`, `evoi.py`, `priority.py`). |
| **3–4** | **T₂ — Pre-register & design** | Write the **pre-registered measurement plan**; **design the experiment** (geo lift / matched-market DiD, or national flighting) with **experiment economics**: incremental-ROAS + power verdict, opportunity cost, Pareto-front design + cool-down. | `preregister_experiment`; `design_experiment_plan` / `suggest_experiment`; `planning/{design,opportunity_cost,simulation,experiment_optimizer}.py`. |
| **4–8** | **Run** | Partner runs the experiment in-market (duration is design-driven; flighting / geo holdout). We monitor; readout recorded. | `record_experiment_readout`; lifecycle registry draft→planned→running→completed. |
| **8–9** | **T₃ — Calibrated refit** | Fold the measured lift **back into the model** via **in-graph calibration** (incl. off-panel windows); refit; the model's beliefs about the tested channel update. | `apply_experiment_calibration` → `add_experiment_calibration`; off-panel path in `model/base.py`. |
| **9–10** | **T₄ — Reallocate** | The **defensible reallocation recommendation**: where budget should move, with bands, anchored to the experiment; the **4-test causal refutation suite** stress-tests it; branded report delivered. | Allocation + refutation suite; branded reporting. |
| **10–12** | **T₅ — Re-evaluate / wrap** | Review the recommendation together; flag what the *next* test should be (information decay → re-test); decide on the case study / reference (§3, §6); product feedback retro. | Re-test triggers; case-study template (§7). |

Realistically the **experiment run window (weeks 4–8) is the long pole** and depends on
their media cadence and how big a lift we need to detect — the Pareto-front design makes
that tradeoff (shorter test ⇄ bigger detectable effect) explicit and *their* choice. If
their decision deadline is tight, we can deliver a **fit + priorities + designed
experiment + opportunity-cost analysis** in **~4 weeks** and run the experiment after the
formal engagement.

---

## 5. Success criteria

We agree these up front, in the pre-registration:

- **Primary (process):** We complete the loop — fit → priorities → a *pre-registered,
  powered* experiment → calibrated refit → a recommendation **the partner trusts enough
  to act on (or to consciously decide not to act).** This is the win that's almost
  entirely in our control.
- **Secondary (outcome), conditional:** A reallocation recommendation that, by the
  partner's own judgment, **improves expected KPI per dollar** versus their current plan
  — *or* a credible finding that the current plan is already near-optimal.
- **Product:** ≥ 4 bi-weekly feedback sessions; a concrete list of what they'd pay for.
- **Reference (conditional):** Logo + co-authored case study + reference call **iff** the
  outcome lands (§3).

What we **don't** promise: a specific lift number, "X% more ROI," or that every channel
will have a clean answer. We promise an **honest, calibrated** answer.

---

## 6. Honest risk framing (this is a feature, not fine print)

The whole pitch is *credibility*. So we say the quiet parts out loud:

- **The model may find the spend is already near-optimal.** That is a **valid, valuable
  outcome** — "don't reallocate, you're already close" saves a partner from a bad move
  and is a real deliverable. We will not manufacture a reallocation to look impressive.

- **The model may find we need an experiment before acting.** EVOI exists precisely
  because some answers aren't worth trusting until a test resolves them. "The honest
  move is to run this $X test before reallocating" is a legitimate, defensible
  recommendation — and it's *the loop working as designed.*

- **An experiment can come back underpowered or inconclusive.** The experiment-economics
  layer says so **before** the partner spends money on a test that can't answer the
  question (the powered / underpowered verdict + MDE). Catching that early is value, not
  failure.

- **Forecasting skill is not guaranteed — and we report it honestly.** Our headline
  backtest number — **3.0% out-of-time MAPE, ~¼ of a naive baseline** — is on a
  **SYNTHETIC benchmark world** built to stress-test the methodology, **not a client
  result.** We will *not* present it as "what we got for a customer."

  The one **genuinely external, real dataset** we've published against is **Lydia
  Pinkham (54 years of data)**, and we lead with it *because* it's mixed:
  - **Causal-anchor WIN:** estimated **ROAS 1.48, 90% interval [1.12, 1.85]** — squarely
    inside the bracket from the published academic literature on that dataset.
  - **Forecasting NON-WIN:** **MASE 1.36** — i.e., **no skill over a naive baseline** on
    out-of-time point forecasts — **but honestly calibrated** (the uncertainty bands tell
    the truth about that lack of skill).

  We tell partners this **on purpose.** A platform that will publish its own no-skill
  result is a platform that will tell *you* the truth about *your* data. That is the
  product.

- **SOC 2 is readiness, not a certificate; security posture depends on deployment.**
  For a data-sensitive partner the answer is **self-hosted single-tenant** so nothing
  leaves their network. Full detail and the honest gap list:
  `technical-docs/soc2-readiness.md`.

If a partner can't accept that the answer might be "don't reallocate" or "test first,"
they're not the right design partner — and it's better to find out now.

---

## 7. Case-study artifact — **TEMPLATE (do not send as real)**

> ⚠️ **THIS IS A TEMPLATE WITH `[PLACEHOLDERS]`. There is no real customer yet.**
> Every bracketed field is filled in *with the partner, after the engagement, with the
> partner approving every number.* **Do not** invent a company name, a logo, or a metric
> and present it as a real case study. Until a real engagement closes, this section is a
> mock-up of the *deliverable shape*, nothing more.

---

### Case Study — `[PARTNER NAME]` × MMM Framework

**Partner:** `[PARTNER NAME]` — `[mid-market brand / agency]`, `[category]`
**Data:** `[N]` months of weekly `[spend by M channels]` + `[KPI: revenue / units / …]`,
`[national / geo-level]`.
**Decision on the table:** `[e.g., "reallocate the FY[YY] paid-media budget across
[channels]"]`.

**The loop we ran**
- **T₀ Fit:** `[baseline finding — e.g., "channel [A] showed wide ROI uncertainty: [X.X],
  90% CI [lo, hi]"]`.
- **T₁ Priorities:** `[EIG/EVOI picked channel [A] as the highest-dollar uncertainty to
  resolve — EVOI ≈ $[…]]`.
- **T₂ Pre-registered experiment:** `[matched-market geo lift / national flighting on
  channel [A]; pre-registered MDE [X]%, target power [80]%, duration [W] weeks]`.
- **Run + readout:** `[measured incremental ROAS [X.X], verdict: powered]`.
- **T₃ Calibrated refit:** `[channel [A]'s posterior ROAS tightened from [wide] to
  [narrow] after calibration]`.
- **T₄ Reallocation:** `[recommended moving $[…] from [B] to [A]; expected KPI lift
  [X]% with 90% band [lo, hi]; OR: "recommended NO reallocation — current plan within
  [X]% of optimal"]`.
- **Refutation suite:** `[4/4 passed / which test flagged what]`.

**Outcome (in the partner's words):** `[approved partner quote]`

**Honest caveats:** `[what we could NOT conclude; what the next test should be]`

> **Reference:** `[contact title, available for reference calls — partner-confirmed]`

---

## 8. The founder-sendable email (paste-and-edit)

> **Subject:** Free hands-on measurement engagement — `[their channel allocation]` decision
>
> Hi `[Name]`,
>
> I'm building a causal, agent-operated marketing-mix-modeling platform, and I'm taking on
> **1–3 design partners**. I think `[Company]` is a strong fit: you've got `[~12+ months]`
> of weekly spend + `[KPI]` across `[a few channels]`, and you've got a real
> `[budget-allocation]` decision coming up.
>
> **The offer:** an 8–12 week, founder-led engagement, **free** (or steeply discounted).
> We fit a Bayesian MMM on your data (self-hosted, so nothing leaves your network if that
> matters), figure out **in dollars** which channel's uncertainty is worth resolving,
> **design and read out one experiment**, fold that result back into the model, and hand
> you a **defensible reallocation recommendation** — with honest uncertainty bands and a
> causal stress-test behind it.
>
> **What I ask back:** your data under NDA, a ~bi-weekly feedback call, and — *only if the
> results genuinely land* — your logo, a co-authored case study, and a reference call.
>
> **What I won't do:** promise you a lift number, or manufacture a reallocation to look
> good. If the model says your spend is already near-optimal, or that you should run a
> bigger test before acting, that's what I'll tell you — that's the whole point of the
> approach.
>
> 20 minutes this week to see if it fits?
>
> `[Founder]`

---

> **Maintenance note.** Measured facts that anchor the credibility of this offer —
> **~17s national fit, 6.0× NumPyro speedup, 3.0% MAPE (synthetic), Pinkham ROAS 1.48
> [1.12,1.85] / MASE 1.36** — are the *honest* numbers. Do not round them up, do not move
> the synthetic backtest onto a "customer," and do not soften the Pinkham forecasting
> non-win. If any of these change, update here and in `technical-docs/packaging.md`.
