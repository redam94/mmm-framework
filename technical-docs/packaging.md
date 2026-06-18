# Packaging — Open-Core Strategy (one-pager)

> **Status:** decision-useful draft. **Dollar prices below are PROPOSED /
> ILLUSTRATIVE** — the founder sets finals. The enforceable contract is the
> *limits + feature gates* in `src/mmm_framework/auth/plans.py` (source of truth),
> not the prices. SOC2 is **readiness, not certified** (`technical-docs/soc2-readiness.md`).
> Cloud-cost figures are labeled assumptions, not benchmarks.

The product is **open-core**: a free, self-hosted OSS core, plus paid *managed*
tiers whose only differentiators are the multi-tenancy / hosting capabilities a
single-tenant self-host doesn't need. Those capabilities are the **P0 work that
already shipped** — auth/orgs, hosted sandbox, branding, SSO/audit hooks.

---

## 1. Free/OSS core vs. paid managed tiers

**FREE — the self-hosted OSS core (Apache-2.0, no plan, no limits).**
You run it; you own it. With `MMM_AGENT_HOSTED` unset and auth off, the API runs
as the dev principal and **plan limits do not apply at all** (`plans.py` lines
50–62; the gate dependencies no-op for `is_dev`). This is the "open" in
open-core — a *deployment posture*, distinct from the billed "Free" plan below.
The core includes:

- The **full modeling library** — `BayesianMMM`, all transforms, calibration /
  experiment likelihoods, the extensions (nested / multivariate / combined),
  reporting, the LangGraph agent, the planning/EIG/EVOI + experiment-economics
  stack. No feature is held back from the model.
- The **single-tenant UI + API** — one workspace, one team, no org boundary.
- The **in-process / subprocess agent kernel** (`MMM_AGENT_KERNEL=inprocess|subprocess`).

**PAID — managed tiers (hosted by us).** A self-host doesn't need any of this;
a multi-tenant SaaS does. These map 1:1 to the `FEATURES` flags in `plans.py`:

| Capability | Flag | Why it's a managed-tier thing |
|---|---|---|
| Org isolation + RBAC across many users | `multi_tenant` | Pointless for a single-tenant self-host |
| Sandboxed agent code execution (container kernel) | `hosted_sandbox` | We carry the blast-radius / isolation cost (`MMM_AGENT_KERNEL=container`) |
| Client branding + branded reports/slides | `branding` | Agency/multi-client convenience |
| External IdP / SSO (OIDC/SAML) | `sso` | Enterprise identity |
| Tamper-evident audit-log export | `audit_export` | Compliance reporting |
| Priority support + SLA | `priority_support` | A service, not code |

---

## 2. Tier matrix (mirrors `plans.py` `PLANS`)

Limits are enforced **only in hosted multi-tenant mode**. Prices are illustrative.

| Tier | Seats | Projects | Fits/mo | Features | Proposed price |
|---|---|---|---|---|---|
| **Free** | 2 | 3 | 20 | — (none) | $0 |
| **Team** | 5 | 10 | 200 | `multi_tenant`, `hosted_sandbox`, `branding` | ~$X / seat·mo |
| **Business** | 25 | 50 | 1000 | + `sso`, `audit_export`, `priority_support` | ~$Y / seat·mo |
| **Enterprise** | ∞ | ∞ | ∞ | **all** | Custom (SLA / on-prem / VPC) |

`None` in `plans.py` = unlimited (Enterprise). `frozenset()` = no premium
features (Free). The matrix is additive: each tier is a strict superset of the
one below.

---

## 3. Monetization model

- **Per-seat subscription** (Team / Business) — the price-anchored line item; a
  seat = one accepted `org_members` row.
- **Metered fits** — each tier ships a monthly fit quota (`monthly_fit_quota`);
  a "fit" = one `run_metrics` row this calendar month. Overage is the natural
  **metered/usage** lever (charge per block of fits beyond quota, or hard-stop —
  see §4). Fits are the honest unit because **hosted compute is the COGS
  driver**: the platform is OSS + self-hosted by default, so we only pay for the
  fits people run on *our* infra.
- **Enterprise = custom** — unlimited limits, all features, negotiated SLA and
  on-prem/VPC. Priced as a contract, not a list.

**COGS sizing (measured facts — do not invent benchmarks).** A national NUTS fit
completes in **~17 seconds**; **NumPyro is 6.0× faster than stock PyMC** at equal
draws. Extension models (nested / multivariate / combined) and larger geo panels
cost proportionally more. *Assumption (not a quote):* at a representative
**~$0.10–0.50 / vCPU-hr** (varies by instance type and region), a national fit is
fractions of a cent of raw compute — so quotas are a **margin/abuse guardrail and
upsell trigger**, not cost recovery. Re-derive any unit economics from these two
measured numbers plus the founder's actual cloud bill.

---

## 4. Upgrade triggers (where a wall becomes a checkout)

Every limit is a contextual upgrade prompt. All "limit hit" paths surface as
**HTTP 402** to the client.

| Trigger | Enforced at | Mechanism |
|---|---|---|
| **Seat cap** | `POST /auth/accept-invite` | `service.accept_invite` → `plans.assert_within_seat_limit` raises `PlanLimitError`. You can *invite* freely; the wall is at **accept** (so you don't burn a seat on a bounce). |
| **Project cap** | `POST` create-project (`api/main.py` ~L1302) | `plans.assert_within_project_limit` before insert. |
| **Fit-quota overage** | fit submission | Compare `run_metrics` count-this-month vs `monthly_fit_quota`; over → 402 / upsell (today the count is exposed via `/auth/usage`; the hard gate at submit is the §6 TODO). |
| **Feature gate** (SSO, audit export, …) | any gated route | `require_plan_feature("sso")` dep (`auth/deps.py`) → **402** unless the org's plan `has()` the flag. Dev principal no-ops. |

Free→Team unlocks multi-tenant + sandbox + branding; Team→Business unlocks
SSO + audit export + priority support; Business→Enterprise removes all limits.

---

## 5. Metering mechanics

`GET /auth/usage` (`auth/routes.py`, → `plans.org_usage`) returns the org's plan,
its feature set, and a `{used, limit, remaining, over}` slot for each metered
axis. Counts come straight from the system tables:

- **seats** ← `store.count_org_members` (rows in `org_members`)
- **projects** ← `store.count_org_projects` (rows in `projects` for the org)
- **fits** ← `store.count_org_fits_since(org, month_start)` (rows in `run_metrics`
  since `_month_start_ts()`)

`limit: null` ⇒ unlimited (Enterprise); `over: true` ⇒ already past the line. The
frontend reads this to draw usage bars and trigger upgrade nudges before a 402.
The `run_metrics` table is the same per-fit history the measurement loop writes
at fit time — metering is free-riding on existing telemetry, no extra plumbing.

---

## 6. Intentionally NOT gated yet + the billing TODO

**Not gated yet (by design):**
- **Fit-quota hard stop.** Usage is *metered and exposed* (`/auth/usage`), but
  fit submission isn't blocked on quota yet — it's a count + nudge today. Flip to
  a `PlanLimitError` at submit when overage policy (hard-stop vs. metered
  overage billing) is decided.
- **Within-tier metering of compute weight.** A geo-panel / extension fit costs
  more than a national fit but counts as one `run_metrics` row. A weighted unit
  ("fit credits") is deferred until the bill justifies the complexity.
- **Per-project / per-seat egress, storage, KB size.** No limits; revisit if a
  whale appears.

**Track-3 billing integration (TODO).** There is **no payment processor wired
in**. Today, **plan changes are ops-driven**: an operator calls
`store.set_org_plan(org_id, plan)` directly. To self-serve:

1. Stripe Checkout / Billing for per-seat subscription + metered-fit usage records.
2. A **webhook** (`checkout.session.completed`, `customer.subscription.updated/deleted`)
   that maps the Stripe price → plan key and calls **`store.set_org_plan`** — the
   single, already-built mutation point. No other code path needs to know about money.
3. Push `/auth/usage` fit counts as Stripe **usage records** for metered overage.
4. Dunning / past-due → downgrade to `free` (same `set_org_plan` call).

The gating mechanism is complete and enforced; only the **payment → plan-key**
bridge is missing. That isolation (one mutation, one webhook) is deliberate.
