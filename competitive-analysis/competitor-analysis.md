# MMM Platform Competitor Analysis

> **Snapshot:** mid‑2026 · 9 platforms · 15 dimensions · researched and adversarially fact‑checked across ~90 primary sources.
>
> This is the reference writeup. See also the interactive version **[`mmm-competitor-comparison.html`](./mmm-competitor-comparison.html)** and the raw verified data **[`research-data.json`](./research-data.json)** (every profile with its source URLs, confidence grade, and the corrections the verification pass applied).

## Positioning

This app is a research-grade, agentic, causally-rigorous open MMM framework that sits in a distinct niche: it matches or exceeds the open-source Bayesian tools (Meridian, PyMC-Marketing, the deprecated LightweightMMM) on methodological depth while adding a workflow layer none of them ship — an LLM 'Oracle' agent, an adaptive measurement loop with EIG/EVOI experiment prioritization, a model-free continuous-learning bandit, a declarative estimand system, and a Model Garden for custom and non-MMM Bayesian families. Against mature commercial SaaS (Recast, Mutinex) and enterprise consulting (Analytic Partners, the Kantar/Ipsos/Circana camp), it trades brand trust, managed support, turnkey data connectors, and proven at-scale adoption for openness, transparency, and an unusually broad experiment-and-uncertainty toolkit. Note the marketing caveat: despite 'extends PyMC-Marketing' framing, the shipped engine is a standalone PyMC 6 reimplementation and PyMC-Marketing is only an optional report-reading extractor. In short: the most feature-broad and methodologically ambitious open framework here, but the least commercially mature.

## How this was built, and how to read it

The subject platform ("this app" — the MMM Framework) was characterized directly from its source tree, marking capabilities **shipped vs. deferred** rather than taking marketing copy at face value. Each competitor was researched from primary sources (vendor docs, GitHub/PyPI, technical papers, funding announcements) and then **adversarially fact‑checked** by a second pass, with every correction logged and a confidence grade assigned (see `research-data.json`).

Ratings compress a nuanced landscape into a four‑step scale for scanning — **Strong / Moderate / Limited / None** — so read the per‑dimension notes and profiles for the real texture. Caveats that keep this honest:

- **Competitor‑sourced claims are flagged and treated as directional**, not settled fact — most notably the head‑to‑head speed/accuracy benchmarks PyMC Labs and Recast publish about rivals.
- **Pricing for closed vendors** (Recast, Mutinex, Analytic Partners, the enterprise camp) is third‑party estimate, not a quote — none publish price lists.
- Some ratings reflect **documented methodological opacity** (closed vendors publish little), which is not the same as a proven weakness.
- This is a **point‑in‑time snapshot**; open‑source versions, funding, and analyst positions drift. Independent analysis, not affiliated with or endorsed by any vendor named.

## The matrix at a glance

| Dimension | This app | Robyn | Meridian | PyMC‑Mktg | Recast | Mutinex | LMMM | Analytic&nbsp;P. | Enterprise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Statistical methodology** | **Strong** | Moderate | Strong | Strong | Strong | Moderate | Moderate | Moderate | Moderate |
| **Uncertainty quantification** | **Strong** | Limited | Strong | Strong | Strong | Moderate | Strong | Limited | Moderate |
| **Causal identification rigor** | **Strong** | Limited | Strong | Strong | Moderate | Moderate | Limited | Moderate | Moderate |
| **Experiment / lift-test calibration** | **Strong** | Moderate | Strong | Strong | Strong | Limited | Limited | Moderate | Moderate |
| **Continuous / sequential learning** | **Strong** | None | None | Limited | Moderate | Moderate | None | Moderate | Moderate |
| **Geo & hierarchical modeling** | **Strong** | Limited | Strong | Strong | Moderate | Moderate | Strong | Moderate | Moderate |
| **Custom-model extensibility** | **Strong** | Limited | Limited | Strong | None | None | Moderate | None | None |
| **LLM / agent interface** | **Strong** | None | None | Moderate | None | Moderate | None | Limited | Limited |
| **Automated reporting & deliverables** | **Strong** | Moderate | Moderate | Moderate | Strong | Strong | Limited | Strong | Strong |
| **Experiment design & value-of-information (EIG/EVOI)** | **Strong** | None | None | Limited | Moderate | Limited | None | Limited | Moderate |
| **Licensing & cost** | **Strong** | Strong | Strong | Strong | Limited | Limited | Strong | Limited | Limited |
| **Ease of use / onboarding** | **Moderate** | Limited | Moderate | Limited | Strong | Strong | Limited | Strong | Strong |
| **Vendor support & maturity** | **Limited** | Strong | Strong | Strong | Moderate | Moderate | None | Strong | Strong |
| **Deployment / multi-tenant hosting** | **Moderate** | Limited | Limited | Limited | Strong | Strong | Limited | Strong | Strong |
| **Data-connector ecosystem** | **Limited** | Limited | Moderate | Limited | Moderate | Strong | Limited | Strong | Strong |

_Legend: **Strong** = a first‑class, well‑developed capability · **Moderate** = present but partial/indirect · **Limited** = minimal, bolt‑on, or manual · **None** = not offered._

## Dimension detail

<details>
<summary><b>Statistical methodology</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **fully Bayesian PyMC 6** — Standalone PyMC 6/PyTensor 3 engine; geometric/delayed/Weibull adstock, Hill/logistic/tanh/MM saturation, Fourier seasonality, linear/piecewise/spline/GP trend, ROI-parameterized default priors. |
| Robyn | Moderate | **ridge, not MCMC-Bayesian** — Frequentist L2 ridge (glmnet) + Nevergrad multi-objective search; point estimates (MAP-equivalent under a normal prior), no posterior sampling. |
| Meridian | Strong | **hierarchical Bayesian NUTS** — Geo-level hierarchical Bayesian on TensorFlow Probability; adstock + Hill, reach x Hill(frequency); media effects constant over time. |
| PyMC-Marketing | Strong | **fully Bayesian PyMC** — NUTS across NumPyro/BlackJAX/Nutpie; standard carryover+shape formulation with rich transform menu and HSGP time-varying effects. |
| Recast | Strong | **Bayesian Stan HMC, time-varying** — 8k+ lines of Stan; locally-periodic GPs give time-varying ROI/intercept/saturation; negative-binomial integer-lag adstock, single-param Hill. |
| Mutinex | Moderate | **Bayesian but opaque** — Marketed as Bayesian with time-varying baseline and Campaign-Varying decomposition, but sampler and adstock/saturation forms are undisclosed. |
| LightweightMMM | Moderate | **Bayesian but limited/deprecated** — NumPyro/JAX NUTS, Normal likelihood, three transform families; polynomial trend + fixed harmonic seasonality only. Archived and unmaintained. |
| Analytic Partners | Moderate | **econometric/ML blend, undisclosed** — 'Commercial mix modeling' spanning price/promo/distribution/macro; third parties describe a Bayesian/econometric/ML blend but AP publishes no engine details. |
| Enterprise MMM | Moderate | **mixed, modernizing to Bayesian** — Historically OLS/GLS/ridge econometrics; modern platforms (Keen, Ipsos MMA, Kantar HamiltonAI) are Bayesian per marketing materials, internals proprietary. |

</details>

<details>
<summary><b>Uncertainty quantification</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **full posteriors, approx flagged** — Genuine HDI/credible intervals on ROI/contributions; MAP/ADVI/Pathfinder approximate fits surfaced end-to-end as 'uncertainty not calibrated' (nulled R-hat/ESS, banners). |
| Robyn | Limited | **model-selection spread** — No posterior; uncertainty approximated via k-means over Pareto models + bootstrapped ROAS/CPA CIs, reflecting selection spread not principled posteriors. |
| Meridian | Strong | **genuine posteriors** — Full NUTS posteriors and credible intervals on ROI/effects. |
| PyMC-Marketing | Strong | **well-calibrated posteriors** — Full Bayesian credible intervals on ROAS/contributions/budgets. |
| Recast | Strong | **HMC posteriors on all outputs** — Credible intervals on every ROI/saturation/time-shift estimate; prospective out-of-sample scoring (1 - scaled CRPS). |
| Mutinex | Moderate | **marketed, unverifiable** — Claims distributions of plausible ROI rather than point estimates, but the mechanism is not externally documented. |
| LightweightMMM | Strong | **genuine posteriors (deprecated)** — Full posteriors/credible intervals, but the project is end-of-life. |
| Analytic Partners | Limited | **not documented** — No public statement on whether deliverables report full posteriors/credible intervals vs point estimates + significance. |
| Enterprise MMM | Moderate | **inconsistent across vendors** — Bayesian platforms yield intervals; classic econometric deliverables typically report point estimates + significance. |

</details>

<details>
<summary><b>Causal identification rigor</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **DAG + refutation + sensitivity** — Role-tagged confounder controls, DAG-based specification, fit-based refutation suite, structural-identification designs, and sensitivity-to-unobserved-confounding. |
| Robyn | Limited | **business-logic penalty only** — DECOMP.RSSD nudges spend-vs-effect alignment (itself critiqued as ad hoc); no causal DAG or confounding framework. |
| Meridian | Strong | **causal-inference designed** — Built for causal attribution with ROI priors and experiment calibration; cannot model synergies and is explicitly not for forecasting. |
| PyMC-Marketing | Strong | **DAG-based identification** — DAG-based causal identification (added 2025) to mitigate confounding, plus principled experiment likelihoods. |
| Recast | Moderate | **observational, assumptions exposed** — Assumption-heavy observational model with publicly documented causal assumptions; leans on lift tests for ground truth. |
| Mutinex | Moderate | **baseline separation, opaque** — Time-varying baseline to avoid crediting organic demand to media; foundation-model priors, but identification method undocumented. |
| LightweightMMM | Limited | **no causal framework** — Standard Bayesian regression; no DAG, refutation, or experiment calibration. |
| Analytic Partners | Moderate | **econometric base + validation** — Argues for a comprehensive econometric base validated by experiments; ROI Genome benchmarks as priors; specifics unpublished. |
| Enterprise MMM | Moderate | **triangulation-based** — MMM + MTA + experiments + benchmark priors (Ipsos MMA); NCSolutions adds purchase-based ground truth; mechanics proprietary. |

</details>

<details>
<summary><b>Experiment / lift-test calibration</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **in-graph likelihood + off-panel** — In-graph likelihood on contribution/ROAS/mROAS estimands, lift-test->beta-prior derivation, and off-panel calibration for experiments outside the training window. |
| Robyn | Moderate | **optimization objective** — MAPE.LIFT is a third Nevergrad objective when lift data is supplied; a penalty on model selection, not a probabilistic prior; also a reach&frequency curve calibrator. |
| Meridian | Strong | **ROI-prior helpers** — Converts experiment point-estimate+SE or 95% CI into a LogNormal ROI prior; guided-manual given estimand mismatch warnings. |
| PyMC-Marketing | Strong | **likelihood-integrated** — add_lift_test_measurements() adds an in-inference likelihood term; geo-lift workflow updates hierarchical parameters for untested geos. |
| Recast | Strong | **time-localized ROI prior** — Lift results tighten the ROI prior only over the dates the test ran, propagating staleness and handling conflicting tests. |
| Mutinex | Limited | **deliberately de-emphasized** — CEO publicly argues against forcing lift tests as hard priors ('lazy models'); experiments used to validate, not as a first-class calibration loop. |
| LightweightMMM | Limited | **manual only** — No built-in lift-test mechanism; folding in experiments is manual (a primary reason for Meridian's succession). |
| Analytic Partners | Moderate | **validate-not-force** — Explicitly critiques experiment-as-prior; uses experiments to validate an econometric base and ROI Genome benchmarks as priors. |
| Enterprise MMM | Moderate | **triangulation + purchase data** — Benchmark priors, MMM+MTA+experiment triangulation, NCSolutions purchase-based lift; exact experiment-to-prior mechanics proprietary. |

</details>

<details>
<summary><b>Continuous / sequential learning</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **model-free geo bandit** — Standalone NumPyro/JAX response-surface bandit with central-composite designs, Thompson allocation, ENBS stopping, Laplace-KG/D-optimal design selection, drift discounting. |
| Robyn | None | **** — Static batch fit; robyn_refresh() exists but docs flag it as not thoroughly tested; no sequential learning loop. |
| Meridian | None | **** — Batch model fit; no adaptive/sequential learning framework. |
| PyMC-Marketing | Limited | **refresh via serialization** — Serialize/refresh and time-slice CV, but no bandit or sequential value-of-information loop. |
| Recast | Moderate | **always-on weekly refresh** — Automated weekly model refresh with time-varying coefficients; continuous but not a designed-experiment bandit. |
| Mutinex | Moderate | **continuous MMM** — GrowthOS refreshes continuously rather than quarterly; not a formal sequential-experiment loop. |
| LightweightMMM | None | **** — One-shot batch fit. |
| Analytic Partners | Moderate | **dynamic decisioning** — Dynamic Decisioning / Live Modeling for near-real-time refresh, managed. |
| Enterprise MMM | Moderate | **always-on overlays** — SaaS overlays (Kantar LIFT ROI daily; Keen adaptive prior updating) narrow the cadence gap. |

</details>

<details>
<summary><b>Geo & hierarchical modeling</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **panel + per-geo betas** — Geo and geo x product MFF panels, hierarchical partial pooling, opt-in per-geo channel effectiveness (V3 vary_media_by_geo), geo/DMA allocation. |
| Robyn | Limited | **national aggregate** — No native hierarchical geo random-effects; typically one coefficient set on national time series. |
| Meridian | Strong | **geo hierarchical core** — Geo-level hierarchical partial pooling of media effects with national fallback. |
| PyMC-Marketing | Strong | **arbitrary hierarchical dims** — multidimensional.MMM supports arbitrary dims (geo x product) with partial pooling. |
| Recast | Moderate | **national top-down** — Primarily national time-series; designed geo experimentation is a separate product (GeoLift); geo-panel parity undocumented. |
| Mutinex | Moderate | **hierarchical claimed** — Geo-level and portfolio/category/brand hierarchical modeling marketed; internals undisclosed. |
| LightweightMMM | Strong | **national+geo (deprecated)** — Native national and sub-national hierarchical models (a reviewer-cited 'killer feature'), but archived. |
| Analytic Partners | Moderate | **multi-region enterprise** — Handles multi-country/multi-brand data at scale; hierarchical structure not publicly documented. |
| Enterprise MMM | Moderate | **multi-market managed** — Handles complex multi-market portfolios; geo structure vendor-specific and proprietary. |

</details>

<details>
<summary><b>Custom-model extensibility</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **Model Garden + non-MMM** — Author/version/publish CustomMMM subclasses with CONFIG_SCHEMA + pluggable likelihood + estimands, 9-tier compat contract; genuine non-MMM families (CFA/LCA/latent-factor) ride the same pipeline. |
| Robyn | Limited | **fixed pipeline** — Transform choices and hyperparameter bounds are configurable, but the ridge+Nevergrad pipeline is fixed. |
| Meridian | Limited | **fixed TFP backend** — Opinionated fixed model spec and sampler; no swappable model classes. |
| PyMC-Marketing | Strong | **custom priors/transforms/terms** — Custom priors, custom adstock/saturation, custom model terms, arbitrary dims; plus CLV/choice/Bass beyond MMM. |
| Recast | None | **closed managed** — Proprietary Stan model; cannot self-host or extend. |
| Mutinex | None | **closed** — Proprietary engine; the only open artifact (mmm-eval) evaluates other frameworks, not GrowthOS. |
| LightweightMMM | Moderate | **code-level custom** — Custom priors and code edits possible; three fixed transform families; no friendly extension API. |
| Analytic Partners | None | **proprietary** — Managed proprietary method; no user-extensible modeling. |
| Enterprise MMM | None | **proprietary managed** — Vendor-built models; no self-serve extensibility. |

</details>

<details>
<summary><b>LLM / agent interface</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **Oracle is the flagship UX** — LangGraph agent with ~100 tools across data/EDA/config/fit/validation/estimands/reporting/experiments/learning, running fits in an isolated kernel; multi-provider LLM. |
| Robyn | None | **** — No agent interface. |
| Meridian | None | **** — No LLM agent; a no-code Scenario Planner exists but is not conversational. |
| PyMC-Marketing | Moderate | **MMM Agent in gated beta** — Proprietary AI 'MMM Agent' (invite-only beta) automates data prep/model selection/scenarios; not part of the open library. |
| Recast | None | **** — Managed service with human data scientists, not an agent. |
| Mutinex | Moderate | **MAITE assistant** — LLM-based 'AI growth consultant' layered on each customer's MMM for NL querying/recommendations. |
| LightweightMMM | None | **** — Code-first library, no agent. |
| Analytic Partners | Limited | **analyst-led** — Guidance is human-analyst delivered; no conversational agent surfaced. |
| Enterprise MMM | Limited | **mostly analyst-led** — Some self-serve scenario UIs; no flagship conversational agent. |

</details>

<details>
<summary><b>Automated reporting & deliverables</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **multi-format + prefit + branding** — Classic + evidence-coded 'Augur' HTML readouts, pre-fit Model Design Readout, model-defense, estimand-CI + PPC sections, PPTX decks, client-branding auto-recolor; all downloadable artifacts. |
| Robyn | Moderate | **rich diagnostics** — One-pager per model, waterfall decomposition, response/adstock curves; analyst-oriented, not client decks. |
| Meridian | Moderate | **Scenario Planner** — No-code Looker Studio Scenario Planner over a fitted model; core outputs are charts/optimization, not polished client reports. |
| PyMC-Marketing | Moderate | **plotting, DIY UIs** — Interactive Plotly plotting and model outputs; stakeholder UIs and reports are DIY. |
| Recast | Strong | **managed weekly dashboards** — Live out-of-sample accuracy dashboards and planning/optimization UI, refreshed weekly. |
| Mutinex | Strong | **GrowthOS platform** — Integrated scenario planning + optimization + MAITE narrative; polished decision workflow. |
| LightweightMMM | Limited | **basic plots** — Standard MMM plots; weaker out-of-the-box reporting than Robyn/Recast. |
| Analytic Partners | Strong | **executive-ready managed** — PROPHET planning/optimization plus analyst-delivered strategic storytelling for executives. |
| Enterprise MMM | Strong | **managed deliverables** — Analyst-built insights presentations plus SaaS scenario/optimization overlays. |

</details>

<details>
<summary><b>Experiment design & value-of-information (EIG/EVOI)</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **EIG/EVOI + Pareto optimizer** — EIG/EVOI experiment prioritization, pre-registered lifecycle registry, geo-lift/DiD/flighting designers with power+placebo math, model-anchored economics, 4-objective Pareto optimizer. |
| Robyn | None | **** — No experiment design or VOI tooling. |
| Meridian | None | **** — Consumes experiment results as priors but has no experiment-design/VOI module. |
| PyMC-Marketing | Limited | **consumes lift tests** — Integrates lift tests but offers no experiment-design or value-of-information prioritization. |
| Recast | Moderate | **GeoLift product** — Augmented-synthetic-control geo incrementality testing (separate product); no VOI/EIG prioritization surfaced. |
| Mutinex | Limited | **** — Experiments used to validate; no formal VOI/experiment-design engine documented. |
| LightweightMMM | None | **** — No experiment design. |
| Analytic Partners | Limited | **validation testing** — Uses in-market tests to validate; no published VOI-driven design engine. |
| Enterprise MMM | Moderate | **in-market test programs** — Run structured in-market experiments and triangulation, but not information-theoretic VOI prioritization. |

</details>

<details>
<summary><b>Licensing & cost</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Strong | **Apache-2.0, free** — Open source; cost is compute + operator time. No seat/subscription fees. |
| Robyn | Strong | **MIT, free** — Permissive open source; free for commercial use. |
| Meridian | Strong | **Apache-2.0, free** — Free open source; cost is GPU compute + analyst time. |
| PyMC-Marketing | Strong | **Apache-2.0, free** — Free library; consulting/MMM Agent priced separately (undisclosed). |
| Recast | Limited | **premium enterprise SaaS** — Opaque custom pricing; ~$10M+ recommended media spend; not viable for small brands. |
| Mutinex | Limited | **enterprise custom** — No public tiers; enterprise ACV; On-Demand mid-market pricing also undisclosed. |
| LightweightMMM | Strong | **Apache-2.0, free (EOL)** — Free open source, but deprecated/archived. |
| Analytic Partners | Limited | **premium quote-based** — No free tier; six-figure+ managed engagements (third-party estimates). |
| Enterprise MMM | Limited | **high six-figure** — Non-public negotiated pricing (~$200K-$500K/project ballpark); Keen is a transparent-subscription exception. |

</details>

<details>
<summary><b>Ease of use / onboarding</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Moderate | **agent lowers barrier, BYO-infra** — The Oracle chat and Data Studio ease modeling, but it is a self-hosted, early-version framework requiring technical setup; no turnkey onboarding. |
| Robyn | Limited | **R/stats expertise** — Requires R/Python fluency and judgment to select among ~100 Pareto-front models. |
| Meridian | Moderate | **Python + no-code planner** — Model build needs applied-Bayesian skill; Scenario Planner gives non-technical users a no-code consumption layer. |
| PyMC-Marketing | Limited | **steep library** — Requires Python + real Bayesian/PyMC expertise; a framework, not a product. |
| Recast | Strong | **fully managed** — Vendor operates the model; low in-house lift (after ~2-3 week onboarding). |
| Mutinex | Strong | **managed + MAITE** — Abstracted behind the platform; On-Demand builds a model in <24h with agentic onboarding. |
| LightweightMMM | Limited | **code-first** — No GUI; requires Python/Bayesian skill and manual data scaling. |
| Analytic Partners | Strong | **managed service** — Consulting-led; no in-house data science required (but longer cycles). |
| Enterprise MMM | Strong | **managed service** — Analyst-delivered; 'no data scientist required' positioning (esp. Keen). |

</details>

<details>
<summary><b>Vendor support & maturity</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Limited | **single author, v0.2.0** — Ambitious single-author project at an early version; real bus-factor, support-SLA, and long-term-maintenance risk; newest features lightly battle-tested. |
| Robyn | Strong | **Meta-backed, mature** — Widely adopted since ~2020-2021, active CRAN releases, methodology paper, large community. |
| Meridian | Strong | **Google-backed** — Actively maintained (roughly monthly releases), certified-partner ecosystem, official LightweightMMM successor. |
| PyMC-Marketing | Strong | **PyMC Labs + community** — Maintained by PyMC creators, frequent releases, active community; still pre-1.0 with API churn. |
| Recast | Moderate | **seed-stage, established product** — Small (~11-50), ~$4.5M raised, no round since 2022; but a mature managed product with credible adopters and support. |
| Mutinex | Moderate | **funded scale-up** — ~A$132.5M valuation, named adopters, Dentsu/WARC partnerships; thin independent review volume. |
| LightweightMMM | None | **deprecated/archived** — Unsupported; repo archived read-only Jan 2026. |
| Analytic Partners | Strong | **Gartner Leader, 25 yrs** — Founded 2000, Gartner MQ Leader 2024+2025, Forrester Wave Leader; deep enterprise pedigree. |
| Enterprise MMM | Strong | **established, consolidating** — Ipsos MMA/Kantar/Circana/Keen; long histories, major parents, Gartner-recognized. |

</details>

<details>
<summary><b>Deployment / multi-tenant hosting</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Moderate | **self-host, auth layer, BYO-infra** — FastAPI + React with org/tenant JWT auth, roles, rate limiting, sandbox-kernel isolation and GCP/k8s assets; but no managed SaaS and the operator must build the sandbox image. |
| Robyn | Limited | **self-host library** — Runs in the user's environment; no hosting/multi-tenancy. |
| Meridian | Limited | **self-host, no managed fitting** — Self-run Python; Scenario Planner is a layer over a self-fitted model, not a hosted fitting service. |
| PyMC-Marketing | Limited | **self-host DIY** — Runs anywhere Python runs; hosting/scheduling/UIs are the user's responsibility. |
| Recast | Strong | **managed SaaS, SOC 2** — Cloud-native AWS SaaS, SOC 2 compliant, dedicated support. |
| Mutinex | Strong | **multi-tenant SaaS** — Hosted multi-tenant cloud platform (GrowthOS + DataOS). |
| LightweightMMM | Limited | **self-host (EOL)** — Self-hosted library, no hosting; archived. |
| Analytic Partners | Strong | **hosted managed platform** — Web-based GPS-Enterprise on cloud marketplaces, delivered as managed/hybrid engagements. |
| Enterprise MMM | Strong | **hosted managed** — Vendor-hosted SaaS overlays + managed delivery; not self-host. |

</details>

<details>
<summary><b>Data-connector ecosystem</b></summary>

| Platform | Rating | Detail |
| --- | --- | --- |
| **This app** | Limited | **upload + GCS/BigQuery, ad stubs** — CSV/Excel/MFF upload with validation, Data Studio EDA, synthetic DGP, GCS/BigQuery/S3; Meta/Google/TikTok connectors are explicit NotImplementedError stubs. |
| Robyn | Limited | **mostly file-based** — Analyst-supplied data; a Meta MMM API connector demo exists, but no broad connector suite. |
| Meridian | Moderate | **MMM Data Platform** — Can ingest Google Query Volume, YouTube reach/frequency and core media signals; otherwise advertiser-supplied aggregate data. |
| PyMC-Marketing | Limited | **DIY ingestion** — No native connectors; data prep is on the user. |
| Recast | Moderate | **managed integration** — Vendor handles data integration during onboarding; not a self-serve connector catalog. |
| Mutinex | Strong | **DataOS** — Dedicated data-provisioning layer ingesting marketing/sales/pricing/audience data; WARC + Tracksuit integrations. |
| LightweightMMM | Limited | **manual data prep** — Requires manual data scaling; no connectors. |
| Analytic Partners | Strong | **ADAPTA + partners** — ADAPTA ETL/ELT orchestration with direct Meta/Google/Amazon/Roku integrations. |
| Enterprise MMM | Strong | **enterprise integration** — Managed multi-source data integration incl. clean-room/purchase-data (NCSolutions). |

</details>

## Where this app wins

- LLM 'Oracle' agent as the primary interface (~100 tools running fits in an isolated kernel) — no open-source competitor (Robyn, Meridian, LightweightMMM) ships a conversational agent, and even PyMC-Marketing's MMM Agent is an invite-only proprietary beta outside the open library.
- Adaptive measurement loop with EIG/EVOI experiment prioritization and a pre-registered lifecycle registry — every other tool here consumes experiment results at best (Meridian/PyMC-Marketing as priors) but none prioritize WHICH experiment to run next via value-of-information; Recast's GeoLift is a testing product, not VOI-driven prioritization.
- Model-free continuous sequential-learning geo bandit (Thompson, funding-line, ENBS stopping, Laplace-KG/D-optimal design) that learns spend->outcome with NO pre-fit MMM — a category none of the batch-fit libraries (Robyn/Meridian/PyMC-Marketing/LightweightMMM) offer at all.
- Declarative, named, capability-gated estimand subsystem with bit-stable post-hoc and in-graph evaluation — a level of causal-quantity formalization competitors expose only as fixed ROI/contribution outputs.
- Model Garden + genuine non-MMM Bayesian families (CFA/LCA/joint latent-factor MMM) through the same fit->estimand->serialize->report pipeline — far broader extensibility than Robyn's fixed pipeline or the closed commercial engines, and beyond even PyMC-Marketing's custom-term flexibility.
- Off-panel experiment calibration (calibrate from an experiment run outside the training window by evaluating the global response curve) — a capability not documented in Meridian or PyMC-Marketing's lift-test workflows, which condition on in-window spend points.
- Causal-DAG model builder plus a fit-based refutation suite and sensitivity-to-unobserved-confounding — more explicit causal-identification tooling than Robyn's DECOMP.RSSD heuristic or the opaque enterprise engines.
- First-class approximate fits (MAP/ADVI/Pathfinder) surfaced end-to-end as 'uncertainty not calibrated' (nulled R-hat/ESS, banners, artifact metadata) — a rigor-forward honesty feature absent from competitors that silently present point estimates.
- Pre-fit Model Design Readout (priors enumerated, prior-predictive checks, prior-implied ROI, SBC verdict) as an auditable pre-registration document before fitting — an anti-specification-shopping deliverable no competitor ships.
- Broad reporting engine (evidence-coded HTML readouts, model-defense report, PPTX decks, client-branding auto-recolor) that rivals the managed-SaaS deliverable polish while remaining open and self-hosted.

## Where this app lags

- Single primary author at v0.2.0 — real bus-factor, support-SLA, and long-term-maintenance risk versus Meta (Robyn), Google (Meridian), PyMC Labs (PyMC-Marketing), or 25-year Gartner-Leader Analytic Partners.
- No vendor-managed hosted SaaS: multi-tenancy requires self-deploying (GCP Terraform/GCE or k8s) and building a sandbox kernel image, versus Recast/Mutinex/enterprise vendors that deliver turnkey, SOC 2-compliant hosted platforms with onboarding.
- Thin data-connector ecosystem — the Meta/Google Ads/TikTok connectors are explicit NotImplementedError stubs; real ingestion is CSV/MFF + GCS/BigQuery, well behind Mutinex's DataOS or Analytic Partners' ADAPTA with direct Meta/Google/Amazon/Roku integrations.
- No proven large-scale or third-party adoption: validation is internal/synthetic (self-generated DGP worlds + a limited rolling-origin backtest), with no published independent benchmarks or competitive bake-offs, versus named enterprise rosters and analyst recognition (Gartner/Forrester) for the commercial vendors.
- Brand trust and commercial/compliance maturity are aspirational — SOC2/DR/runbook docs are readiness plans, not certifications, and go-to-market is a design-partner concept, not a proven product.
- Very large, rapidly built surface area (~131k LOC, ~292 modules by one author) means breadth may outpace hardening; much of the newest functionality (agent, continuous learning, Model Garden, non-MMM) is recent and lightly battle-tested in production.
- Partial feature-completeness in places: MultivariateMMM/CombinedMMM keep a joint MvNormal-LKJ likelihood (no per-outcome non-Gaussian families), non-MMM families ship as worked examples rather than a documented general API, and several items (in-graph experiment-calibration ROI bridge for impression channels, time-varying CPM, ragged-panel spend columns) remain deferred.
- Ease-of-use for non-technical buyers still trails fully managed offerings — the agent lowers the barrier, but it is a self-hosted framework, not a point-and-click product like Recast or Mutinex GrowthOS.
- No native reach & frequency modeling surfaced — a genuine differentiator Meridian (and Recast's spend-only stance aside) offers for video/YouTube-heavy plans.

## Platform profiles

### 🔭 This app — MMM Framework (Bayesian Marketing Mix Modeling platform)

**Type:** Open source (Apache‑2.0) · single primary author · v0.2.0 &nbsp;|&nbsp; **Subject of this comparison**

A research-grade, fully-Bayesian Marketing Mix Modeling platform (v0.2.0, Apache-2.0, Python 3.12+) that pairs a custom PyMC 6 modeling engine with an unusually ambitious surrounding system: an LLM "Oracle" agent as the primary interface, an adaptive measurement loop (fit → EIG/EVOI experiment prioritization → pre-registered experiments → calibrated refit → allocate → re-test), a model-free continuous-learning geo bandit, a declarative estimand subsystem, a "Model Garden" for versioned custom/non-MMM Bayesian models, and evidence-coded HTML/PPTX deliverables. It is methodologically rigor-forward (pre-registration, SBC, prior-predictive readouts, genuine posterior uncertainty) and feature-broad, but it is a single-author project at an early version with a stubbed data-connector ecosystem and no managed SaaS — powerful and distinctive on methodology and workflow, less mature on commercial polish, integrations, and third-party benchmarks. Note: despite README/marketing language, it does NOT depend on or subclass PyMC-Marketing — the core engine is a standalone reimplementation; pymc-marketing exists only as an optional read-compat report extractor.

**Methodology.** Fully Bayesian, implemented directly on PyMC 6 / PyTensor 3 (not a PyMC-Marketing subclass — `pymc_marketing` is referenced only by `reporting/extractors/pymc_marketing.py` for reading foreign models into reports). The core is a standalone `BayesianMMM` class (src/mmm_framework/model/base.py:341) with a multiplicative/additive-in-log spec: geometric adstock (parametric + delayed/weibull variants, transforms/adstock.py), Hill and logistic/tanh/Michaelis-Menten saturation (transforms/saturation.py), Fourier seasonality, and a full trend family — linear, piecewise, B-spline, and Gaussian-Process (HSGP) — built inline in `_build_trend_component`. Hierarchical partial pooling across geography/product panels, plus an opt-in V3 `vary_media_by_geo` for partial-pooled per-geo channel effectiveness. Primary inference is NUTS via NumPyro (JAX), PyMC, or NutPie. Approximate posteriors (MAP, ADVI, full-rank ADVI, Pathfinder VI via pymc-extras) are first-class for fast model-checking and are surfaced end-to-end as "uncertainty not calibrated" (reports null R-hat/ESS, banners, artifact metadata). Default media priors are ROI-parameterized (prior lives on the decision scale, break-even median 1.0) rather than on the arbitrary standardized coefficient. Philosophy is explicitly anti-specification-shopping: pre-specified analyses, SBC, prior-predictive pre-registration, and experiment-calibrated priors/likelihoods to reduce researcher degrees of freedom.

**Distinctive differentiators (shipped unless noted):**

- LLM 'Oracle' agent as the primary interface: a LangGraph multi-agent scaffold with ~100 tools (agents/tools.py) over multiple providers (Anthropic/OpenAI/Google/Vertex/LM Studio), running model fits in an isolated Jupyter kernel (in-process/subprocess/sandboxed-container) — configure/fit/validate/report all via chat. SHIPPED.
- Adaptive measurement loop (T0→T5): fit → EIG/EVOI experiment prioritization (planning/eig.py, evoi.py, priority.py) → pre-registered experiment lifecycle registry (draft→planned→running→completed→calibrated) → calibrated refit → allocation → information-decay-triggered re-test. SHIPPED.
- Model-free continuous sequential learning bandit (continuous_learning/): a NumPyro/JAX geo response-surface that learns spend→outcome directly from designed geo experiments with NO pre-fit MMM — pluggable Hill/logistic/monotone-spline activations, central-composite designs, Thompson sampling, funding-line/marginal-ROAS decisions, ENBS stopping, and Laplace knowledge-gradient/D-optimal EIG design selection. SHIPPED.
- Declarative estimand subsystem (estimands/): named, serializable, capability-gated causal quantities (contribution_roi, marginal_roas, counterfactual_roi, contribution, awareness_lift, cost_per_conversion, plus latent fit_index/factor_loading) evaluated post-hoc (numpy) or in-graph (pytensor) with bit-stable equivalence to legacy paths. SHIPPED.
- Model Garden + Atelier: author/version/publish bespoke Bayesian model classes (CustomMMM subclass, per-model CONFIG_SCHEMA + pluggable likelihood + DEFAULT_ESTIMANDS) governed by a 9-tier compatibility contract (garden/compat.py); consumed by ref with no kernel edits; in-browser IDE + Jupyter-like notebook + copilot. SHIPPED.
- Genuinely non-MMM Bayesian families ride the same fit→estimand→serialize→report pipeline: worked Confirmatory Factor Analysis and Latent Class Analysis models, plus a joint latent-factor MMM (measurement block + MMM in one graph to close a demand-confounding back-door). SHIPPED as worked garden examples; EFA/rotation and a reusable src primitive deferred.
- Off-panel experiment calibration: an experiment run in a period the model was NOT fit on still calibrates, by evaluating the channel's global response curve at the experiment's spend (steady-state or cold-start carryover) — no training-window overlap required. SHIPPED.
- In-graph experiment-calibration likelihood on contribution/ROAS/mROAS estimands (calibration/likelihood.py), plus lift-test→beta-prior derivation — fuses causal experiment evidence into the posterior, not just as a soft prior. SHIPPED.
- Pre-fit Model Design Readout: an HTML pre-registration document (priors enumerated, prior-predictive checks, prior-implied ROI, and a Simulation-Based Calibration verdict) generated BEFORE fitting to make the design auditable. SHIPPED.
- Experiment economics + optimizer: model-anchored opportunity-cost/short-term-risk analysis, A/A & A/B empirical-power simulation (autocorrelation-corrected false-positive rate), and a 4-objective (MDE × power × cost × duration) Pareto experiment optimizer. SHIPPED.
- Simulation-Based Calibration (diagnostics/sbc.py) as a routine posterior-calibration diagnostic with rank-hist/ECDF-diff charts and an agentic interpreter. SHIPPED.
- Impression/click measurement descriptors: per-channel ROI-vs-efficiency resolution so channels modeled on impressions/clicks report efficiency-per-1k with a break-even of 0 instead of a nonsensical dollar ROI. SHIPPED.

**Honest gaps:**

- Single primary author (Matthew Reda) at version 0.2.0 — this is an ambitious individual/research project, not a team-backed mature commercial product; bus-factor, support SLAs, and long-term maintenance are real risks vs an established vendor.
- Data-connector ecosystem is thin: the Meta/Google Ads/TikTok connectors are explicit stubs (raise NotImplementedError). Real ingestion is CSV/Excel/MFF upload + GCS + BigQuery + S3 storage. No turnkey pull from ad platforms, CRM, or analytics tools.
- Marketing/README language says it 'extends/builds on PyMC-Marketing', but the shipped core engine is a standalone PyMC reimplementation; pymc-marketing is only an optional report-reading extractor. Buyers expecting the PyMC-Marketing ecosystem/community should know the modeling code is bespoke.
- No vendor-managed hosted SaaS. Running it multi-tenant requires self-deploying (GCP Terraform / GCE appliance or k8s) and building a sandbox kernel container image; the hosted security posture is present but operator-assembled, not a polished cloud product.
- Non-MMM families (CFA/LCA, joint latent-factor MMM) ship as worked garden EXAMPLES demonstrating the pipeline, not as a general, documented modeling API; EFA/rotation, multiple latent factors, and per-geo factors are explicitly deferred.
- Benchmarks and validation evidence are internal/synthetic (self-generated DGP worlds and a rolling-origin backtest on limited real data) — there are no published third-party or industry-standard accuracy benchmarks or competitive bake-offs.
- Extended models are partially feature-complete: MultivariateMMM/CombinedMMM keep a joint MvNormal-LKJ likelihood and cannot take per-outcome non-Gaussian families; several documented items (in-graph experiment-calibration ROI bridge for impression channels, time-varying CPM, ragged-panel spend columns) remain deferred.
- Commercial/compliance maturity is aspirational: SOC2/security-questionnaire/operations-runbook documents exist as readiness plans, and the go-to-market is described as a design-partner program — not certified, not a proven at-scale product.
- Continuous-learning bandit is deliberately additive (multiplicative effects not implemented), and its guarantees rest on stated assumptions (stable geo sets, structural stationarity, randomized local designs); it is powerful but research-grade and requires careful operation.
- Very large surface area (~131k LOC of src, 292 modules, ~193 test files) built rapidly by one author: breadth may outpace hardening in places, and much of the newest functionality (agent, learning, garden, non-MMM) is recent and lightly battle-tested in production settings.

**Experiment calibration.** See calibration field — kept consistent.

**Experimentation.** Unusually deep for an MMM tool and one of its strongest differentiators. A full adaptive measurement loop: EIG/EVOI experiment prioritization ranks which channels to test next (planning/eig.py, evoi.py, priority.py); a pre-registered experiment lifecycle registry (draft→planned→running→completed→calibrated) enforces legal transitions and audits readouts; experiment designers cover randomized matched-pair geo lift, matched-market DiD (with power+placebo math), and budget-neutral national flighting; model-anchored economics compute per-channel incremental ROAS, opportunity cost / short-term risk (signed net-$), and A/A·A/B empirical power/MDE simulations that correct the analytic false-positive rate for autocorrelation; and a 4-objective Pareto optimizer trades MDE × statistical power × short-term cost × duration and recommends a runnable setup (test/control geo groups or flighting schedule + cool-down). All exposed as agent tools and REST endpoints with a React DesignStudio. planning/.

**Continuous learning.** A distinctive, self-contained module (continuous_learning/) that is the inverse of the classic MMM: a model-free Bayesian geo response-surface bandit that learns spend→outcome DIRECTLY from designed geo experiments with NO pre-fit MMM. Backed by NumPyro/JAX so one differentiable response surface serves the likelihood, the DGP, and the allocator. Pluggable activations (Hill, logistic, and a shape-agnostic monotone I-spline), sign-informed interaction priors for cannibalization/complementarity, central-composite experimental designs with shutoff cells, Thompson-sampling allocation under posterior draws, a marginal-ROAS 'funding line' decision rule, expected-regret/ENBS stopping, and cheap Laplace knowledge-gradient / D-optimal EIG design selection (milliseconds vs NUTS-refit). Handles count (NegBinomial) and heavy-tailed (Student-t) KPIs, information discounting + P-spline shrinkage for drifting media behavior, CUPED variance reduction, and adstock pre-pass. Fully wired to agent/API/UI (the 'Sextant' page). Deliberately additive (multiplicative effects documented as not implemented).

**Deployment.** Self-hosted, not a managed SaaS. The supported runtime is the FastAPI agent API (mmm_framework.api.main:app) + React/Vite frontend; the agent API runs fits in-kernel so Redis/ARQ is only needed by the legacy REST+Streamlit path. Multi-tenancy is provided by a dependency-free org/tenant JWT auth layer (auth/) with roles (owner/analyst/viewer), plans, rate limiting, audit logging, and a SQLite sessions/checkpoints store (relocatable via MMM_SESSIONS_DB). A hosted posture (MMM_AGENT_HOSTED=1) hardens isolation (container kernel, egress-deny, server-minted thread IDs) but requires the operator to build the sandbox kernel image. Deploy assets: GCP Terraform + a GCE VM 'appliance' (deploy/gcp) and Kubernetes manifests (deploy/k8s: gVisor RuntimeClass, egress-deny NetworkPolicy, warm-pool, HPA, kernel pod template). Readiness docs (SOC2, disaster-recovery, secrets-rotation, operations-runbook) exist as plans. Bottom line: deployable and security-conscious for a technical operator, but it is BYO-infrastructure, not turnkey.

**Tech stack:**

- Python 3.12+ (Apache-2.0, v0.2.0)
- PyMC 6.0+ / PyTensor 3.x / ArviZ 1.x (DataTree era; version drift centralized in utils/arviz_compat.py)
- NumPyro 0.19+ + JAX (fast NUTS, and the sole backend for the continuous-learning bandit)
- NutPie 0.16+ (alternative NUTS), Numba 0.63+ (JIT)
- pymc-extras (Pathfinder VI, MAP/ADVI approximate fits)
- FastAPI 0.124+ backend; LangGraph 1.x + langchain-{anthropic,openai,google-genai,google-vertexai} for the Oracle agent; multi-provider LLM incl. LM Studio
- Redis 7 + ARQ (legacy async job queue; the agent API runs fits in-kernel via jupyter_client/ipykernel — in-process, subprocess, or podman container sandbox)
- React 18 + TypeScript + Vite + Tailwind 4 + Zustand + TanStack Query + Plotly.js + KaTeX (modern UI); Streamlit 1.52 legacy UI (deprecated)
- Plotly 6.5+ (charts), python-pptx (slide decks), Pydantic 2.12+ (config validation)
- Optional: google-cloud-storage/bigquery (gcp extra), boto3 (s3 extra); SQLite for sessions/auth/checkpoints
- Deploy: GCP Terraform + GCE VM appliance, Kubernetes manifests (gVisor RuntimeClass, egress-deny NetworkPolicy), Containerfile kernel sandbox
- Sphinx API docs + a hand-authored static docs site (docs/*.html)

### Meta Robyn

**Type:** Open source &nbsp;|&nbsp; **Verification confidence:** high

**Vendor:** Meta / Meta Marketing Science. Open-source R + Python package under the "facebookexperimental" GitHub org; MIT-licensed. R-package maintainer of record: Bernardo Lares (authors incl. Gufeng Zhou, Igor Skokan, Leonel Sentana); copyright holder Meta Platforms, Inc.

**Methodology.** Frequentist/ML, NOT Bayesian in the MCMC/posterior sense. Core estimator is ridge (L2-regularized) linear regression (via glmnet), chosen to shrink rather than zero-out coefficients under heavy media multicollinearity. Media variables are transformed by adstock — geometric decay (1 parameter, theta) or Weibull CDF/PDF (2 parameters each, flexible/lagged decay) — then Hill saturation (alpha = curve shape, gamma = inflection) before regression. Trend, seasonality, holiday and weekday baselines are decomposed automatically with Facebook Prophet, reducing manual baseline specification. The distinctive engine is Nevergrad (Meta's gradient-free evolutionary optimizer) running MULTI-OBJECTIVE hyperparameter search over adstock/saturation/lambda/train-split params, minimizing up to three normalized objectives simultaneously: NRMSE (predictive fit), DECOMP.RSSD (a business-logic penalty on the gap between a paid channel's spend share and its modeled effect share), and MAPE.LIFT (calibration error vs experiments, active only when lift data is supplied). This yields a Pareto front of non-dominated candidate models (recommended >=2000 iterations x >=5 trials ~= 10,000 runs) from which the analyst selects. Budget allocation is a separate downstream step using gradient-based nonlinear optimization (nloptr: AUGLAG global + SLSQP local) on the fitted response curves. NUANCE (from Robyn's own docs): ridge is mathematically equivalent to a Bayesian MAP estimate under a normal prior — but Robyn produces point estimates with NO posterior sampling, so it is not Bayesian in the sense of Meridian/PyMC-Marketing.

**Strengths:**

- Free and open source under the permissive MIT license (private + commercial use)
- Automates hyperparameter tuning via evolutionary search, reducing manual specification and some researcher degrees of freedom
- Ridge handles the multicollinearity endemic to media spend data without zeroing channels
- Prophet decomposition automates trend/seasonality/holiday baselines so analysts do not hand-craft dummies
- Supports rigorous calibration to incrementality experiments / geo lift tests via MAPE.LIFT; Robyn docs cite (from a third-party Analytic Edge whitepaper) that uncalibrated models average ~25% difference vs ground truth
- Strong transparency and visualization for interpretation and stakeholder communication
- Backed by Meta with an active open-source community, CRAN distribution, documentation, and a peer-style methodology paper (arXiv:2403.14674)
- Well suited to data-rich digital / direct-response advertisers with many channels and granular data

**Limitations:**

- Produces point estimates, not genuine Bayesian posteriors; uncertainty is approximated (k-means clustering of Pareto models + bootstrapped ROAS/CPA CIs) and reflects model-selection spread rather than principled posterior uncertainty
- No informative priors: business knowledge can only enter via hyperparameter bounds or the calibration objective, not as prior distributions, so it has heavier data demands and higher overfitting risk on short/sparse datasets
- DECOMP.RSSD objective is controversial/ad hoc; critics argue penalizing spend-vs-effect divergence can bias toward 'not telling marketers they were wrong' (Recast)
- Model selection is subjective: Nevergrad returns roughly ~100 Pareto-optimal models that can imply quite different channel ROIs, and the analyst must pick one
- Instability across refreshes (channel ROAS can swing substantially between runs) — a Recast example cites Facebook ROAS falling 0.75 -> 0.2 over 5 refreshes (competitor-sourced, illustrative)
- No native hierarchical / geo-level random-effects modeling with partial pooling — typically operates on national aggregate time series with one coefficient set; contrasts with Google Meridian
- Calibration is bolted on as an optimization penalty rather than integrated probabilistically into the estimator; experimental uncertainty is not propagated into a posterior
- Reach & frequency and time-varying coefficients are not first-class inputs (R&F only via the beta curve calibrator; coefficients are static)
- Python port is an LLM-translated beta acknowledged as potentially buggy; R remains the reference implementation
- Requires R/Python and statistical expertise to run and to judge model plausibility; potential conflict-of-interest concern since the platform (Meta) authored the tool

**Experiment calibration.** Calibration is a core, documented capability, implemented as an optimization objective rather than as Bayesian priors. When the user supplies experimental ground truth (randomized conversion-lift studies or geo lift tests) to robyn_inputs(), Robyn activates MAPE.LIFT as a third Nevergrad objective, penalizing candidate models whose implied incremental effect diverges from the experiment. Robyn's docs cite a third-party Analytic Edge whitepaper stating uncalibrated models show ~25% average difference vs ground truth (NOT a Meta-originated benchmark). Separately, a beta curve calibrator (robyn_calibrate(), shipped v3.12.0, Dec 2024) uses cumulative reach-and-frequency data (Project Halo) to bound saturation-curve parameters (e.g., using a reach-1+ inflection as a lower boundary for gamma), addressing the identifiability problem that aggregate data cannot uniquely pin down saturation shape. Because calibration is a penalty term in a multi-objective search rather than a prior, it steers model selection toward experiment-consistent solutions but does not propagate experimental uncertainty into a posterior the way a Bayesian tool (Meridian, PyMC-Marketing) does.

**Pricing.** Free. MIT-licensed open source with no license, seat, or subscription fee; total cost is analyst time and compute. Any cost arises only from third-party consultants or hosting built around it.

**Deployment.** Self-hosted open-source library — R primary (on CRAN), Python beta. No Meta-hosted SaaS. Runs in the user's own environment; a Meta MMM API connector demo exists for pulling ad data. Third parties offer managed/consulting services built on it.

**Target users.** In-house marketing data scientists / analysts at brands and agencies, plus consultancies — particularly data-rich digital and direct-response advertisers with many channels and granular data. Assumes R/Python fluency and enough statistical judgment to evaluate and select among Pareto-front candidate models. Part of Meta's stated mission to 'democratise' MMM knowledge.

**Maturity.** Mature, widely adopted open-source MMM, first open-sourced by Facebook around 2020-2021 (exact initial public-release date not firmly pinned by primary sources; sources vary between late-2020 beta and a 2021 broader launch). Actively maintained and NOT archived/deprecated as of this check (July 2026): latest CRAN release is R version 3.12.1 (published 2025-07-02); the GitHub R release v3.12.0 was 2024-12-19; roughly 30-40+ releases total across the history. Point-in-time community metrics (vary by source and date): ~1.5k GitHub stars, ~427 forks, on the order of ~30 contributors. A robyn_refresh() model-refresh workflow exists but docs flag it as not thoroughly tested. Frequently benchmarked against Google Meridian (the Bayesian successor to Google's LightweightMMM), Uber Orbit, and PyMC-Marketing.

**Sources:** <https://github.com/facebookexperimental/Robyn> · <https://github.com/facebookexperimental/Robyn/releases> · <https://facebookexperimental.github.io/Robyn/docs/features/> · <https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/> · <https://cran.r-project.org/package=Robyn> · <https://arxiv.org/abs/2403.14674> · <https://getrecast.com/facebook-robyn/> · <https://ppc.land/metas-robyn-who-really-benefits-when-a-platform-builds-your-mmm/> · <https://facebookexperimental.r-universe.dev/Robyn>

---

### Google Meridian

**Type:** Open source (Apache-2.0) &nbsp;|&nbsp; **Verification confidence:** high

**Vendor:** Google — the Meridian Marketing Mix Modeling team (Google's ads measurement / data-science org; publicly associated with Harikesh Nair, Sr. Director of Data Science & Engineering). Not the "Google Ads / Marketing Analytics measurement team" label used in the draft.

**Methodology.** Hierarchical Bayesian causal-inference MMM fit by full MCMC (No-U-Turn Sampler) on TensorFlow Probability, with out-of-the-box GPU acceleration via tensors — a switch from the deprecated LightweightMMM's NumPyro/JAX stack. Normal likelihood (residuals ~ Normal(0, sigma_g)) on a geo-level (or national) hierarchical regression: media effects are geo-specific random coefficients (partial-pooled across geos) but constant over time, while adstock and Hill saturation parameters (ec, slope) are shared/common across geos. Media enters as a normalized weighted-decay adstock (geometric decay by default; configurable via adstock_decay_spec — not the 'binomial' adstock of LightweightMMM) composed with a Hill saturation curve. Reach & frequency channels are modeled as Adstock(reach x Hill(frequency)). Trend/seasonality via a knot-interpolated time-varying intercept. Distinctive ROI/mROI-parameterized LogNormal priors (beliefs placed on channel ROI directly), which is also the experiment-calibration mechanism. Outputs full posteriors and credible intervals, not point estimates.

**Strengths:**

- Free and fully open source (Apache-2.0) with transparent, inspectable code and methodology
- Backed and actively maintained by Google — frequent (roughly monthly) releases; v1.7.0 shipped June 18, 2026
- Genuine Bayesian uncertainty — full posteriors and credible intervals on ROI/effects, not point estimates
- Built for causal inference: first-class experiment/lift-test calibration and ROI priors reduce reliance on observational fit alone
- Reach & frequency modeling is rare among open MMM tools and valuable for video/YouTube-heavy plans
- Geo-level hierarchical structure extracts more signal and gives regional insight where geo data exists
- No-code Scenario Planner (Looker Studio) genuinely lowers the barrier for non-technical marketers to run budget scenarios over a fitted model
- Strong opinionated documentation, migration guides, and a certified-partner ecosystem for paid support
- Integrates cleanly with Google's own MMM Data Platform signals (GQV, impressions/clicks/cost, YouTube R&F) when available

**Limitations:**

- Fixed TensorFlow Probability backend — no swappable samplers; a PyMC-Labs benchmark (by the vendor of the competing PyMC-Marketing) measured it ~2–20x slower in effective-samples/second and reported convergence failure at its largest 'enterprise' synthetic scale — competitor-sourced, not independently reproduced here
- Compute-intensive: NUTS can take many minutes to hours on realistic datasets and a CUDA GPU is effectively recommended at scale; macOS/Apple-Silicon is CPU-only (GPU path is CUDA/and-cuda)
- Media effects are held constant over time (only the intercept is time-varying) — no time-varying media coefficients, and Google states it does not support time-varying priors on knot values
- Cannot model channel synergies / interaction effects (Google-stated FAQ limitation)
- Explicitly designed for causal attribution, not forecasting — Google states it cannot forecast raw future outcomes
- Requires Python and applied-Bayesian literacy to build and validate; steep for non-technical users (partly mitigated by the Scenario Planner)
- Same PyMC-Labs benchmark reported lower in-sample fit (R^2 ~0.73 vs ~0.87) and higher channel-contribution RMSE (~0.70 vs ~0.41) on their synthetic data — directional, competitor-sourced, not independently reproduced
- No Google-hosted managed SaaS for the modeling itself — the org must run, tune, and validate the models (the Scenario Planner is a reporting/optimization layer over a self-fitted model, not a hosted fitting service)

**Experiment calibration.** Calibrates channel ROI to incrementality experiments (geo lift, conversion-lift, A/B) by converting an experiment's point-estimate+SE (lognormal_dist_from_mean_std) or 95%-CI range (lognormal_dist_from_range) into a LogNormal ROI prior via built-in helpers; also supports priors from prior MMMs, benchmarks, and SME judgment. Google's docs explicitly warn that experiment and MMM estimands differ (MMM's zero-spend counterfactual vs an experiment's baseline; timing/geography/duration/campaign-setting mismatch), so calibration is guided-manual, not automatic. Default priors are described as moderately-informative LogNormals when no experiment exists (exact default numeric values not fully enumerated in the doc excerpts reviewed).

**Pricing.** Free, open source under Apache-2.0 — no license or seat fees. Costs are compute (GPU time for NUTS) and in-house analyst time; optional paid consulting via Google's certified Meridian partner network. The Scenario Planner (Looker Studio) is likewise free to use over your own fitted model.

**Deployment.** Self-hosted open-source Python library (pip install google-meridian; docs state Python 3.11–3.13; PyPI metadata says >=3.10; [and-cuda] extra for GPU, plus jax/mlflow/colab/scenarioplanner extras). Runs in Colab/Jupyter/cloud VM; GPU recommended for MCMC. No Google-managed SaaS for model fitting. The Meridian Scenario Planner (Looker Studio + a scenarioplanner module) provides a no-code scenario-planning/optimization layer over a fitted model; certified partners offer paid implementation.

**Target users.** In-house data scientists and marketing-science analysts at brands/advertisers, agencies, and measurement consultancies (incl. Google-certified Meridian partners). Building a model requires Python + applied-Bayesian skill; business users consume results via the no-code Scenario Planner / Looker Studio layer or through a partner.

**Maturity.** Announced by Google in 2024; first PyPI release v1.0.0 on Jan 27, 2025 and made 'open to everyone' (GA) on Jan 29, 2025. Positioned as the official successor to Google's LightweightMMM, which is deprecated and no longer actively developed (Google publishes a migration guide). Apache-2.0 licensed; current release v1.7.0 (June 18, 2026) with 31 releases on PyPI (v1.0.0 → v1.7.0; roughly monthly cadence). GitHub adoption is moderate (~1.4k stars, ~275 forks as of mid-2026) — smaller than Meta's Robyn or PyMC-Marketing communities, but growing, Google-backed, with a formal certified-partner ecosystem.

**Sources:** <https://github.com/google/meridian> · <https://pypi.org/project/google-meridian/> · <https://developers.google.com/meridian/docs/advanced-modeling/model-spec> · <https://developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration> · <https://developers.google.com/meridian/docs/faqs> · <https://developers.google.com/meridian/docs/scenario-planning/meridian-scenario-planner> · <https://developers.google.com/meridian/reference/api/scenarioplanner> · <https://developers.google.com/meridian/docs/migrate> · <https://blog.google/products/ads-commerce/meridian-marketing-mix-model-open-to-everyone/> · <https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-google-meridian> · <https://github.com/google/meridian/discussions/1335> · <https://github.com/google/lightweight_mmm>

---

### PyMC-Marketing (with PyMC Labs consulting + MMM Agent beta)

**Type:** Open source (Apache 2.0 library) + Enterprise/consulting (PyMC Labs services). A separate proprietary "MMM Agent" AI assistant built on top of the library is in invite-only/gated beta (commercial model not disclosed). &nbsp;|&nbsp; **Verification confidence:** high

**Vendor:** PyMC Labs — a Bayesian AI consultancy founded by core PyMC developers (e.g. Thomas Wiecki); the open-source library is maintained by PyMC Labs and the PyMC community. PyMC Labs also maintains the related CausalPy library.

**Methodology.** Fully Bayesian probabilistic programming built on PyMC / PyTensor. MMMs are estimated by MCMC with the No-U-Turn Sampler (NUTS) across pluggable backends (NumPyro/JAX, BlackJAX, Nutpie) with optional GPU acceleration; MAP/variational are available but MCMC is the default and recommended path. The MMM follows the standard adstock-carryover + saturation-shape Bayesian MMM formulation popularized by Google's Jin et al. (2017) "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." It produces genuine full posterior distributions (credible intervals on ROAS, contributions, budgets) rather than the point estimates of ridge/OLS approaches like Meta's Robyn.

**Strengths:**

- Genuine, well-calibrated uncertainty: full Bayesian posteriors and credible intervals on ROAS, contributions and budget scenarios
- Highly flexible and extensible — custom priors, custom adstock/saturation, custom model terms, arbitrary hierarchical dims; more configurable than Meridian or Robyn
- Multiple fast NUTS backends (NumPyro/BlackJAX/Nutpie) + GPU; PyMC Labs' own benchmarks claim it samples faster than Meridian and scales to enterprise datasets
- Principled experiment integration (lift-test / geo-lift likelihoods) so RCT ground truth updates the model in-inference rather than being bolted on post-hoc
- Open source and free under Apache 2.0 — no seat/licensing lock-in, transparent, unit-tested code
- Backed by the creators of PyMC with an active community (Discourse, Discord, MMM Hub Slack) and frequent releases
- Broad scope beyond MMM (CLV/BTYD, customer choice, Bass diffusion) from one toolbox
- Time-varying media coefficients and baseline — which, per PyMC Labs' comparison table, Robyn and Meridian do not offer for coefficients

**Limitations:**

- Steep learning curve — requires Python plus real Bayesian/PyMC expertise; it is a library/framework, not a turnkey product
- Slow iteration relative to Robyn's ridge approach: a full MCMC run can take from minutes to hours depending on data size, model complexity and hardware
- No documented native reach & frequency model class (Meridian offers dedicated R&F modeling and tight Google/YouTube-ecosystem integration); inferred from absence in docs — treat as high-but-not-certain
- API churn / maturity risk: the classic mmm.MMM class is deprecated and scheduled for removal in 0.20.0 in favor of multidimensional.MMM, forcing migration; still pre-1.0 (v0.19.x)
- No first-party hosted SaaS/BI dashboard in the open-source library — hosting, scheduling and stakeholder UIs are DIY (or via the still-beta MMM Agent / paid consulting)
- The headline 'faster and more accurate than Meridian' figures (~2-20x speed, ~40% lower channel-contribution error, ~19% higher R2) are PyMC Labs' OWN benchmarks, not independent results
- Correct model specification (priors, likelihood, convergence diagnostics) is on the user; misuse can yield overconfident or biased estimates
- Needs meaningful data history (typically 1-2+ years) and analyst time to set up well

**Experiment calibration.** Experiments/ground truth enter as additional likelihood terms, not post-hoc adjustments. add_lift_test_measurements() forms the model's difference in a channel's saturation curve between spend x and x+delta_x and conditions it on the observed lift (delta_y) and its standard deviation using a specified distribution (Gamma by default, since the curve is monotonic), constraining channel parameters during inference and improving identifiability for correlated channels. A geo-lift calibration workflow lets a subset of geos with lift tests update population-level (hierarchical) parameters, improving estimates for untested geos. Domain knowledge and prior experiment results can also be encoded via custom prior distributions, and DAG-based causal identification is offered to mitigate confounding. Net effect: posteriors that blend historical time-series signal with experimental ground truth.

**Pricing.** Library: free and open source under Apache 2.0 (unrestricted commercial use, no seat/subscription fees). PyMC Labs consulting/training: custom, quote-based enterprise engagements — no public pricing. MMM Agent: pricing/commercial model not published (invite-only beta, apply for access). No per-seat or per-model SaaS pricing is publicly available for any tier.

**Deployment.** Self-hosted open-source Python library (pip/conda install pymc-marketing; Python >=3.12). Runs anywhere Python/PyMC runs, with optional GPU acceleration and MLflow for experiment tracking; models serialize (netCDF/Zarr) for refresh/deployment. There is no official hosted SaaS for the core library — production deployment, scheduling and stakeholder UIs are the user's responsibility. Two managed paths exist: (1) PyMC Labs consulting engagements (custom models, integration, training, an Expert Access Program) and (2) the proprietary AI 'MMM Agent', an invite-only/gated beta.

**Target users.** Data scientists and quantitative marketing/analytics engineers comfortable with Python and Bayesian statistics; in-house analytics teams at mid-market to enterprise brands; agencies and consultancies building bespoke MMM/CLV pipelines. Less suited to non-technical marketers wanting a point-and-click tool (that audience is the target of the MMM Agent / consulting). Known PyMC Labs clients/adopters include HelloFresh, Colgate-Palmolive, Haleon, Fox Entertainment and Fabletics; PyMC Labs cites 50+ organizations served.

**Maturity.** Mature, actively developed open-source project, still pre-1.0. Latest release 0.19.4 on May 6, 2026 (PyPI; classified Production/Stable, supports Python 3.12-3.14). 40+ releases total; the 0.19.x line in 2026 focuses on the multidimensional.MMM rewrite, discrete-choice models, serialization/scaling stability and interactive plotting. GitHub (mid-2026): ~1.2k stars, ~385 forks, growing contributor base. Underlying PyMC/PyTensor/ArviZ stack is well established; PyMC Labs reports tens of millions of PyMC downloads across the ecosystem (vendor-reported).

**Sources:** <https://pypi.org/project/pymc-marketing/> · <https://github.com/pymc-labs/pymc-marketing> · <https://www.pymc-marketing.io/en/stable/guide/mmm/comparison.html> · <https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_lift_test.html> · <https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_geolift_calibration.html> · <https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mmm.multidimensional.MMM.html> · <https://www.pymc-labs.com/blog-posts/ai-mmm-agent-beta> · <https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-google-meridian> · <https://www.pymc-labs.com/> · <https://www.pymc-labs.com/team> · <https://github.com/pymc-labs/pymc-marketing/releases>

---

### Recast (Recast MMM)

**Type:** Commercial SaaS (proprietary, closed-source; high-touch managed service, not a self-serve open-source library) &nbsp;|&nbsp; **Verification confidence:** high

**Vendor:** Recast Labs, Inc. (getrecast.com) — headquartered in Brooklyn / New York, NY (some registries list Albany, NY). Co-founded in 2019 by Michael Kaminsky and Thomas ("Tom") Vladeck, both statisticians (Vladeck serves as CEO/co-CEO). Venture-backed and still seed-stage: ~$1.1M pre-seed (2021, led by Good Friends) plus a $3.4M seed round announced Dec 15, 2022 (led by Lerer Hippeau; Good Friends, Data Tech Fund, Vibe Capital and angel investors participating) — roughly $4.5M raised in total. No later funding round surfaced as of mid-2026. Small team (~11–50 employees; ~13 reported early 2024).

**Methodology.** Fully Bayesian hierarchical time-series MMM written in Stan (vendor cites 8,000+ lines) and fit via Hamiltonian Monte Carlo (HMC/NUTS) — genuine MCMC posteriors, not point estimates. Multiple components (intercept, per-channel ROIs, and a shared cross-channel saturation multiplier) each follow a locally-periodic Gaussian Process, so channel effectiveness and seasonality evolve over time (time-varying coefficients). Adstock/time-shift is modeled with a negative-binomial distribution over integer lags (an atypical, documented choice rather than conventional geometric-decay adstock). Saturation uses a single-parameter Hill function (a.k.a. the Monod equation). Explicit demand-generation → demand-capture (upper→lower funnel) structure with hierarchical pooling across channels. Spend-based only — no impressions/clicks and no reach & frequency. Automated weekly refresh with no manual coefficient hand-tuning. Promotions/spikes are modeled with pull-forward/push-back effects (the exact distributional form is per vendor docs and was not independently re-verified in this pass).

**Strengths:**

- Genuine full Bayesian uncertainty quantification (HMC/Stan posteriors) with credible intervals on all outputs — not point estimates
- Time-varying coefficients are a real methodological differentiator vs static-coefficient MMMs (including many open-source tools)
- Rigorous prospective validation culture: out-of-sample forecast scoring (1 − scaled CRPS on cumulative predictions) surfaced on live weekly dashboards, explicitly rejecting in-sample fit as the quality metric
- Statistically principled experiment calibration — a lift test becomes a prior on ROI only at the dates the test ran, propagating uncertainty as experiments go stale and sidestepping the 'which conflicting lift test do I trust' problem
- Always-on managed service: weekly refresh with dedicated marketing data scientists (vendor claims >20% of staff hold a PhD in math/stats/engineering/economics)
- Deep Stan/probabilistic-computing expertise — vendor states it employs several core Stan contributors (corroborated in its own methodology docs)
- Unusual transparency for a commercial vendor — public methodology docs and exposed causal/model assumptions
- Privacy-durable and iOS14-resilient (spend-based, cookieless, no user-level data)
- Credible enterprise adoption claimed (Block/Square/Cash App/Afterpay, Canva, Harry's, Rocket Money, PODS, GetYourGuide, Daily Harvest — vendor-reported)

**Limitations:**

- Proprietary and closed-source — despite public docs, the actual Stan code cannot be independently audited or self-hosted; you are locked into the managed service
- Expensive and gated to larger advertisers — vendor recommends ~$10M+ annual media spend (roughly a $5M floor for enough signal); not viable for small brands
- Opaque, custom enterprise pricing (no public price list); third-party price estimates conflict widely and are unverified
- High-touch managed model with a ~2–3 week initial setup, then new channels take additional weeks to stabilize — slower and more vendor-dependent than a self-serve library
- No explicit reach & frequency modeling — fits on spend only (contrast Google Meridian, which added R&F)
- Primarily national/top-down time-series; designed geo experimentation is a separate product (GeoLift), adding cost/complexity for incrementality testing; parity of geo-panel modeling with national modeling is not clearly documented
- Heavy HMC/Stan estimation carries the sampling pitfalls Recast itself documents (divergences, tree-depth saturation, multimodality) and is computationally intensive
- Small, seed-stage company (~11–50 staff, ~$4.5M raised, no round since 2022) — vendor-longevity/scale risk vs Google/Meta open-source ecosystems or larger measurement vendors
- Requires a long data history (18+ months, ideally ~27), limiting newer brands
- Headline '95% accuracy' is a 1 − scaled-CRPS framing on cumulative predictions, not a simple MAPE — easy to misread without context

**Experiment calibration.** Incrementality/lift tests are folded directly into the Bayesian model as a TIME-LOCALIZED prior on ROI: rather than a single constant ROI prior, the experiment's result (with its uncertainty) tightens the ROI prior ONLY over the dates the test ran, with uninformative priors elsewhere. Because ROI is estimated over time and the prior is date-scoped, uncertainty propagates correctly as experiments become 'stale,' and multiple/conflicting lift tests are each treated as a point-in-time snapshot rather than one overriding truth. Recast is explicit that the MMM is NOT dependent on experiments to produce results (it is an assumption-heavy observational model), but recommends lift tests — via its own GeoLift synthetic-control tool or third-party tests — as the ground-truth calibration path because experiments require fewer assumptions for causal claims. Priors also encode business domain knowledge broadly. (Confirmed against getrecast.com/how-to-use-lift-tests-to-calibrate-an-mmm and docs.getrecast.com/docs/experiments.)

**Pricing.** Not publicly listed — a custom/enterprise annual subscription negotiated via sales, scaled by media spend under measurement, refresh cadence, KPIs/channels and support scope. Third-party estimates conflict and are unverified: an affiliate review cited ~$8,000 (single KPI, ~12–20 channels) and ~$4,000 for a limited tier (cadence not stated), while industry aggregators estimated roughly $50K–$300K+/year (some cite $5K–$50K/month). Practical entry point is a large-advertiser budget (~$10M+ annual media spend recommended; ~$5M signal floor). GeoLift by Recast is priced separately and cheaply — free for the first 6 months, then starting at ~$100/month. Treat all specific figures as approximate.

**Deployment.** Hosted, cloud-native SaaS on AWS with a web platform; SOC 2 compliant. Delivered as a high-touch MANAGED service — Recast configures and maintains each client's model (dedicated account/data-science support), backed by a self-serve web app for planning, optimization and dashboards. Not self-hostable and not an open library; ~2–3 weeks of onboarding before the first weekly results.

**Target users.** Mid-market to enterprise consumer brands with substantial paid media (~$10M+/yr recommended; a ~$5M signal floor) that want an always-on managed measurement partner rather than to build in-house. Strong presence in DTC/e-commerce, subscription, fintech, CPG and similar. Buyers are typically CMO/VP-Marketing, growth/performance-marketing and finance/analytics teams; the in-house data-science lift is low because Recast operates the model. Named adopters (per vendor): Block (Square, Cash App, Afterpay), Canva, Harry's, Rocket Money, PODS, GetYourGuide, Daily Harvest, and others.

**Maturity.** Established, actively developed commercial product but a small, seed-stage company. Founded 2019; ~$4.5M raised (~$1.1M pre-seed 2021 led by Good Friends + $3.4M seed Dec 2022 led by Lerer Hippeau), no later round surfaced; ~11–50 employees. Vendor claims 3,000+ production models scored weekly and $5B+ (elsewhere $4B) in annual marketing spend guided — self-reported. Recent product motion includes GeoLift by Recast (launched Sept 30, 2025) and multi-stage/funnel models. No large public community (proprietary), but the founders publish substantial educational content (blog, MMM Academy) and full public methodology documentation.

**Sources:** <https://docs.getrecast.com/docs/recast-model-technical-documentation> · <https://docs.getrecast.com/docs/methodology-faq> · <https://getrecast.com/recast-llm-information/> · <https://getrecast.com/accuracy-dashboards/> · <https://getrecast.com/how-to-use-lift-tests-to-calibrate-an-mmm/> · <https://docs.getrecast.com/docs/experiments> · <https://getrecast.com/bayesian-methods-for-mmm/> · <https://getrecast.com/geolift-by-recast/> · <https://fox4kc.com/business/press-releases/ein-presswire/853748775/recast-launches-geolift-by-recast-with-free-trial-to-bring-incrementality-testing-to-every-marketing-team/> · <https://www.prnewswire.com/news-releases/recast-raises-3-4-million-for-automated-marketing-data-science-platform-301703759.html> · <https://www.alleywatch.com/2022/12/recast-marketing-mix-modeling-analytics-forecasting-platform-omnichannel-holistic-ad-spend-data-science-michael-kaminsky/> · <https://medium.com/lerer-hippeau-ventures/please-welcome-recast-the-automated-marketing-data-science-platform-for-the-digital-era-bb842512b438> · <https://www.mediaplanningtool.com/recast/pricing/> · <https://costbench.com/software/marketing-attribution/recast/> · <https://aazarshad.com/resources/getrecast-review/> · <https://getrecast.com/google-meridian/> · <https://getrecast.com/multi-stage-models/> · <https://www.crunchbase.com/organization/recast-28f3>

---

### Mutinex GrowthOS

**Type:** Commercial SaaS (proprietary, closed-source model engine). Mutinex ALSO publishes a separate open-source tool, mmm-eval (Apache 2.0) — but that is a vendor-neutral MMM validation/benchmarking toolkit, NOT the GrowthOS model itself. The two must not be conflated. &nbsp;|&nbsp; **Verification confidence:** medium

**Vendor:** Mutinex (Mutinex Pty Ltd). Co-founded by Henry Innis (CEO) and Matt Farrugia (co-founder; roles reported variously as COO / Chief Customer Officer, moved to an APAC leadership role in 2024). Originally launched as "Mutiny," rebranded to Mutinex. Founding year reported as 2018 (SmartCompany) with some aggregators (Tracxn) listing 2017 — minor discrepancy. HQ South Melbourne, Victoria, Australia; expanding into the US (US leadership expanded in 2026 — Mike Finnerty named President, United States, Mar 2026) and NZ. Backers include Marbruck Investments (led the Oct 2024 round), EVP and Archangel Ventures. Total disclosed venture funding is roughly A$32M summing the named AUD rounds (~US$23.3M per Tracxn/aggregators — currency conventions differ across sources); ~A$132.5M valuation after the Oct 2024 A$17.5M round.

**Methodology.** Bayesian hierarchical econometric modeling, marketed as "commercial mix modeling." Bayesian inference is explicitly and repeatedly confirmed in Mutinex's own content (blog posts on "What is Bayesian inference?" and "Bayesian priors in MMM"); it is marketed as probabilistic — producing distributions of plausible ROI outcomes rather than single point estimates. Mutinex markets a time-varying baseline that adjusts to demand/competition/seasonality (to avoid crediting brand-driven demand to media). Headline differentiator is "Campaign-Varying MMM" (patent-pending, verified on Mutinex's own page): rather than one coefficient per channel over a period, it decomposes channel performance to the decision level — creative, ad format (e.g. 15s vs 30s), publisher/network, geography, audience segment, CPM. Supports geo-level and portfolio/category/brand hierarchical modeling. Mutinex describes the engine as "foundation modelling" trained across billions of dollars of cross-brand spend, implying pooled/network-informed priors rather than per-client-only estimation. CRITICAL GAP (independently confirmed): the specific inference engine/sampler (PyMC/Stan/NumPyro/proprietary MCMC or VI), and the exact adstock and saturation functional forms, are NOT publicly disclosed — the Campaign-Varying page and product pages are marketing collateral, not technical specs. (An SEO aggregator, mediaplanningtool.com, publishes numeric "Adstock/Saturation controls" and "Incrementality calibration" scores; these appear fabricated/unsourced, contradict the disclosed opacity, and were discarded.)

**Strengths:**

- Genuine uncertainty quantification — Bayesian posteriors / credible intervals rather than only point estimates (confirmed as the marketed approach in Mutinex's own Bayesian-inference content)
- Speed: model builds/refreshes in hours-to-24h (Hershey's ~4h) vs 6-8 weeks for legacy consulting MMM, enabling always-on decisioning
- Granularity below the channel level (creative/format/publisher/audience/CPM) via Campaign-Varying MMM — rare among MMM vendors
- Time-varying baseline intended to reduce misattribution of organic/brand demand to media
- Cross-brand 'foundation model' data scale (claimed 100+ brands, billions in spend) can inform priors and benchmarking
- Strong go-to-market: Dentsu exclusive agency 'Platinum Partnership' launch partner (ANZ, confirmed), plus WARC and Tracksuit integrations and notable named adopters (e.g. Hershey, Samsung, Intuit, Asahi)
- Credibility/transparency play via the open-source, vendor-neutral mmm-eval validation toolkit (Apache 2.0) — a genuine transparency signal even if the core model stays closed
- Integrated decision workflow (scenario planning + optimization + MAITE AI assistant) rather than a static report

**Limitations:**

- Methodological opacity (independently confirmed): sampler/inference engine, adstock and saturation functional forms, and reach & frequency handling are not publicly documented — buyers rely on vendor demos, not technical papers
- Reach & frequency support could not be verified either way from public sources (unknown, not confirmed present)
- Experiment/incrementality calibration is deliberately DE-emphasized: CEO Henry Innis publicly argues (WARC, 'Why forcing lift test priors makes for lazy models') that forcing lift tests as hard priors makes 'lazy models' and that experiments should VALIDATE models, not solely power/calibrate them — so it is not positioned as a first-class hard-prior calibration workflow (a competitor, Sellforte, claims Mutinex offers no geo testing/A-B/incrementality calibration — a biased, unverified source)
- Flagship tier is not self-serve — requires sales, DataOS data integration and setup; On-Demand adds self-serve for mid-market but is currently limited to 'single-node deployments only'
- Pricing is opaque (enterprise custom, no public tiers)
- Thin independent validation: third-party review volume is very low (G2 shows a single 2.5/5 review — statistically non-representative); the '33% lower predictive error than Google Meridian' figure is a Mutinex self-claim published on its own page, with no independent methodology shown
- Proprietary/closed — no self-hostable model/library; the open-source artifact (mmm-eval) evaluates OTHER frameworks and does not expose the GrowthOS model
- Reliance on cross-brand 'foundation modelling' priors is under-documented — hard to audit how much a given client's results are shaped by network priors vs their own data

**Experiment calibration.** Philosophy-driven rather than experiment-forced. Mutinex leans on Bayesian priors plus its cross-brand 'foundation modelling' data to inform effect priors, and uses a time-varying baseline to separate demand from media. CEO Henry Innis is publicly critical of forcing lift-test/incrementality results in as hard priors to calibrate MMM — his WARC piece 'Why forcing lift test priors makes for lazy models' argues models should recognize incrementality in the data and that experiments should hold models to account (validate), not merely power them. Note the nuance: Innis is NOT against all experiment use — the alternative he endorses (broad uninformative priors, applying experiment results only to the periods the experiment ran) is itself a calibration technique — so 'no calibration at all' would overstate it. Practically, experiment calibration appears available/possible but is not marketed as a core self-serve loop; WARC and Tracksuit add external benchmark and brand-equity signals. Exact prior specification and any formal experiment-likelihood mechanism are not publicly detailed.

**Pricing.** Not public. Enterprise custom pricing via sales consultation (no published tiers); described as an enterprise package bundling continuous MMM, MAITE AI, scenario planning and DataOS data integration — likely significant annual contract value. GrowthOS On-Demand targets mid-market/growth-stage at lower friction but its pricing is also undisclosed. mmm-eval (the validation toolkit) is free/open-source (Apache 2.0). (Third-party pages claiming to list Mutinex pricing are aggregators and are not authoritative.)

**Deployment.** Hosted, multi-tenant SaaS (cloud). No self-hostable model/library. Flagship GrowthOS is a sales-led enterprise engagement with DataOS-driven data integration and setup. GrowthOS On-Demand (launched ~May 2026) is a faster, more self-serve mid-market onboarding (agentic guided data upload, working model in <24h) but is currently limited to 'single-node deployments only' (confirmed in Mutinex's own On-Demand announcement). The only self-installable Mutinex artifact is the open-source mmm-eval CLI (pip install from GitHub; Python 3.11/3.12), which evaluates OTHER frameworks (PyMC-Marketing, Google Meridian), not GrowthOS.

**Target users.** Enterprise brand marketers, CMOs and marketing-finance/investment teams, media strategists and cross-channel planners at large advertisers (flagship GrowthOS); mid-market and growth-stage brands wanting fast MMM without agency involvement (GrowthOS On-Demand); agencies (via the Dentsu partnership). Positioned for marketing decision-makers rather than in-house data scientists — the modeling is abstracted behind the platform and MAITE. Data scientists/analytics engineers are more the audience for the separate open-source mmm-eval.

**Maturity.** Well-funded scale-up (founded ~2018, originally 'Mutiny'). ~A$132.5M valuation post the Oct-2024 A$17.5M round (led by Marbruck; up from ~A$75M after the ~A$9.5M Oct/Nov-2023 round and ~A$37M at the Feb-2023 A$5M seed extension). Claims 100+ global brands and billions in spend under analysis; named adopters include Hershey, Samsung, Intuit and Asahi. Dentsu is exclusive ANZ agency partner; WARC integration launched Nov 2025; Tracksuit API integrated; actively expanding into North America (US leadership expanded Mar 2026). The commercial product is mature and adopted, but externally-verifiable methodology documentation and independent user reviews remain limited. The open-source mmm-eval is young (v0.12.0, Jul 2025; ~29 GitHub stars — modest community traction).

**Sources:** <https://mutinex.co/product/growthos/> · <https://mutinex.co/insights/growthos-on-demand-is-here/> · <https://mutinex.co/insights/introducing-campaign-varying-mmm-go-beyond-the-channel/> · <https://mutinex.co/open-mmm-validation-framework/> · <https://github.com/mutinex/mmm-eval> · <https://mutinex.github.io/mmm-eval/user-guide/tests/> · <https://www.businesswire.com/news/home/20250916130465/en/Mutinex-Brings-Its-Vendor-Neutral-Open-Source-MMM-Validation-Framework-to-North-America-for-Marketers-Who-Demand-Proof-Versus-Promises> · <https://www.businessnewsaustralia.com/articles/marketing-analytics-scale-up-mutinex-raises--17-5m--boosting-valuation-to--132-5m.html> · <https://mutinex.co/2024/10/10/mutinex-raises-17-5m-aud-at-132-5m-valuation-set-to-fuel-us-expansion> · <https://mutinex.co/2023/10/30/mutinex-raises-9-and-a-half-million-aud-in-fresh-capital/> · <https://www.dentsu.com/au/en/news/news-releases/dentsu-signs-on-as-exclusive-agency-partner-of-econometrics-business-mutinex> · <https://www.businesswire.com/news/home/20251124431052/en/WARC-and-Mutinex-Unite-to-Give-Marketers-Their-Own-Answers-on-Demand-and-Growth-Operating-System> · <https://www.warc.com/newsandopinion/opinion/why-forcing-lift-test-priors-makes-for-lazy-models/en-gb/6674> · <https://www.g2.com/sellers/mutinex> · <https://tracxn.com/d/companies/mutinex/__lZweJ9GogJI2-TUcD2xJ7VBTpmEnbZxqUFNLKLNxJnw> · <https://www.smartcompany.com.au/startupsmart/mutinex-startup-raises-17-5-million-us-expansion/> · <https://www.businesswire.com/news/home/20260313941879/en/Mutinex-Bolsters-Its-U.S.-Presence> · <https://mumbrella.com.au/mutiny-rebrands-to-mutinex-senior-hires-and-leadership-reshuffle-follows-762604> · <https://www.gotracksuit.com/blog/posts/answering-the-cfo-question> · <https://sellforte.com/blog/mutinex-vs-sellforte-ai-for-mmm-and-incrementality-testing>

---

### Google LightweightMMM (lightweight_mmm)

**Type:** Open source (legacy / deprecated / archived) — Apache-2.0 Python library. No longer maintained; Google declares it "not supported anymore" and points users to its official successor, Meridian. GitHub repo archived (read-only) on 19 Jan 2026. &nbsp;|&nbsp; **Verification confidence:** high

**Vendor:** Google — written and published by Google engineers under the `google` GitHub org, but the repo and PyPI page both carry the disclaimer "This is not an official Google product," and support was always community/best-effort (now none).

**Methodology.** Fully Bayesian marketing-mix model built on NumPyro + JAX, estimated with MCMC (NUTS). Adstock/carryover and saturation are fit jointly in one probabilistic model. Likelihood on the KPI is Normal (CONFIRMED in models.py: dist.Normal(loc=mu, scale=sigma); sigma ~ Gamma(1,1)). Mean = intercept + polynomial trend (μ·t^κ, μ~Normal(0,1), κ~Uniform(0.5,1.5)) + harmonic/Fourier seasonality (γ~Normal(0,1); daily data adds weekday δ~Normal(0,0.5)) + control terms (~Normal(0,1)) + transformed media. Three media transforms: 'adstock' (retention λ_m~Beta(2,1) + exponent ρ_m~Beta(9,1)); 'hill_adstock' (adstock λ_m~Beta(2,1) then Hill K_m,S_m~Gamma(1,1)); 'carryover' (causal convolution τ_m~Beta(1,1), θ_m~HalfNormal(2) + exponent ρ_m~Beta(9,1)). Media coefficients β_m~HalfNormal(v_m) with v_m = channel total cost (cost-informed AND non-negativity-constrained). National and sub-national geo hierarchical (partial-pooling) models. No reach & frequency; time variation limited to polynomial trend + fixed harmonic seasonality (no splines/knots, no dynamic coefficients).

**Strengths:**

- Genuine Bayesian uncertainty — returns full posteriors and credible intervals for effects/ROI, not point estimates
- Joint single-model estimation of adstock + saturation (avoids Robyn's separate ridge + evolutionary two-stage pipeline)
- Native geo/hierarchical support that an independent reviewer (Recast) calls 'a killer feature' credited with large gains in plausibility/accuracy
- Fast and lightweight — minutes vs Robyn's hours (Recast) — thanks to JAX
- Free, open source (Apache-2.0), transparent, fully self-hostable
- Priors act as guard-rails and let you encode domain knowledge / previous-model results (custom_priors)
- Backed by Google's Bayesian-MMM research and widely used as a teaching/reference implementation

**Limitations:**

- DEPRECATED, unmaintained, and ARCHIVED — README states 'LMMM is not supported anymore' and 'highly recommends' switching to Meridian; final release v0.1.9 (23 May 2023); GitHub repo archived read-only on 19 Jan 2026; PyPI development status still '3 - Alpha' (never reached a stable 1.0)
- No reach & frequency modeling — responds to spend regardless of frequency, a real gap for video/YouTube (added in Meridian)
- No built-in experiment/lift-test calibration — folding in incrementality experiments is manual (Meridian's headline addition)
- Limited time-varying dynamics: polynomial trend + fixed harmonic seasonality only; no splines/knots, no dynamic coefficients — reviewers note it can't easily model channels getting better/worse over time
- Production instability — Recast reports re-running on slightly changed data can cause 'dramatic parameter shifts' (a campaign flipping from very effective to very ineffective), requiring expert prior management
- Media coefficients are HalfNormal, so effects are constrained non-negative — cannot represent a genuinely counterproductive/negative channel
- Seasonality treatment assumes marketing effectiveness doesn't change during peak-demand periods, which Recast argues can mislead spend decisions around events like Black Friday
- Fewer adstock/saturation transform options than Robyn (Recast notes Facebook/Robyn is 'way ahead' here), so 'cleaner than Robyn' is a trade-off, not a strict win
- Encoding richer domain knowledge often means code-level changes rather than a friendly API; weaker out-of-the-box reporting than Robyn/Recast
- No paid-search / GQV-specific handling (added in Meridian); requires manual data scaling and solid Python/Bayesian skill

**Experiment calibration.** Prior-based calibration only. custom_priors lets users encode domain knowledge or prior-model results, and the media-coefficient prior scale is cost-informed (β_m ~ HalfNormal(total channel cost)). There is NO built-in mechanism to fold in lift/incrementality experiments — doing so is manual — which is a primary reason Google positions Meridian (native experiment/lift-test calibration) as the replacement.

**Pricing.** Free — open source under Apache-2.0. No license, seat, or subscription cost. Only costs are self-hosted compute (CPU, or optional GPU/TPU) and analyst time.

**Deployment.** Self-hosted Python library (pip install lightweight-mmm). Runs locally, in notebooks/Colab, or on any server/cloud VM; JAX enables optional GPU/TPU acceleration. No hosted SaaS, no managed service, no consulting layer. (Note: the package is installable but archived/unmaintained, and pins to older Python — 3.8–3.10.)

**Target users.** Data scientists, quantitative marketing analysts, in-house measurement teams, and agencies with Python/Bayesian capability who want a free, transparent, code-first MMM. Not aimed at non-technical marketers (no GUI, no hosted product). For new projects Google now directs these users to Meridian.

**Maturity.** First public release in early 2022 (v0.1.x); final release v0.1.9 on 23 May 2023 (PyPI). PyPI development status '3 - Alpha'; supports Python 3.8–3.10; never reached a stable 1.0. Achieved broad community adoption as one of the most-used free Bayesian MMM libraries (adoption is described qualitatively — no exact star/download count independently verified). Now legacy/end-of-life: Google released Meridian on 29 Jan 2025 as the official successor, declared LightweightMMM unsupported, and the GitHub repository was archived (read-only) on 19 Jan 2026 (CONFIRMED on the GitHub repo page).

**Sources:** <https://github.com/google/lightweight_mmm> · <https://pypi.org/project/lightweight-mmm/> · <https://lightweight-mmm.readthedocs.io/en/latest/models.html> · <https://raw.githubusercontent.com/google/lightweight_mmm/main/lightweight_mmm/models.py> · <https://getrecast.com/google-lightweightmmm/> · <https://developers.google.com/meridian/docs/migrate> · <https://github.com/google/meridian> · <https://github.com/google/lightweight_mmm/releases>

---

### GPS-Enterprise (GPS-E)

**Type:** Enterprise/consulting — commercial SaaS platform (GPS-Enterprise) delivered largely as a managed service / hybrid engagement (platform + analysts), not self-serve &nbsp;|&nbsp; **Verification confidence:** medium

**Vendor:** Analytic Partners (founded 2000 by Nancy Smith, who is President & CEO; registered HQ in Miami, FL with a major New York office; privately held, independent, Certified Women's Business Enterprise; Chief Science Officer Hong Jin)

**Methodology.** Econometric/commercial-mix modeling at the core, positioned as "Commercial Analytics / Commercial Mix Modeling" that extends MMM beyond media to pricing, promotion, distribution, supply chain, COGS, channel margins and macro factors (incl. weather). The exact estimation engine is NOT disclosed in Analytic Partners' own primary materials (its solution pages are marketing-only and mention no sampler, adstock/saturation forms, or credible-interval output). VERIFIED CORRECTION vs the researcher's framing: third-party/secondary descriptions (aggregator blogs) characterize AP's toolkit as a proprietary blend of time-series econometrics, Bayesian regression, and machine learning that uses ROI Genome benchmarks as informative priors for channels lacking history — i.e., AP is plausibly Bayesian-inclusive itself. That reframes AP's well-documented public stance: it argues specifically AGAINST the "calibrate a Bayesian MMM by forcing experiment/lift results in as priors" workflow (its blog: basing an MMM on experiment results "inverts the logical order of analytics"; base metrics should come from comprehensive econometric modeling, with experiments used to VALIDATE, not force, the base case), objecting to walled-garden bias and siloed/unstable experiment inputs — this is a critique of a workflow, not necessarily a rejection of Bayesian estimation. Adstock/carryover and diminishing-returns/saturation are implied (standard for any credible MMM and consistent with its halo/synergy claims) but specific functional forms are unpublished.

**Strengths:**

- Deep enterprise pedigree: 25+ years (founded 2000), Fortune 500 / large multi-brand, multi-region global client base
- Named a Leader in the Gartner Magic Quadrant for MMM in BOTH the inaugural 2024 (announced Nov 22, 2024; highest scores across all 8 use cases in the companion Critical Capabilities report) and 2025 (highest in Ability to Execute and furthest in Completeness of Vision; highest scores in 8 of 9 critical capabilities; top-ranked Enterprise Mix Modeling 4.25/5 and Mix Modeling for Branding 4.25/5)
- Forrester Wave Leader — recognized in Marketing Measurement & Optimization, Q3 2023, and AGAIN (newer) in The Forrester Wave: Marketing Measurement and Optimization Services, Q1 2026 (highest possible 5/5 scores across 21 of 31 criteria)
- Broadest commercial scope — extends beyond media to pricing/trade/distribution/margin/macro drivers, aligning marketing measurement with CFO/finance and total-business decisions
- High-touch analyst guidance plus executive-ready strategic storytelling/consulting layered on the platform
- ROI Genome normative benchmarks provide context/priors most standalone tools and open-source libraries lack
- Handles messy, large-scale, multi-country/multi-brand data and integrates many sources
- Forward-looking decisioning/optimization and scenario planning oriented to planning cycles, not just backward reporting

**Limitations:**

- Methodological transparency is low in PRIMARY materials: AP's own pages disclose no statistical engine, no adstock/saturation forms, no geo/hierarchical structure, reach & frequency handling, time-varying-effects handling, and no explicit statement of whether it reports full posteriors/credible intervals vs point estimates — hard to independently audit (third-party descriptions suggest a Bayesian/econometric/ML blend, but AP does not confirm this publicly)
- Consulting-heavy and not self-serve; independent reviews cite heavy analyst dependency and longer lead/refresh cycles (implementation timelines commonly cited by third parties at roughly 10-14+ weeks — vendor-uncorroborated)
- Output tends to sit at the channel/strategic level — better for annual/strategic budget planning than campaign-level, tight-cycle, day-to-day optimization
- Premium, opaque, quote-based pricing with no free/open-source tier; enterprise cost puts it out of reach for SMB/mid-market and far above zero-license open-source (Robyn/Meridian/PyMC-Marketing)
- Less transparency/customization control than open-source libraries; you depend on the vendor's proprietary method and the specific data-science team assigned
- Uncertainty-quantification and calibration mechanics are not documented publicly, so genuine incrementality/credible-interval rigor cannot be independently verified

**Experiment calibration.** AP emphasizes building a "comprehensive econometric" base model FIRST and using experiments to VALIDATE it, rather than calibrating a Bayesian MMM by injecting lift-test results as priors — a workflow it explicitly critiques (blog: 'Experimentation + Bayesian MMM: The Weird Science of Marketing Measurement'), citing walled-garden bias and unstable siloed inputs. ROI Genome (cross-client/industry normative benchmarks accumulated over ~25 years) functions as an informative-benchmark/prior layer to contextualize and sanity-check results, and third-party descriptions say these benchmarks serve as priors for channels lacking history. Public materials do not detail a formal, reproducible mechanism for folding specific incrementality/geo-lift experiments into channel coefficients (unlike the documented experiment-prior workflows in Google Meridian / PyMC-Marketing).

**Pricing.** Not publicly disclosed; premium/enterprise, quote-based, service-heavy, with no open-source or free tier. Third-party estimates vary widely and could NOT be independently corroborated in this check (figures cited range from low-six-figures per engagement to $1M+ per year for large managed programs). Treat all pricing figures as unverified external estimates.

**Deployment.** Web-based SaaS platform (GPS-Enterprise), listed on cloud marketplaces (e.g., SoftwareOne; Azure cited but not independently confirmed here), but in practice delivered as a managed-service / hybrid engagement blending platform outputs with Analytic Partners analysts and consultants. Not a self-serve, code-it-yourself library. Direct integrations to major data/media partners (Meta, Google, Amazon, Roku) and cloud providers.

**Target users.** Large enterprises and Fortune 500 brands with complex multi-region/multi-brand portfolios and large marketing/commercial budgets; buyers are senior executives (CMO, CFO, CDO) seeking strategic planning and total-commercial optimization rather than in-house data-science teams wanting a build-it-yourself tool. Industries include retail, CPG, financial services, hospitality/restaurant, automotive, telecom, pharma and entertainment. (The '$100M+ spend' threshold in the source profile is a reasonable characterization but not a vendor-published cutoff — treat as indicative.)

**Maturity.** Mature market leader. Founded 2000; marked ~25 years in 2025 (April 2025 'Shapes MMM for 25 years / Commercial Analytics' milestone). GPS-Enterprise is the flagship platform. Corrected release chronology: the 'Drives Speed to Insight' release introducing Dynamic Decisioning, Live Modeling and next-gen ADAPTA shipped March 18, 2021 (the source profile's 'March 18, 2025' is an error); GPS-Enterprise 10 shipped Feb 16, 2022. Gartner MQ Leader in the inaugural 2024 and in 2025; Forrester Wave Leader Q3 2023 and again Q1 2026. Private, independent company; third-party aggregators estimate roughly 450-530 employees (one source ~447; LinkedIn band 501-1,000) and revenue in the ~$58M-$94M range (one 2026 aggregator figure ~$93.9M) — all unverified and source-dependent. Global offices (US incl. Miami HQ and New York, plus Dublin, Sydney, etc.). Proprietary vendor — no public community/GitHub/open-source presence.

**Sources:** <https://analyticpartners.com/platform/> · <https://analyticpartners.com/solutions/marketing-mix-modeling/> · <https://analyticpartners.com/knowledge-hub/blog/experimentation-bayesian-mmm-the-weird-science-of-marketing-measurement/> · <https://analyticpartners.com/knowledge-hub/newsroom/new-release-gps-enterprise-drives-speed-to-insight/> · <https://www.businesswire.com/news/home/20210318005260/en/Analytic-Partners-Drives-Speed-to-Insight-with-New-Release-of-GPS-Enterprise> · <https://analyticpartners.com/knowledge-hub/newsroom/gps-enterprise-10-product-announcement/> · <https://analyticpartners.com/knowledge-hub/newsroom/analytic-partners-recognized-as-a-leader-in-inaugural-gartner-magic-quadrant-for-marketing-mix-modeling-solutions/> · <https://www.prnewswire.com/news-releases/analytic-partners-positioned-as-a-leader-in-the-2025-gartner-magic-quadrant-for-marketing-mix-modeling-solutions-302619022.html> · <https://analyticpartners.com/knowledge-hub/resources/forrester-wave-mmo-2023/> · <https://analyticpartners.com/knowledge-hub/resources/forrester-wave-2026/> · <https://analyticpartners.com/knowledge-hub/newsroom/roku-measurement-program-partnership/> · <https://analyticpartners.com/about/who-we-are/> · <https://getlatka.com/companies/analyticpartners.com> · <https://www.crunchbase.com/organization/analytic-partners> · <https://improvado.io/blog/analytic-partners-competitors>

---

### Traditional / Enterprise & Decision-Science MMM Vendors (Nielsen→Circana, Kantar LIFT ROI, Ipsos MMA, Keen Decision Systems, NCSolutions)

**Type:** Enterprise/consulting is the dominant category posture (managed engagements plus hosted analyst-run platforms), but the group is mixed: legacy consulting divested (Nielsen MMM → Circana); enterprise consulting + always-on SaaS overlays (Ipsos MMA "Activate"; Kantar "LIFT ROI"/HamiltonAI); decision-science Commercial SaaS (Keen Decision Systems); and purchase-based causal measurement rather than classic MMM (NCSolutions/Circana). Best labeled "Enterprise/consulting" for the category as a whole. &nbsp;|&nbsp; **Verification confidence:** medium

**Vendor:** Category profile spanning: Nielsen (has exited MMM — divested its MMM business to Circana), Circana (now holds Nielsen's former MMM business AND NCSolutions), Kantar (Analytics Practice / LIFT ROI, powered by the HamiltonAI platform from its 2022 Blackwood Seven acquisition; Kantar is Bain Capital-majority-owned since Dec 2019), Ipsos MMA (Marketing Management Analytics, founded 1989, acquired by Ipsos in 2018 from Hunting Ridge LLC), Keen Decision Systems (independent Bayesian SaaS, Research Triangle Park NC, founded 2010), and NCSolutions (formerly Nielsen Catalina Solutions, a Nielsen+Catalina JV — now Circana-owned).

**Methodology.** Category historically rooted in econometric/regression MMM: multiplicative (log-log) or additive OLS/GLS and ridge regressions on ~2-3 years of aggregate weekly sales vs. media spend and non-marketing drivers (price, distribution, promotion, seasonality, competition, macro), with adstock/carryover transforms and saturation/diminishing-returns curves. Outputs are channel contributions, ROI/ROAS and response curves for scenario planning and budget optimization. The frontier has moved toward Bayesian methods, unevenly by vendor and — importantly — the specific technical descriptions are marketing-page level, not peer-reviewed: (1) Ipsos MMA markets a semi-automated engine combining "Bayesian statistics, machine learning and business guardrails," unifying MMM + attribution/MTA + in-market testing (confirmed on mma.com marketing pages). (2) Keen is explicitly Bayesian/"adaptive," updating priors of tactic elasticity as new data arrives with Monte-Carlo simulation (confirmed on keends.com). (3) Kantar's HamiltonAI (from the 2022 Blackwood Seven acquisition) is documented as Bayesian — priors + likelihood → posterior, with "hierarchical modeling central to the HamiltonAI approach" and "synergy models" for cross-touchpoint effects. CORRECTION: the profile's more specific phrasing — "Bayesian hierarchical probabilistic directed continuous networks" and detecting causality via "ceteris-paribus perturbation" — could NOT be confirmed on the cited Blackwood Seven primary source and should be treated as unverified. (4) NCSolutions is methodologically DIFFERENT — purchase-based sales-lift/incrementality: household in-store/retail purchases (Catalina loyalty/retailer data) integrated with cross-media exposure data across ~90M+ US households, using ML household "proximities" to categorize households and its "Sales Effect" service to estimate incremental sales vs. a no-exposure counterfactual. It produces experimental-style ground truth rather than a mix model. CORRECTION: the profile's claim that NCS matches purchases to exposure "via Experian" is not substantiated in NCS's own materials (Experian is a general retail-media identity pattern, not documented as NCS's mechanism). Sampler/engine internals, exact adstock/saturation functional forms, and prior specifications are generally NOT publicly documented across the enterprise vendors (proprietary vs. open-source Meridian/PyMC-Marketing/Robyn).

**Strengths:**

- Deep methodological pedigree and enterprise trust — MMA (1989), Ipsos, Kantar and Nielsen/Circana are long-established; Ipsos MMA is a Leader (2nd consecutive year) and Kantar a Visionary in the 2025 Gartner Magic Quadrant for MMM
- Handles messy, incomplete enterprise data and complex categories with expert human oversight and business guardrails
- Broad scope beyond media (pricing, promotion, distribution, macro, creative, brand) that pure media-focused challengers often miss
- Strong calibration assets: Ipsos MMA's proprietary benchmark databases, MMM+MTA+experiment triangulation, and NCSolutions' single-source household purchase data as causal ground truth
- Genuine Bayesian/uncertainty capability on the modernized platforms (Keen explicitly; Ipsos MMA and Kantar HamiltonAI per marketing materials), not just point estimates
- Newer offerings add always-on refresh, scenario/optimization UIs and (Keen) financial/P&L forecasting, narrowing the cadence gap with SaaS challengers
- Managed delivery removes the need for an in-house data-science team (explicit Keen and Ipsos positioning); accountability sits with the vendor

**Limitations:**

- Cost and cadence of the classic model: third-party/industry write-ups put full-service enterprise MMM engagements in the low-to-mid six figures (~$200K-$500K per project) with multi-month delivery and quarterly refresh — mismatched with weekly/daily budget decisions (SaaS overlays only partly close this). These are ballpark estimates, not vendor quotes, and the real market spans a very wide range
- Methodological opacity: samplers, priors, adstock/saturation forms and validation are proprietary and largely undocumented publicly; hard to independently verify the claimed Bayesian rigor or reproduce results (contrast open-source Meridian/PyMC-Marketing/Robyn). Some specific technical claims in circulation (e.g., HamiltonAI's exact model class) could not be confirmed against primary sources
- Uncertainty quantification is inconsistent across the category — Bayesian platforms can yield credible intervals, but classic econometric deliverables typically report point estimates + significance, and marketing materials rarely surface calibrated intervals
- Vendor-reported accuracy claims (e.g., Kantar LIFT ROI's '96.97% sales-prediction' figure) are single client testimonials (Telenor), self-reported and not independently validated; the basis (in-sample fit vs. out-of-time predictive accuracy) is not disclosed
- Vendor lock-in and limited transparency/portability; aggregate MMM cannot deliver user-level granularity and is data-hungry (~2-3 years history)
- Consolidation / continuity risk: Nielsen has exited MMM entirely — selling its MMM business AND its NCSolutions stake to Circana (announced Aug 2024; NCS close June 2025; Nielsen MMM-business close Aug 21, 2025) — so 'Nielsen MMM' is a legacy/transferred product line now under Circana
- Category-framing caveats: NCSolutions is purchase-based sales-lift/incrementality measurement, NOT a classic mix model; and 'Cassandra' is an unrelated modern self-serve SaaS startup, not an Ipsos product (see corrections)

**Experiment calibration.** Enterprise vendors triangulate MMM with attribution + in-market experiments + benchmark priors (Ipsos MMA benchmark databases; Kantar LIFT ROI's unified measurement); NCSolutions provides purchase-based sales-lift/incrementality as causal ground truth; Keen updates tactic-elasticity priors adaptively as new data arrives. Precise experiment-to-prior mechanics (how a lift result becomes a prior/likelihood) are proprietary and unverified from public sources.

**Pricing.** Predominantly high-cost, non-public, negotiated for the enterprise players. Third-party/industry write-ups estimate full-service enterprise MMM engagements at roughly $200K-$500K per project and six-figures-annually for ongoing programs, but the broader MMM market spans a very wide range (from low-thousands/month SaaS to $1M-$2.7M multi-year enterprise TCO) — treat any point figure as indicative, not a quote. Exact vendor pricing for Circana/Kantar/Ipsos MMA is not publicly disclosed. Keen Decision Systems is the transparent SaaS exception (subscription with 'transparent annual pricing'; specific tiers not public); Keen has raised roughly $17.7M-$18M+ total (sources vary). NCSolutions/Circana prices as a measurement/data service, not per model.

**Deployment.** Historically consulting engagements (analyst-built custom models, quarterly deliverables and readouts). The category has layered hosted SaaS platforms on top for self-serve scenario/optimization and always-on refresh: Ipsos MMA 'Activate' (web-based unified-measurement platform), Kantar 'LIFT ROI'/HamiltonAI (up to daily sales-KPI granularity, publisher-level ROI), and Keen's cloud subscription. None are self-host open-source libraries; delivery remains vendor-hosted and largely managed. NCSolutions/Circana is a hosted measurement/data-as-a-service (including data clean-room integrations, e.g. NCS 'Insights Stream').

**Target users.** Large enterprise advertisers — especially CPG, retail, financial services, pharma and telecom (Fortune 500) — plus their agencies and finance teams. Buyers are typically CMOs, marketing-effectiveness/insights leaders and finance stakeholders who want managed, defensible measurement without building in-house data science. Keen additionally targets mid-market and 'no data scientist required' buyers; NCSolutions specifically serves CPG brands, retailers and media/platform partners for advertising effectiveness.

**Maturity.** Highly mature, consolidating category. Ipsos MMA traces to Marketing Management Analytics (founded 1989, among the first to commercialize MMM; acquired by Ipsos in 2018) — a Leader in both the 2024 and 2025 Gartner MMM Magic Quadrants. Kantar (Bain Capital majority owner since Dec 2019) acquired Blackwood Seven in 2022 for the HamiltonAI Bayesian platform; a Visionary in the 2025 Gartner MMM MQ. Keen Decision Systems founded 2010 (CEO/co-founder Greg Dolan; Chief Decision Science Officer John Busbice), an AI/Bayesian SaaS, multi-year Inc. 5000 lister (~43-51 employees by source); evaluated in the 2025 Gartner MMM MQ (not a Leader). Nielsen has exited MMM, divesting its MMM business and its NCSolutions stake to Circana (announced Aug 2024; NCS close June 2025; MMM-business close Aug 21, 2025). The 2025 Gartner MMM MQ (published Nov 10, 2025) Leaders were Analytic Partners, Ipsos MMA and TransUnion; Ekimetrics and Kantar were among Visionaries; C5i, Circana, Fractal, Keen and OptiMine were also evaluated. These are established, well-funded, backed by major research/consulting parents — the opposite of a nascent open-source community.

**Sources:** <https://www.nielsen.com/news-center/2024/circana-to-acquire-ncsolutions-and-nielsens-marketing-mix-modeling-business/> · <https://www.circana.com/post/circana-to-acquire-ncsolutions-and-nielsen-s-marketing-mix-modeling-business> · <https://finance.yahoo.com/news/circana-completes-acquisition-nielsen-marketing-130000590.html> · <https://mma.com/about/our-history/> · <https://mma.com/solutions/unified-marketing-measurement/> · <https://mma.com/resources/thought-leadership/ipsos-mma-named-a-leader-2025-gartner-magic-quadrant-for-marketing-mix-modeling-solutions/> · <https://www.kantar.com/campaigns/lift-roi> · <https://blackwoodseven.com/the-next-generation-of-marketing-mix-modeling/> · <https://blackwoodseven.com/hamilton-ai-brings-agility-to-marketing-investments/> · <https://www.kantar.com/company-news/kantar-to-acquire-blackwood-seven> · <https://keends.com/> · <https://keends.com/about/> · <https://keends.com/blog/the-benefits-of-bayesian-marketing-mix-modeling/> · <https://www.crunchbase.com/organization/keen-decision-systems> · <https://www.gartner.com/en/documents/7160530> · <https://analyticpartners.com/knowledge-hub/newsroom/analytic-partners-leader-in-2025-gartner-magic-quadrant-for-marketing-mix-modeling/> · <https://newsroom.transunion.com/transunion-named-a-leader-in-the-2025-gartner-magic-quadrant-for-marketing-mix-modeling-solutions/> · <https://www.ekimetrics.com/articles/ekimetrics-recognized-in-the-2025-gartner-magic-quadrant-for-marketing-mix-modeling> · <https://www.prnewswire.com/news-releases/advertising-outcome-measurement-gets-a-lift-with-the-launch-of-the-next-generation-sales-effect-from-ncsolutions-301561349.html> · <https://www.businesswire.com/news/home/20190130005701/en/Nielsen-Catalina-Solutions-NCS-Sales-Effect-Measurement> · <https://cassandra.app/resources/cassandra-2m-seed-round> · <https://www.pitchdrive.com/stories/cassandra-raises-eu2-million-to-disrupt-marketing-investment-decisions-with-ai-powered-modeling>

---

## Choose by use case

| If you are… | Best fit | Why |
| --- | --- | --- |
| In-house data-science team wanting full control, causal rigor, and a broad experiment/uncertainty toolkit — and comfortable self-hosting | **This app (or PyMC-Marketing)** | This app uniquely bundles a causal-DAG builder, EIG/EVOI experiment prioritization, off-panel calibration, a continuous-learning bandit, and an agent interface in one open framework; PyMC-Marketing is the more mature, better-supported fallback if bus-factor and community matter more than the workflow breadth. |
| Enterprise needing SLAs, managed delivery, executive storytelling, and total-commercial (price/promo/distribution) scope | **Analytic Partners (or the Enterprise MMM camp: Ipsos MMA / Kantar)** | Gartner/Forrester-recognized pedigree, managed analyst teams, ROI Genome benchmarks, and P&L-level scope that a single-author open framework cannot match on trust or support. |
| Mid-market/enterprise brand wanting an always-on, fully managed measurement partner with time-varying coefficients and weekly refresh | **Recast (or Mutinex GrowthOS)** | Both deliver hosted, SOC 2-style managed SaaS with genuine Bayesian uncertainty and continuous refresh; Recast adds rigorous prospective out-of-sample scoring, Mutinex adds sub-channel Campaign-Varying granularity and faster onboarding. |
| Startup or analyst wanting a free, well-supported open-source Bayesian MMM to stand up quickly | **Google Meridian (or PyMC-Marketing)** | Meridian is free, Google-backed, actively maintained, offers native reach & frequency and a no-code Scenario Planner; PyMC-Marketing if maximum modeling flexibility is wanted. Both carry far less maintenance risk than a v0.2.0 single-author project. |
| Data-rich digital/direct-response advertiser wanting automated open-source tuning with heavy calibration to lift tests | **Meta Robyn** | Free, mature, automates hyperparameter search and folds lift tests in as a calibration objective; accept that it is frequentist point estimates, not posteriors, and requires manual Pareto-model selection. |
| Team whose core need is rigorous, experiment-driven, information-theoretic measurement (what to test next, when to stop, how to calibrate) | **This app** | Its EIG/EVOI prioritization, Pareto experiment optimizer, in-graph + off-panel calibration, SBC, and model-free learning bandit form an experiment-design-and-value-of-information stack that no competitor here matches — provided the team can operate a research-grade self-hosted framework. |
| Non-analyst marketer wanting a guided, point-and-click tool with a hosted UI and support | **Mutinex GrowthOS or Recast** | Fully managed platforms with conversational/assistant layers (MAITE) and no in-house data-science requirement; this app's Oracle agent helps but still assumes self-hosted, technical operation. |

## Bottom line

This app is the most feature-broad and methodologically ambitious open MMM framework in this set: it stands shoulder-to-shoulder with Meridian and PyMC-Marketing on Bayesian rigor and then adds a workflow layer — an LLM agent, an EIG/EVOI measurement loop, a model-free learning bandit, declarative estimands, a Model Garden with non-MMM families, and off-panel calibration — that no competitor, open or commercial, matches in one package. But it is early-stage and single-author (v0.2.0), with internal-only benchmarks, stubbed ad-platform connectors, no managed SaaS, and none of the brand trust, support SLAs, or proven at-scale adoption that Recast, Mutinex, Analytic Partners, or the Google/Meta-backed open tools bring. For a technical team that values causal rigor, experiment-driven measurement, and full control — and can self-host a research-grade framework — it is genuinely differentiated and often ahead. For buyers who need turnkey hosting, vendor accountability, rich data connectors, or independent validation, an established commercial or better-supported open-source option remains the safer choice today.
