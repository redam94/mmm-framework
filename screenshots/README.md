# Augur platform — screenshots

Clean, retina (2×) captures of the running app (React UI + agent API), taken
2026-07-23 against the seeded demo projects. Companion to the 90-second demo
video at `recordings/augur_demo_90s.mp4`.

Regenerate with `recordings/clean_shots.py` (needs the app running — see the
`run-app` skill); record the video with `recordings/demo90_driver.py`.

| # | File | Surface | What it shows |
|---|------|---------|---------------|
| 01 | `01-oracle-agent-chat.png` | Oracle (Workspace) | Chat-aided modeling — the agent's model comparison + honest recommendation, with the 9-step scientific workflow |
| 02 | `02-oracle-model-spec.png` | Oracle · Model | The assembled model spec — KPI, inference (NUTS), trend, seasonality, channels |
| 03 | `03-oracle-results-roi.png` | Oracle · Results | Contribution decomposition + channel ROI with 94% HDI and P(profitable) |
| 04 | `04-chronicle-runs-lineage.png` | Chronicle · Runs | **Inspect previous models** — lineage of every fit: data fingerprint, spec diffs, model health, parent run |
| 05 | `05-chronicle-trajectories.png` | Chronicle · Trajectories | Cycle-over-cycle CI contraction + budget-share migration |
| 06 | `06-chronicle-estimands.png` | Chronicle · Estimands | ROI by channel across models, coded by evidence tier |
| 07 | `07-chronicle-saturation-roas.png` | Chronicle · Saturation & ROAS | Average vs marginal ROAS + response curves, with curve-position flags |
| 08 | `08-orrery-measurement-cycle.png` | Orrery (Program) | The T₀–T₅ measurement loop + program KPIs |
| 09 | `09-auspices-experiment-priority.png` | Auspices (Experiments) | EIG × EVOI priority matrix — what to test next, ranked by value of information |
| 10 | `10-almanac-budget-planner.png` | Almanac (Planner) | Budget allocation + forward flighting off the calibrated posterior |
| 11 | `11-constellation-portfolio.png` | Constellation (Portfolio) | Every brand benchmarked — channel ROI P25–P75, model freshness, calibration coverage |
| 12 | `12-atelier-model-garden.png` | Atelier (Model Garden) | Versioned, governed registry of bespoke models the agent can run |

The three requested flows map to: **talk with the agent** → 01; **build a
model** → 02–03; **inspect previous models** → 04 (plus 05–07).
