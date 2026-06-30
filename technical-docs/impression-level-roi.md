# Impression-/click-measured media: ROI, mROI, and efficiency

## Problem

ROI and marginal ROAS are formed as `incremental KPI / spend`. The framework
historically assumed the **modeled media variable is the spend**, so it computed
the denominator by summing that variable (`X_media_raw[:, channel]`). That breaks
when a channel is measured in **impressions** or **clicks**: summing impressions
is not a dollar amount, so dividing by it does not give ROI.

Three ways to handle it (all supported, one unified mechanism):

- **(a) a separate spend variable** — the channel carries a second MFF column with
  actual dollars.
- **(b) a CPM / CPC** — derive a spend series from the volume:
  `spend = (impressions / 1000) · CPM` or `spend = clicks · CPC`.
- **(c) no cost at all** — report **efficiency** instead of ROI: incremental KPI
  per 1,000 impressions (or per click), and the marginal efficiency of an extra
  1,000 impressions (or click). Its break-even reference is **0**, not 1.0.

(b) is really a way to derive the (a) spend series; (c) is a relabeling + a
different normalizer used only when neither (a) nor (b) is available.

## The measurement descriptor (config)

`MediaChannelConfig` (`config/variables.py`) gains four additive fields; the
default keeps every existing spend model **byte-identical**:

```python
measurement_unit: MeasurementUnit = SPEND   # SPEND | IMPRESSIONS | CLICKS | OTHER
spend_column: str | None = None             # (a) name of a separate $ MFF variable
cpm: float | None = None                    # (b) cost per 1,000 modeled units
cpc: float | None = None                    # (b) cost per click (requires CLICKS)
```

A validator enforces: at most one of `{spend_column, cpm, cpc}`; a cost implies a
non-spend unit; `cpc` requires `clicks`; positive costs. Builder fluent setters:
`.measured_in(...)`, `.with_spend_column(...)`, `.with_cpm(...)`, `.with_cpc(...)`.
Agent spec keys (`agents/fitting._mff_config_from_spec`): `measurement_unit`,
`spend_column`, `cpm`, `cpc` on each `media_channels[*]`.

**The response curve is always fit on the modeled variable** (impressions). The
descriptor only changes the ROI/efficiency **denominator** and **labels** — spend
never enters the PyMC graph.

## The resolver (the single math hub)

`reporting/helpers/measurement.py::resolve_channel_divisor(model, channel, mask)`
returns `ChannelDivisor(total, found, meta)`. `total` is the **spend-equivalent or
volume-equivalent of the (masked) window** — one number that serves both metrics:

- average ROI / efficiency = `contribution / total`
- marginal denominator     = `total · (factor − 1)` for a `ScaleInput(factor)` bump

Precedence: `spend` default → external `spend_column` → `cpm`/`cpc` → efficiency.
The default (`SPEND`, no overrides) reproduces the legacy `X_media_raw` column sum
exactly (same panel→raw→X_media precedence as the old `_extract_spend_from_model`).

`MetricMeta` carries `is_monetary`, `cost_basis`, `roi_label`, `marginal_label`,
`value_units`, `divisor_units`, `reference` (1.0 ROI / 0.0 efficiency), and
`supports_profitability` (False for efficiency — `prob_profitable` is dropped).

A declared `spend_column` is loaded by `MFFLoader.build_panel` into
`PanelDataset.spend_raw` (`{channel: per-obs $ array}`) → `model.spend_raw`; the
resolver reads it for branch (a). Missing ⇒ warn + degrade to efficiency.

## Wired sites (all measurement-aware)

| Site | File |
|---|---|
| Dashboard ROI + `_extract_spend_from_model` | `reporting/helpers/roi.py` |
| Counterfactual ROI | `analysis.py::compute_channel_roi` |
| Marginal ROAS | `model/base.py::compute_marginal_contributions` |
| Declarative estimands (`_observed_spend`/`_marginal_spend`) | `estimands/evaluate.py` |
| Report ROI section + forest plot reference line | `reporting/sections.py`, `charts/roi.py` |
| Report extractors (`channel_roi`, `blended_roi`) | `reporting/extractors/{bayesian,extended}.py` |
| Estimand report rows + Performance API | `extractors/mixins.py`, `agents/estimand_rows.py`, `api/estimands.py` |
| Agent ops (`roi_metrics`, `compute_estimands`, `marginal_analysis`) | `agents/model_ops.py` |

**Mixed portfolios** (some spend, some efficiency channels): per-channel numbers
only — the blended "Marketing ROI" headline is suppressed (`_compute_blended_roi`
returns `None`) because mixing $/$ with KPI/1k is not comparable. cpm/cpc channels
ARE dollars, so they blend normally with spend channels.

## Deferred

- **In-graph experiment-calibration ROI bridge** (`model/base.py` ≈1919-1931,
  `calibration/likelihood.py`/`estimands/graph.py`) still divides by
  `X_media_raw`. The experiment readout supplies its own (dollar) spend, so this
  is a narrow path; calibrating an impression channel against a dollar-ROAS
  experiment would currently mismatch the denominator.
- **Time-varying CPM/CPC column** (only scalar `cpm`/`cpc` today).
- **`spend_column` on the RaggedMFFLoader path** (standard `MFFLoader` only;
  ragged + spend_column degrades to efficiency).
- **Dedicated Performance-dashboard React panels** beyond the server-driven
  `units`/`evidence`/`reference` they already render (the agent-op Results tables
  are already measurement-aware via generic `TableCard`).

## Tests

`tests/test_measurement_metrics.py` — validator, every resolver branch,
mask/marginal identity, loader `spend_column` integration, agent spec threading.
The estimands equivalence gate (`tests/test_estimands.py`) confirms byte-stability
for spend models.
