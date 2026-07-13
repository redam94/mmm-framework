# Reach & frequency modeling — #141

For brands with reach/frequency data (TV, YouTube, programmatic), the lever is not
raw impressions but *how many distinct people were reached and how often*. Two
plans with equal impressions but different frequency distributions have very
different effects (the 3+ frequency wearout). A volume MMM — where the channel is
one impressions number — cannot express this, so it cannot answer the core
planning question **"buy more reach or more frequency?"**.

## Model

A channel declared reach/frequency-measured has its column treated as **reach**
(distinct people reached per period) and its effect modulated by a
**frequency-saturation** curve:

```
effective_reach(t) = reach(t) · g(frequency(t))
contribution(t)    = beta · sat(adstock(effective_reach(t)))
```

`g` maps the mean-normalized average frequency to a per-period factor in `(0, 1]`
— diminishing returns to added exposures. The effective-reach signal then flows
through the channel's **normal** adstock → saturation → `beta` pipeline, so
`beta_<channel>` and `channel_contributions` keep their meaning (R0.4) and every
downstream number (ROI, decomposition, counterfactuals) is unchanged in shape.

Two shapes (`FrequencyResponse`):

- **`exponential`** (default) — `g(f) = 1 - exp(-k f)`: monotone, diminishing from
  the first exposure, asymptotes to 1. The safe planner default.
- **`hill`** — `g(f) = f^s / (f^s + h^s)`: S-shaped, models a *minimum effective
  frequency* threshold (low frequency nearly wasted, then it kicks in).

## Usage

```python
from mmm_framework.config import ReachFrequencyConfig, FrequencyResponse

model_config = (
    ModelConfigBuilder()
    .with_reach_frequency(
        ReachFrequencyConfig(
            channel="TV",              # its media column is REACH
            frequency_column="Frequency",  # avg frequency, a control column
            response=FrequencyResponse.EXPONENTIAL,
        )
    )
    .build()
)
```

Off by default (`reach_frequency=[]`): a channel is an ordinary volume channel and
any frequency column stays a plain linear control — byte-identical to today
(R0.1/R0.2).

### Data requirements

- The channel's **media column is reach** — a distinct-audience measure (a count
  or a 0–1 fraction), *not* impressions or spend.
- The **frequency column is a control column** (average exposures per period). It
  is pulled out of the linear control block (not double-counted) and used only to
  build the frequency gain. If you only have impressions + reach, derive
  `frequency = impressions / reach` and supply it as the control.
- Because the channel column is reach (a volume, not dollars), set the channel's
  `measurement_unit` to `impressions` or provide a `spend_column` so ROI /
  efficiency stays coherent (a reach channel has an efficiency basis, not a
  divide-by-reach ROI).

## Reporting

The fit registers `freq_gain_<channel>` (the per-period gain series),
`effective_frequency_<channel>` (the raw-unit frequency at 90% of the asymptote —
exponential — or the half-saturation frequency — Hill), and the shape RV
(`freq_k_<channel>`, or `freq_halfsat_/freq_slope_<channel>`). The Channel ROI
section surfaces a **reach-vs-frequency insight**: *"effectiveness plateaus around
N average exposures — beyond this, spend the next dollar on reach, not
frequency."*

## Identification & assumptions

- The frequency curve is identified by **frequency variation that is not collinear
  with reach**. If reach and frequency always move together, `k`/`h` and `beta`
  trade off and the curve is weakly identified — vary a media plan's frequency
  independently (or add a frequency experiment) to pin it.
- The curve is a *shape on the modeled reach series*, not a causal claim about
  incremental frequency; treat the effective-frequency number as directional, and
  confirm a reach-vs-frequency reallocation with an experiment.
- MAP fits recover the shape's direction but bias `k` low (as with other
  saturation parameters); re-fit with NUTS before trusting the effective-frequency
  interval.

## Ground truth & tests

`synth/dgp.py::make_reach_frequency` plants TV's effect as `reach ·
(1 - exp(-k·freq))` with a known `k` and records `true_freq_k` /
`true_effective_frequency` on the scenario notes. `tests/test_reach_frequency.py`
covers off-is-linear-control, on-pulls-frequency + adds the gain, Hill shape RVs,
misconfiguration warnings, gain bounds + monotonicity, frequency-saturation
recovery, and that the frequency model fits at least as well as ignoring frequency.
