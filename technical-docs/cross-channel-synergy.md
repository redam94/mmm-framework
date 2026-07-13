# Cross-channel synergy / interaction — #142

The base model is strictly additive across channels, so it structurally cannot
express that two channels together do **more** (synergy / halo — TV priming
Search) or **less** (cannibalization) than the sum of their parts. This adds an
opt-in interaction term per named channel pair.

## Usage

```python
from mmm_framework.config import ChannelInteraction

model_config = (
    ModelConfigBuilder()
    .with_channel_interactions(
        ChannelInteraction(channel_a="TV", channel_b="Search",
                           expected_sign="positive", prior_sigma=0.3),
    )
    .build()
)
```

For each configured pair the mean gains

```
beta_ij · sat_i(adstock(x_i)) · sat_j(adstock(x_j))
```

— the product of the two channels' **saturated responses** (each in `[0, 1]`),
so the term is well-behaved and shares the media block's adstock/saturation RVs.
`beta_int_<a>_<b>` is emitted as the effective (correctly-signed) coefficient.

- **`expected_sign`** encodes the belief: `"positive"` (synergy) ⇒ HalfNormal,
  so the effect can only lift; `"negative"` (cannibalization) ⇒ its reflection,
  `≤ 0`; `"any"` ⇒ Normal(0, σ).
- **`prior_sigma`** (default 0.3) shrinks the interaction toward zero.

## Off by default / regression

`channel_interactions` defaults to `[]` and the interaction block is added to
`mu` only when non-empty, so the default graph is byte-identical to the additive
model (R0.1/R0.2).

## Reporting

`interaction_contributions` (obs × pair) and `interaction_component` (total) are
registered deterministics; the decomposition carries `interactions` /
`total_interactions`, so the waterfall still closes and synergy appears as its
own line ("Synergy / Interactions" in the summary, `"Synergy"` in the report's
component totals) rather than being credited to the individual channels.

## Identifiability caveat

**Interactions are weakly identified without designed variation.** With
observational data the two channels usually move together, so the product term
`sat_i · sat_j` is collinear with the main effects, and the *split* between
"channel i", "channel j" and "i × j" is poorly determined — the prior does much
of the work. The shrink-to-zero prior guards against spurious synergy, and the
recovered **sign** is more trustworthy than the magnitude (verified on
`synth/dgp.make_synergy`, where a planted `gamma > 0` TV×Search synergy is
recovered as a positive coefficient). To actually identify a synergy, vary the
two channels independently — a designed flighting/geo experiment. Not supported
under the multiplicative specification (ignored with a warning — it is an
additive-model term).

## Tests

`tests/test_channel_interactions.py` — config, off-is-additive, on-structure,
unknown-channel skip, positive-synergy sign recovery, negative-sign prior,
decomposition closure + reporting separation.
