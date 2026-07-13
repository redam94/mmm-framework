# Future Implementation Notes

Working notes for features that are specified but not yet built. Each entry is an
implementation guide: the design, the code, the guardrails, and the evidence behind
the design decisions. This complements [`deferred-causal-features.md`](./deferred-causal-features.md),
which holds the formal contracts (R0.x cross-cutting requirements, acceptance gates).

---

## Grouped / hierarchical priors for collinear channels (DF-2)

**Status:** **Implemented behind the flag** (`ModelConfig.use_grouped_media_priors`,
builder `.with_grouped_media_priors()`), off by default. The acceptance gates checkable
today pass: A1/R0.2 (flag off ⇒ bit-identical, same-seed MAP fit), A2 (a strong channel's
between-channel ratio is preserved — no silent over-shrinkage), A3/DF2.4 (calibrated /
explicitly-priored channels excluded from the pool), and DF2.6 (pooled channels disclosed
in the ROI section). **A4 (held-out experimental validation) is still pending**, so it
ships behind the flag but is NOT recommended as a default. Detection also ships
(`validation/channel_diagnostics.py::_detect_collinear_clusters`, surfaced in §7 of
`nbs/demos/causal_features_showcase.ipynb`). Tests: `tests/test_grouped_media_priors.py`.

> **Reporting caveat:** pooled `beta_<ch>` are log-normal, so their absolute posterior
> means are not directly comparable to the independent-Gamma betas; treat the *combined*
> group effect (and per-channel *ratios* / the disclosed pooling) as the reliable read.
> This is exactly why the report discloses the pool rather than asserting a split.

### The crux: collinearity ≠ "should be pooled"

This is the distinction that decides whether a grouped prior is even the right tool, and
it is easy to miss.

Collinearity tells you the **split is unidentified** — it says *nothing* about whether the
channels' effects are *similar*. A partial-pooling prior (`β_c ~ shared mean`) is only
valid when the channels are **a priori exchangeable** (e.g. five social platforms). It
does not add information; it *imposes* the assumption that the effects are alike. Pool two
collinear-but-dissimilar channels and you do not regularize — you inject bias toward a
mean the data never supported.

**Empirical demonstration** (isolated PyMC model, `corr(Digital, Search) = 0.999`; true
`β_TV = 1.0`, `β_Digital = 2.0`, `β_Search = 0.5`; 2 chains × 400 draws):

| quantity                     | Independent (today) | Grouped (pool D+S) |
|------------------------------|---------------------|--------------------|
| β Digital                    | 1.52 ± 0.50         | 1.33 ± **0.23**    |
| β Search                     | 1.11 ± 0.52         | 1.30 ± **0.24**    |
| **split (D − S)**            | 0.41 ± **1.02**     | 0.03 ± 0.47        |
| **combined (D + S)**         | 2.64 ± **0.06**     | 2.64 ± 0.06        |
| corr(β_D, β_S) across draws  | −0.99               | −0.97              |

Reading the table:

- The data **pins the sum** (±0.06, ≈ the true 2.5) and leaves the **split** floating
  (±1.02, anti-correlated −0.99). This is the signature of collinearity.
- Pooling shrinks the split to a confident-*looking* 1.33 vs 1.30 (±0.23) — but that
  confidence is the **prior**, not data. Since the true split is 2.0 / 0.5, it is
  **biased**.
- With only 2 channels, `tau_group` is estimated from ~1 effective observation about
  between-channel spread, so the *amount* of shrinkage is set by the `tau` prior, not
  learned. It is a fixed regularizer wearing an "adaptive pooling" costume. Partial
  pooling earns its keep with **many** exchangeable channels, not a collinear pair.

### Three honest responses to collinearity

1. **Pool** — only if you genuinely believe exchangeability (many similar channels).
2. **Report the combined effect** — the group total *is* identified (±0.06 above); report
   the cluster ROI, do not assert a split. This is what the shipped diagnostic text
   already recommends, and it needs **no model change** (aggregate the cluster's columns
   of `ContributionResults.channel_contributions`).
3. **Run an experiment** — the only thing that actually identifies the split → feeds
   `roi_prior` via the calibration path (`calibration/`, §11 of the showcase notebook).

For the canonical Digital/Search case, prefer **#2 or #3**, not pooling.

### Implementation (only when the group is genuinely exchangeable)

Small, contract-preserving change. The PyMC pattern below was verified to sample cleanly
and to keep the `beta_{channel}` / `channel_contributions` contract intact.

**1. Flag (off by default — R0.1).** In `config/model.py`, `ModelConfig`:

```python
use_grouped_media_priors: bool = False   # DF2.1: partial-pool channels sharing a parent_channel
```

**2. Group source.** Already exists: set `parent_channel="paid"` on the channels deemed
exchangeable (via `MediaChannelConfigBuilder.with_parent_channel`).
`mff_config.get_hierarchical_media_groups()` (`config/mff.py:192`) returns
`{parent: [children]}` — currently computed but unused for priors. Auto-grouping from
detected collinear clusters must stay **opt-in and logged** (DF2.2), never default.

**3. The hierarchical block** — drop-in helper, called inside `with model:` before the
channel loop in `_build_model` (mirrors the geo block at `model/base.py:867`).

> **Deliberate deviation from the spec:** DF2.3 writes `β_c ~ Normal(μ, τ)`, but media
> betas are *positive* (the `Gamma` default at `model/base.py:914`). So this uses
> **log-normal** pooling — it pools toward a shared *mean* (correct), where a HalfNormal
> "shared scale" would pool toward *zero* (wrong for a coefficient that should be ~1).

```python
def _build_grouped_media_betas(self) -> dict[str, "pt.TensorVariable"]:
    """Partial-pool coefficients of channels sharing a parent_channel group.
    Non-centered, log-normal (positive). Calibrated channels (roi_prior set) are
    EXCLUDED -- their experiment prior wins (DF2.4). Emits beta_{channel} (R0.4)."""
    grouped: dict[str, "pt.TensorVariable"] = {}
    if not getattr(self.model_config, "use_grouped_media_priors", False):
        return grouped                                    # off => loop below is byte-identical
    for parent, children in self.media_groups.items():
        members = [c for c in children
                   if getattr(self.mff_config.get_media_config(c), "roi_prior", None) is None]
        if len(members) < 2:                              # nothing to pool
            continue
        mu_g  = pm.Normal(f"beta_mu_{parent}",  mu=np.log(1.5), sigma=0.5)   # log-space group mean
        tau_g = pm.HalfNormal(f"beta_tau_{parent}", sigma=0.5)              # adaptive spread
        z     = pm.Normal(f"beta_z_{parent}", 0.0, 1.0, shape=len(members)) # non-centered
        for i, ch in enumerate(members):
            grouped[ch] = pm.Deterministic(f"beta_{ch}", pt.exp(mu_g + tau_g * z[i]))
        self._pooled_channels.update(members)             # for DF2.6 reporting
    return grouped
```

**4. The one-line loop change** (`model/base.py:914`) — grouped channels use the pooled β,
everyone else keeps today's prior verbatim:

```python
beta = grouped_betas.get(channel_name)
if beta is None:
    beta = _sample_from_prior_config(
        f"beta_{channel_name}", roi_prior,
        lambda: pm.Gamma(f"beta_{channel_name}", mu=1.5, sigma=1.0))
```

That is the whole behavioral change. **Calibration precedence** (DF2.4) falls out of the
`roi_prior is None` filter; the **contract** (R0.4) holds because grouped betas are still
emitted as `beta_{channel}` Deterministics (verified present and positive).

### Guardrails this must clear (it changes posteriors)

From [`deferred-causal-features.md`](./deferred-causal-features.md) — non-negotiable,
because a feature that changes the numbers can be confidently wrong in a way a report
section cannot:

- **A1 / R0.2 — off ⇒ bit-for-bit identical** (same seed). Depends on PyMC *RV
  declaration order*, not just the flag — it holds here only because the grouped block is
  fully skipped when off. **Actually run the same-seed fit both ways and diff the
  posterior**; do not assert it. This is the one criterion checkable *today*.
- **A2 — no silent over-shrinkage:** a strongly-identified channel must escape the pool.
  (Directly in tension with 2-channel pooling — another reason that case is wrong.)
- **A3 — calibration precedence:** a calibrated channel's posterior must match its
  `roi_prior` estimate whether or not grouping is on.
- **A4 / R0.3 — held-out validation before it is a *default*:** pooled per-channel ROIs
  must show *better interval coverage* and an unbiased group total against held-out lift
  data. Ships behind the flag without it; cannot be recommended as default until it passes.
- **DF2.6 — honest reporting:** mark pooled channels, disclose "partially pooled," link
  back to the §7 collinearity finding that motivated it.

### Collinear controls are a different problem

If the collinear variables are **controls, not media**, this is the wrong prior — you want
*shrink-to-zero*, not *pool-to-shared-mean*. The framework already ships that:
`mmm_extensions/components/variable_selection.py` provides regularized horseshoe,
spike-slab, and Bayesian LASSO for correlated controls.

### Suggested build order

1. Add the flag + helper + loop change (above).
2. Write the **A1** regression test first (same-seed fit, flag on vs off, diff the
   posterior) — it is the acceptance gate for "off by default."
3. Add **A2** (strong channel escapes) and **A3** (calibration precedence) tests.
4. Wire **DF2.6** reporting (mark `_pooled_channels` in the extractor / report).
5. **A4** held-out validation is the gate for recommending it as a default — until then it
   ships behind the flag only.
