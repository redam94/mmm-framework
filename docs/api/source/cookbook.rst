Cookbook: End-to-End Recipes
============================

Four self-contained recipes that cover the everyday library workflow:

#. :ref:`set up a model and its data <cookbook-setup>`,
#. :ref:`create custom estimands <cookbook-custom-estimands>`,
#. :ref:`compute estimands from a fitted model <cookbook-compute-estimands>`, and
#. :ref:`save and load a fitted model <cookbook-save-load>`.

Every code block below is copy-pasteable and was run against the library. The
later recipes reuse the fitted ``mmm`` and ``panel`` objects built in Recipe 1,
so run them in order (or in one script / notebook).

.. note::

   These recipes use the **lean modeling core** only — everything imports from
   ``mmm_framework`` and its submodules, with no web or LLM dependencies. Install
   it with ``pip install mmm-framework``.


.. _cookbook-setup:

1. Set up a model and its data
------------------------------

A model needs three things: a **panel** of data, a **model configuration**
(priors + inference settings), and an optional **trend configuration**. There
are two ways to get a panel.

Option A — a bundled example (zero-effort)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library ships two ready-to-model example datasets so a first fit needs no
data-loading code. :func:`~mmm_framework.datasets.load_example` returns a fully built
:class:`~mmm_framework.data_loader.PanelDataset`.

.. code-block:: python

   from mmm_framework import load_example

   panel = load_example("national")   # 104 weeks, 7 channels, 6 controls
   print(panel.summary())

Option B — build a panel from your own data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most real data starts **wide**: one column per variable, one row per week.
:func:`~mmm_framework.data_loader.mff_from_wide_format` reshapes it into the internal
long-format Master Flat File (MFF), and
:func:`~mmm_framework.config.create_simple_mff_config` declares which variables
are the KPI, the media channels, and the controls. Then
:func:`~mmm_framework.data_loader.load_mff` builds the panel.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mmm_framework import mff_from_wide_format
   from mmm_framework.config import DimensionType, create_simple_mff_config
   from mmm_framework.data_loader import load_mff

   # A small wide frame — one column per variable (usually you would read a CSV).
   rng = np.random.default_rng(1)
   weeks = pd.date_range("2023-01-02", periods=60, freq="W-MON")
   wide = pd.DataFrame({
       "week": weeks,
       "Sales": rng.gamma(3.0, 200.0, size=len(weeks)),
       "TV": rng.gamma(2.0, 50.0, size=len(weeks)),
       "Search": rng.gamma(2.0, 30.0, size=len(weeks)),
       "Price": rng.normal(10.0, 0.5, size=len(weeks)),
   })

   # 1. Reshape wide -> MFF long form.
   mff_df = mff_from_wide_format(
       wide,
       period_col="week",
       value_columns={"Sales": "Sales", "TV": "TV", "Search": "Search", "Price": "Price"},
   )

   # 2. Declare the variable roles.
   config = create_simple_mff_config(
       kpi_name="Sales",
       media_names=["TV", "Search"],
       control_names=["Price"],
       kpi_dimensions=[DimensionType.PERIOD],   # national (no geo)
   )

   # 3. Build the panel.
   my_panel = load_mff(mff_df, config)
   print(my_panel.n_channels, "channels:", my_panel.coords.channels)

.. tip::

   For a geo (panel) model, pass ``geo_col="dma"`` to
   :func:`~mmm_framework.data_loader.mff_from_wide_format` and add
   ``DimensionType.GEOGRAPHY`` to ``kpi_dimensions``. For finer control over
   priors, adstock, and saturation per channel, build an
   :class:`~mmm_framework.config.MFFConfig` explicitly from
   :class:`~mmm_framework.config.KPIConfig`, ``MediaChannelConfig``, and
   :class:`~mmm_framework.config.ControlVariableConfig` objects instead of the
   ``create_simple_mff_config`` shortcut.

Configure inference and fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~mmm_framework.builders.model.ModelConfigBuilder` is a fluent builder for the
inference and prior settings; a :class:`~mmm_framework.model.trend_config.TrendConfig` sets the
baseline trend. Construct a :class:`~mmm_framework.model.base.BayesianMMM` from the panel
and configs, then :meth:`~mmm_framework.model.base.BayesianMMM.fit`.

.. code-block:: python

   from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig, TrendType

   model_config = (
       ModelConfigBuilder()
       .bayesian_numpyro()     # fast JAX/NumPyro NUTS sampler
       .with_chains(4)
       .with_draws(500)        # small + fast for a first run
       .with_tune(500)
       .build()
   )
   trend_config = TrendConfig(type=TrendType.LINEAR)

   mmm = BayesianMMM(panel, model_config, trend_config)
   results = mmm.fit(random_seed=42)

   print("approximate:", results.approximate)
   print("max R-hat:", round(results.diagnostics["rhat_max"], 3))   # ~1.0 = converged

On a laptop this national model fits in roughly 15–25 seconds.

.. note::

   Need an answer in *seconds* while iterating on priors or structure? Pass an
   **approximate** method — ``mmm.fit(method="map", random_seed=42)`` (or
   ``"advi"`` / ``"pathfinder"``) returns in seconds. The result is a drop-in for
   the full trace (``predict`` and reporting work), but its uncertainty is **not
   calibrated** and R-hat / ESS are ``None`` — re-fit with the default NUTS
   sampler before trusting the intervals. ``results.approximate`` is ``True`` for
   these fits.


.. _cookbook-custom-estimands:

2. Create custom estimands
--------------------------

An **estimand** is a named, serializable counterfactual quantity —
``reduce( op( quantity | intervention, quantity | baseline ) ) / denominator`` —
realized from the posterior as a mean + credible interval. The library ships
built-ins (ROI, marginal ROAS, contribution); you can declare your own from the
same building blocks in :mod:`mmm_framework.estimands`.

Start from a built-in
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mmm_framework.estimands import registry

   builtin = registry.get("marginal_roas")   # a fresh Estimand instance
   print(builtin.name, "|", builtin.units)    # marginal_roas | mROAS

The MMM built-ins are ``contribution_roi``, ``counterfactual_roi``,
``marginal_roas``, ``contribution``, ``awareness_lift`` and
``cost_per_conversion``.

A single-channel ROI
~~~~~~~~~~~~~~~~~~~~~~

An :class:`~mmm_framework.estimands.spec.Estimand` has a ``numerator`` (a
:class:`~mmm_framework.estimands.spec.Contrast` or a bare quantity), an optional
``denominator``, a ``realization`` profile that pins the exact arithmetic, and
``required_capabilities`` that gate which models it applies to. This one is the
dashboard decomposition ROI, but scoped to a single channel (``target="TV"``):

.. code-block:: python

   from mmm_framework.estimands import (
       Contribution, Estimand, ObservedInput, Realization, Summary,
   )

   tv_roi = Estimand(
       name="tv_roi",
       kind="roi",
       numerator=Contribution(target="TV", source="in_graph_deterministic"),
       denominator=ObservedInput(target="TV", source="panel"),
       op_ratio_zero_denominator="skip",
       realization=Realization(point_rule="mean_of_samples", hdi_method="az_hdi"),
       summaries=[Summary(name="prob_profitable", threshold=1.0, side="gt")],
       required_capabilities=["HAS_CONTRIBUTION_DETERMINISTIC"],
       units="ROI",
       causal_assumptions="Decomposition ROI for TV over the full period.",
   )

A per-channel marginal ROAS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the sentinel target :data:`~mmm_framework.estimands.spec.ALL_CHANNELS` (``"*"``)
to mean *"expand this estimand once per channel"* — the evaluator emits one
result per channel, keyed ``"<name>:<channel>"``. This is a marginal ROAS from a
larger ``+25%`` spend bump than the built-in's default ``+10%``:

.. code-block:: python

   from mmm_framework.estimands import (
       ALL_CHANNELS, Contrast, Estimand, MarginalSpend, Outcome, Realization, ScaleInput,
   )

   mroas_plus25 = Estimand(
       name="mroas_plus25",
       kind="marginal_roas",
       numerator=Contrast(
           quantity=Outcome(),
           intervention=ScaleInput(target=ALL_CHANNELS, factor=1.25),
           baseline=None,          # None == the factual (observed) world
           op="difference",
           reduce="sum",
           paired_seed=True,        # share draws so observation noise cancels
       ),
       denominator=MarginalSpend(target=ALL_CHANNELS, intervention_ref="numerator"),
       op_ratio_zero_denominator="zero",
       realization=Realization(point_rule="diff_of_means", hdi_method="finite_percentile"),
       required_capabilities=["HAS_CONTRIBUTIONS"],
       units="mROAS",
       causal_assumptions="Incremental KPI per incremental dollar from a +25% spend bump.",
   )

Estimands are pure Pydantic, so they serialize to and from JSON for storage or
transport:

.. code-block:: python

   as_json = tv_roi.to_dict()
   restored = Estimand.from_dict(as_json)
   assert restored == tv_roi


.. _cookbook-compute-estimands:

3. Compute estimands from a fitted model
----------------------------------------

:meth:`~mmm_framework.model.base.BayesianMMM.evaluate_estimands` realizes estimands from the
fitted posterior and returns a dict of
:class:`~mmm_framework.estimands.spec.EstimandResult` keyed by name (wildcard
estimands expand to ``"<name>:<channel>"``). It never raises for an
unsupported estimand — it returns one with ``status="unsupported"``.

.. code-block:: python

   # (a) With no argument: the model's capability-based defaults
   #     (contribution_roi, marginal_roas, contribution), one per channel.
   defaults = mmm.evaluate_estimands()

   # (b) Specific built-ins by name.
   by_name = mmm.evaluate_estimands(["contribution_roi"])
   for key, res in list(by_name.items())[:3]:
       print(f"{key:24} mean={res.mean:.3f} "
             f"[{res.hdi_low:.3f}, {res.hdi_high:.3f}] {res.units}")
   # contribution_roi:TV      mean=2.640 [0.987, 4.305] ROI   (illustrative;
   # contribution_roi:Search  mean=0.405 [0.158, 0.682] ROI    exact values
   # contribution_roi:Social  mean=2.474 [0.675, 4.791] ROI    vary per fit)

   # (c) Your own estimands, mixed with built-in names.
   custom = mmm.evaluate_estimands([tv_roi, mroas_plus25, "marginal_roas"])

   r = custom["tv_roi"]                          # fixed target -> a single key
   print(r.mean, r.hdi_low, r.hdi_high, r.extra["prob_profitable"])

   for key in [k for k in custom if k.startswith("mroas_plus25:")]:
       rc = custom[key]                          # wildcard -> one key per channel
       print(key, round(rc.mean, 3), rc.units, rc.status)

Each :class:`~mmm_framework.estimands.spec.EstimandResult` carries the point estimate
(``mean``), the credible interval (``hdi_low`` / ``hdi_high`` at ``hdi_prob``),
``units``, a ``status``, and an ``extra`` dict with companion numbers — the tail
probabilities you declared in ``summaries`` (e.g. ``prob_profitable``), the raw
contribution and its interval, spend, and cross-channel ``contribution_pct``.

Declare estimands on the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``declared_estimands`` and the model uses them as its defaults — they are
also auto-evaluated at fit time (attached to ``results.estimands``) and travel
with the model when you save it.

.. code-block:: python

   mmm.declared_estimands = [tv_roi, mroas_plus25]
   mmm.evaluate_estimands()          # now realizes tv_roi + mroas_plus25

For explicit control over the random seed (paired vs. unpaired draws), use the
evaluator directly:

.. code-block:: python

   from mmm_framework.estimands import EstimandEvaluator

   evaluator = EstimandEvaluator(mmm, random_seed=7)
   results_dict = evaluator.evaluate([registry.get("contribution")])


.. _cookbook-save-load:

4. Save and load a fitted model
-------------------------------

:class:`~mmm_framework.serialization.MMMSerializer` writes a fitted model to a
directory (metadata, configs, scaling parameters, and a compressed trace) and
loads it back with the posterior intact.

.. code-block:: python

   from mmm_framework import load_example
   from mmm_framework.serialization import MMMSerializer

   # Save the fitted model (the trace rides along).
   MMMSerializer.save(mmm, "models/national_mmm")

   # ... later, or in a fresh process ...

   # A core (BayesianMMM) save is rebuilt against a COMPATIBLE panel — same
   # channels, controls, and dimensions as the original.
   panel = load_example("national")
   loaded = MMMSerializer.load("models/national_mmm", panel)

   # The loaded model carries the posterior — use it directly.
   print(loaded._trace is not None)                     # True
   print([e.name for e in loaded.declared_estimands])   # ['tv_roi', 'mroas_plus25']
   again = loaded.evaluate_estimands(["contribution_roi"])

The loaded object is a fully functional :class:`~mmm_framework.model.base.BayesianMMM`:
call :meth:`~mmm_framework.model.base.BayesianMMM.predict`,
:meth:`~mmm_framework.model.base.BayesianMMM.evaluate_estimands`, or feed it to the
reporting layer without re-fitting.

.. warning::

   Core saves **require** a compatible ``panel`` at load time — the model is
   rebuilt against it, and mismatched channels or controls raise a ``ValueError``.
   The extended models (``NestedMMM`` / ``MultivariateMMM`` / ``CombinedMMM`` /
   ``StructuralNestedMMM``) use a different, self-contained flavor and load
   **panel-free**: ``MMMSerializer.load("path/to/extended_model")`` with no second
   argument.

.. seealso::

   The :doc:`api/index` has the full reference for every class used here, and
   :doc:`scientific_modeling` explains the causal reasoning behind the estimands.
