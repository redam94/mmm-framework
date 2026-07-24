Getting Started
===============

This guide takes you from ``pip install`` to a fitted, validated Bayesian
Marketing Mix Model and a client-ready report — first with a one-script quick
start, then step by step.

What you'll build
-----------------

A Marketing Mix Model (MMM) decomposes a business outcome (sales, sign-ups,
revenue) into the contribution of each marketing channel, controlling for
everything else that moves the outcome — price, seasonality, distribution,
competitor activity. This framework builds that model as a **causal, Bayesian**
model:

* **Causal**, because the goal is the *incremental* effect of spend — what would
  have happened without it — not just a correlation. You declare which controls
  are confounders so the framework can close the back-door paths that otherwise
  bias ROI upward.
* **Bayesian**, because every number comes with honest uncertainty. You get a
  posterior distribution for each channel's effect, not a single point estimate
  that implies false precision.

.. note::

   This is the **library reference** and covers the ``mmm-framework`` Python
   package. The web application (data upload, config UI, the LLM "oracle" agent)
   ships separately as ``mmm-framework-server``; see the
   `main documentation site <https://redam94.github.io/mmm-framework/>`_ for the
   platform guide.


Installation
------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   # Lean modeling core — business logic only, no web or LLM dependencies.
   pip install mmm-framework

   # Optional: the LangGraph oracle-agent / LLM stack.
   pip install "mmm-framework[agents]"

The framework requires **Python 3.12+**. The lean core pulls in the numerical
stack (PyMC 6, NumPyro, ArviZ, pandas); the ``[agents]`` extra adds the
LangGraph/LLM dependencies.

From source (development)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/redam94/mmm-framework.git
   cd mmm-framework

   # Full dev workspace: core + [agents] + the API server package (uses uv).
   uv sync

   # ...or just the lean library, with pip:
   pip install -e .

Verify your installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mmm_framework
   print("mmm-framework version:", mmm_framework.__version__)

   from mmm_framework import BayesianMMM, ModelConfigBuilder, load_example
   print("core components import OK")


Fastest path: fit a model in one script
---------------------------------------

The library bundles ready-to-model example datasets (with sealed answer keys),
so a first fit needs no data-loading code. This script loads 104 weeks of
national weekly data, fits a model, and grades the estimated ROI against the
ground truth.

.. code-block:: python

   from mmm_framework import (
       load_example,
       load_example_answer_key,
       BayesianMMM,
       ModelConfigBuilder,
       TrendConfig,
       TrendType,
   )

   # 1. Load a bundled example — a ready-to-fit panel.
   panel = load_example("national")
   print(panel.summary())

   # 2. Configure inference (fast JAX/NumPyro sampler) and a linear trend.
   model_config = (
       ModelConfigBuilder()
       .bayesian_numpyro()      # ~3x faster than PyMC at equal draws
       .with_chains(4)
       .with_draws(500)         # small + fast for a first run
       .with_tune(500)
       .build()
   )
   trend_config = TrendConfig(type=TrendType.LINEAR)

   # 3. Fit. On a laptop this national model takes roughly 15-25 seconds.
   mmm = BayesianMMM(panel, model_config, trend_config)
   results = mmm.fit(random_seed=42)
   print("max R-hat:", round(results.diagnostics["rhat_max"], 3))   # ~1.0 = converged

   # 4. The headline: each channel's return on ad spend (contribution / spend).
   decomp = mmm.compute_component_decomposition()
   roi = (decomp.media_by_channel.sum() / panel.X_media.sum()).sort_values(ascending=False)
   print("\nEstimated ROI by channel:")
   print(roi.round(2))

   # 5. This example ships a SEALED answer key — grade the estimate against truth.
   truth = load_example_answer_key("national")["true_roas"]
   for ch in roi.index:
       print(f"  {ch:<8} estimated {roi[ch]:>5.2f}   true {truth[ch]:>5.2f}")

.. admonition:: What just happened
   :class: tip

   You fit a full Bayesian MMM — per-channel adstock (carryover) and saturation
   (diminishing returns), a trend, and controls — then read the posterior
   channel decomposition and divided by spend to get ROI. Because this example
   ships ground truth, you can *see* that the estimates land near the true
   values: this framework is built and tested against synthetic worlds with a
   known causal answer.


Your first model, step by step
------------------------------

Step 1 — Prepare your data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data enters as a **Master Flat File (MFF)** — a long-format table with one row
per ``(Period, VariableName)`` and a numeric ``VariableValue``. The fastest way
to get a panel is a bundled example; to use your own data, reshape a wide table
(one column per variable) with :func:`~mmm_framework.data_loader.mff_from_wide_format` and
declare the roles with :func:`~mmm_framework.config.create_simple_mff_config`:

.. code-block:: python

   import pandas as pd
   from mmm_framework import mff_from_wide_format
   from mmm_framework.config import DimensionType, create_simple_mff_config
   from mmm_framework.data_loader import load_mff

   # your_data: a wide frame with a date column + one column per variable.
   mff_df = mff_from_wide_format(
       your_data,
       period_col="week",
       value_columns={"Sales": "Sales", "TV": "TV", "Search": "Search", "Price": "Price"},
   )
   config = create_simple_mff_config(
       kpi_name="Sales",
       media_names=["TV", "Search"],
       control_names=["Price"],
       kpi_dimensions=[DimensionType.PERIOD],
   )
   panel = load_mff(mff_df, config)

See :doc:`cookbook` (Recipe 1) for the full data-setup recipe, including geo
(panel) data and per-channel configuration.

Step 2 — Configure the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model settings are built with fluent **builders**. The
:class:`~mmm_framework.builders.model.ModelConfigBuilder` picks the sampler,
draws, and priors; a :class:`~mmm_framework.model.trend_config.TrendConfig`
sets the baseline trend:

.. code-block:: python

   from mmm_framework import ModelConfigBuilder, TrendConfig, TrendType

   model_config = (
       ModelConfigBuilder()
       .bayesian_numpyro()      # fast JAX/NumPyro NUTS sampler
       .with_chains(4)
       .with_draws(1000)
       .with_tune(1000)
       .build()
   )
   trend_config = TrendConfig(type=TrendType.LINEAR)

Each media channel gets a **geometric adstock** (carryover: this week's spend
keeps working for several weeks) and a **saturation** curve (diminishing
returns: the tenth dollar buys less than the first). Sensible defaults apply
out of the box; override them per channel with the
:class:`~mmm_framework.builders.mff.MFFConfigBuilder` and the prior/adstock/
saturation builders (see :ref:`the builder pattern <gs-builder-pattern>`).

Step 3 — Fit and check diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mmm_framework import BayesianMMM

   mmm = BayesianMMM(panel, model_config, trend_config)
   results = mmm.fit(random_seed=42)

   # ALWAYS check convergence before trusting the numbers.
   print("max R-hat:", round(results.diagnostics["rhat_max"], 3))   # want < 1.01
   print("approximate fit:", results.approximate)                   # False for NUTS

.. warning::

   **Check R-hat before reading any result.** An R-hat above ~1.01 means the
   chains have not converged and the estimates are not trustworthy — raise
   ``draws``/``tune`` or revisit the model. For a *fast* smoke check while
   iterating, ``mmm.fit(method="map")`` returns in seconds, but its uncertainty
   is **not** calibrated (``results.approximate`` is ``True`` and R-hat is
   ``None``) — re-fit with the default NUTS sampler before making decisions.

Step 4 — Analyze the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~mmm_framework.model.base.BayesianMMM.compute_component_decomposition`
splits the fitted outcome into per-channel and baseline contributions;
:meth:`~mmm_framework.model.base.BayesianMMM.evaluate_estimands` reports the
decision-grade quantities (ROI, marginal ROAS, incremental contribution) as a
mean + credible interval:

.. code-block:: python

   # Original-scale channel decomposition -> ROI.
   decomp = mmm.compute_component_decomposition()
   roi = decomp.media_by_channel.sum() / panel.X_media.sum()

   # Named estimands with uncertainty (one result per channel).
   for key, res in mmm.evaluate_estimands(["contribution_roi"]).items():
       print(f"{key:24} {res.mean:.2f}  [{res.hdi_low:.2f}, {res.hdi_high:.2f}] {res.units}")

See :doc:`cookbook` (Recipes 2 & 3) for custom estimands and the full evaluation
surface.

Step 5 — Generate a report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mmm_framework.reporting.generator.MMMReportGenerator` turns a fitted
model into a standalone, styled HTML report — decomposition, ROI, diagnostics,
and posterior-predictive fit:

.. code-block:: python

   from mmm_framework.reporting import MMMReportGenerator, ReportConfig

   report = MMMReportGenerator(
       model=mmm,
       panel=panel,
       results=results,
       config=ReportConfig(
           title="Marketing Mix Model Analysis",
           client="Acme Consumer Products",
           analysis_period="Jan 2023 - Dec 2024",
       ),
   )
   report.to_html("mmm_report.html")


.. _gs-core-concepts:

Core concepts
-------------

The MFF data format
~~~~~~~~~~~~~~~~~~~~~

The **Master Flat File** is a long-format table that carries every variable —
KPI, media, controls — in a single, dimension-aware shape:

.. code-block:: text

   Period      | Geography | VariableName | VariableValue
   2024-01-01  | (blank)   | Sales        | 15000
   2024-01-01  | (blank)   | TV           | 50000
   2024-01-08  | (blank)   | Sales        | 16500
   ...

Leave the dimension columns (``Geography``, ``Product``, ``Campaign``,
``Outlet``, ``Creative``) blank for a national model, or fill ``Geography`` for
a geo panel — the framework aligns dimensions automatically. Use
:func:`~mmm_framework.data_loader.mff_from_wide_format` to convert a wide table into this
shape.

.. _gs-builder-pattern:

The builder pattern
~~~~~~~~~~~~~~~~~~~~~

Configuration is assembled with **fluent builders** rather than large
dictionaries, so settings are discoverable and validated as you go. Every layer
has one — data (:class:`~mmm_framework.builders.mff.MFFConfigBuilder`), the model
(:class:`~mmm_framework.builders.model.ModelConfigBuilder`), and the individual
components:

.. code-block:: python

   from mmm_framework import (
       MFFConfigBuilder,
       AdstockConfigBuilder,
       SaturationConfigBuilder,
       PriorConfigBuilder,
   )

   # An adstock with an explicit decay prior.
   adstock = (
       AdstockConfigBuilder()
       .geometric()
       .with_max_lag(8)
       .with_alpha_prior(PriorConfigBuilder().beta(alpha=2, beta=2).build())
       .build()
   )

   # A data config: KPI + two national media (with carryover) + a price control.
   mff_config = (
       MFFConfigBuilder()
       .with_kpi_name("Sales")
       .add_national_media("TV", adstock_lmax=8)
       .add_national_media("Digital", adstock_lmax=4)
       .add_price_control()
       .build()
   )

The Bayesian workflow
~~~~~~~~~~~~~~~~~~~~~~~

A trustworthy model is not "fit once and read the number." The framework
supports the full Bayesian workflow:

#. **Prior predictive check** — before fitting, simulate outcomes from the
   priors alone with
   :meth:`~mmm_framework.model.base.BayesianMMM.sample_prior_predictive` to
   confirm the priors imply plausible data. (The returned ``y_obs`` is on the
   model's standardized scale; the pre-fit *Model Design Readout* renders the
   calibrated, original-scale version.)
#. **Fit and diagnose** — sample the posterior and check convergence (R-hat,
   effective sample size) via ``results.diagnostics``.
#. **Posterior predictive check** — confirm the fitted model reproduces the
   observed data; the generated report includes observed-vs-predicted fit and
   calibration views.
#. **Sensitivity & validation** — stress the result against alternative
   specifications, refutation tests, and simulation-based calibration (SBC).

See :doc:`scientific_modeling` for the reasoning behind this workflow and why it
matters for honest measurement.


Next steps
----------

* :doc:`cookbook` — four copy-pasteable recipes: data setup, custom estimands,
  computing estimands, and saving/loading a fitted model.
* :doc:`scientific_modeling` — the methodology: causal identification,
  uncertainty, and pre-specified analysis.
* :doc:`api/index` — the complete API reference for every class and function.
* The `main documentation site <https://redam94.github.io/mmm-framework/>`_ —
  tutorials, methodology deep-dives, the platform/UI guide, and the research
  blog.

.. seealso::

   Questions or issues? Open one on
   `GitHub <https://github.com/redam94/mmm-framework/issues>`_.
