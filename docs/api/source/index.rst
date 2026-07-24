MMM Framework Documentation
===========================

**MMM Framework** is a production-ready Bayesian Marketing Mix Modeling framework
built on a standalone PyMC 6 engine — it does not subclass or depend on
PyMC-Marketing — with advanced capabilities for modeling marketing effectiveness.

The framework emphasizes:

* **Methodological rigor** over specification shopping
* **Genuine uncertainty quantification** through Bayesian inference
* **Hierarchical modeling** for partial pooling across geographies and products
* **Pre-specified analyses** to reduce researcher degrees of freedom

Installation
------------

Install the modeling library from PyPI:

.. code-block:: bash

   # Lean modeling core — business logic only (no web/LLM deps)
   pip install mmm-framework

   # Optional: the LangGraph oracle-agent / LLM stack
   pip install "mmm-framework[agents]"

The FastAPI web app ships as the separate ``mmm-framework-server`` workspace
package and is not part of the library install.

For development, install from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/redam94/mmm-framework.git
   cd mmm-framework

   # Full dev workspace: core + [agents] + the API server package
   uv sync

   # Lean library only
   uv sync --no-dev

   # Or with pip
   pip install -e .

Quick Example
-------------

The library bundles ready-to-model example datasets (with sealed answer keys),
so a first fit needs no data-loading code:

.. code-block:: python

   from mmm_framework import (
       load_example,
       load_example_answer_key,
       BayesianMMM,
       ModelConfigBuilder,
       TrendConfig,
       TrendType,
   )

   # 1. Load a bundled example — 104 weeks of national weekly data, ready to model.
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
   print("\nchannel   estimated   true")
   for ch in roi.index:
       print(f"  {ch:<8} {roi[ch]:>8.2f}   {truth[ch]:>5.2f}")

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   introduction
   scientific_modeling

.. toctree::
   :maxdepth: 2
   :caption: Guides

   cookbook

.. toctree::
   :maxdepth: 2
   :caption: Platform

   architecture
   contracts

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

More documentation
------------------

This site is the **library reference** (guides + full API docs, versioned with
the package). The broader documentation — tutorials, methodology deep-dives,
the platform/UI guide, the research blog, and client-facing material — lives
on the `main docs site <https://redam94.github.io/mmm-framework/>`_:

* `Getting started <https://redam94.github.io/mmm-framework/getting-started.html>`_
* `Architecture & flow diagrams <https://redam94.github.io/mmm-framework/architecture.html>`_
* `API contracts (v1.0) <https://redam94.github.io/mmm-framework/api-contracts.html>`_
* `REST API reference <https://redam94.github.io/mmm-framework/rest-api.html>`_
* `Research blog <https://redam94.github.io/mmm-framework/blog.html>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
