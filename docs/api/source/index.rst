MMM Framework Documentation
===========================

**MMM Framework** is a production-ready Bayesian Marketing Mix Modeling framework
extending PyMC-Marketing with advanced capabilities for modeling marketing effectiveness.

The framework emphasizes:

* **Methodological rigor** over specification shopping
* **Genuine uncertainty quantification** through Bayesian inference
* **Hierarchical modeling** for partial pooling across geographies and products
* **Pre-specified analyses** to reduce researcher degrees of freedom

Installation
------------

Install mmm-framework using UV (recommended):

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/redam94/mmm-framework.git
   cd mmm-framework

   # Install with UV
   uv sync

   # Or with pip
   pip install -e .

Quick Example
-------------

.. code-block:: python

   from mmm_framework import (
       MFFLoader,
       ModelConfigBuilder,
       BayesianMMM,
   )

   # Load data
   loader = MFFLoader(config=mff_config)
   panel = loader.load("data.csv")

   # Build configuration
   config = (
       ModelConfigBuilder()
       .with_kpi("sales")
       .build()
   )

   # Fit model
   model = BayesianMMM(panel, config)
   results = model.fit(draws=1000, tune=500)

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
