API Reference
=============

This section contains the complete API reference for MMM Framework,
automatically generated from docstrings.

The ``mmm-framework`` package is the lean modeling core (business logic only);
the LangGraph agent stack is the optional ``mmm-framework[agents]`` extra, and
the FastAPI application ships separately as ``mmm-framework-server``
(``server/`` in the repository — run it with
``uvicorn mmm_framework_server.main:app``). Everything documented here is part
of the core package.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   model
   config
   builders
   data_loader
   transforms
   serialization
   analysis
   datasets

Estimands & Validation
----------------------

.. toctree::
   :maxdepth: 2

   estimands
   validation
   diagnostics
   calibration

Planning & Experiments
----------------------

.. toctree::
   :maxdepth: 2

   planning
   continuous_learning
   ltv

Data Quality & Synthetic Worlds
-------------------------------

.. toctree::
   :maxdepth: 2

   eda
   synth

Reporting
---------

.. toctree::
   :maxdepth: 2

   reporting

Platform & Extensibility
------------------------

.. toctree::
   :maxdepth: 2

   platform
   garden
   utils

Advanced Modules
----------------

.. toctree::
   :maxdepth: 2

   mmm_extensions
   dag_model_builder
   jobs
