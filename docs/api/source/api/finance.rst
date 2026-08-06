Finance
=======

.. module:: mmm_framework.finance

The margin/valuation basis behind every dollar recommendation (v1.4). A KPI
unit is never silently worth one dollar: ``kpi_to_dollars`` resolves a
declared valuation with provenance and **refuses** (``is_dollar=False``) when
nothing resolved. The bridge-line vocabulary
(``MODELLED``/``OBSERVED``/``RESIDUAL``/``ABSORBING``/``SUPPLIED``) makes
every financial decomposition state where each number came from, and the
closure module prices whether a decomposition's parts genuinely sum to its
whole.

.. automodule:: mmm_framework.finance
   :members:
   :no-index:

Valuation
---------

.. automodule:: mmm_framework.finance.valuation
   :members:
   :undoc-members:
   :show-inheritance:

Lines
-----

.. automodule:: mmm_framework.finance.lines
   :members:
   :undoc-members:
   :show-inheritance:

Closure
-------

.. automodule:: mmm_framework.finance.closure
   :members:
   :undoc-members:
   :show-inheritance:

Evidence
--------

.. automodule:: mmm_framework.finance.evidence
   :members:
   :undoc-members:
   :show-inheritance:
