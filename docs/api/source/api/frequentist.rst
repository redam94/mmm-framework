Frequentist
===========

.. module:: mmm_framework.frequentist

Penalized point estimation with bootstrap confidence intervals: the ridge and
CVXPY-constrained estimators behind ``inference.method = "frequentist_ridge"``
/ ``"frequentist_cvxpy"``. These produce **no posterior** — intervals are
bootstrap CONFIDENCE intervals, and the framework's recovery comparison
(``tests/frequentist/test_recovery_comparison.py``) records why the Bayesian
path remains the default: ridge's value is hard constraints and speed, not
accuracy. ``cvxpy`` is imported lazily inside the constrained solver, so the
lean core install never pays for it.

.. automodule:: mmm_framework.frequentist
   :members:
   :no-index:

Design
------

.. automodule:: mmm_framework.frequentist.design
   :members:
   :undoc-members:
   :show-inheritance:

Ridge
-----

.. automodule:: mmm_framework.frequentist.ridge
   :members:
   :undoc-members:
   :show-inheritance:

Constrained
-----------

.. automodule:: mmm_framework.frequentist.constrained
   :members:
   :undoc-members:
   :show-inheritance:

Search
------

.. automodule:: mmm_framework.frequentist.search
   :members:
   :undoc-members:
   :show-inheritance:

Bootstrap
---------

.. automodule:: mmm_framework.frequentist.bootstrap
   :members:
   :undoc-members:
   :show-inheritance:
