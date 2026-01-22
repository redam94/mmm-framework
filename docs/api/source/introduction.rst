Introduction
============

A vision for rigorous, actionable marketing measurement—built on honest uncertainty
quantification, experimental validation, and the belief that better methodology leads
to better decisions.

The Vision
----------

The marketing measurement industry is at an inflection point. Clients are asking harder
questions: not just "what is the ROI of this channel" but "how confident should we be
in that number?" They notice when last year's model says television was the top performer
and this year's model says it's digital—with no change in strategy. They're beginning
to ask about validation.

This framework represents a fundamental rethinking of how marketing measurement should
work. It's built on the premise that honest uncertainty is more valuable than false
precision, and that validated predictions matter more than impressive-looking outputs.

.. note::

   **The Core Insight**: The industry is moving toward greater rigor. Organizations that
   lead this transition—rather than resist it—will build differentiated capabilities and
   client relationships grounded in demonstrated rather than asserted credibility.


The Problem We're Solving
-------------------------

Traditional marketing mix modeling often involves a practice known as **specification
shopping**: iteratively adjusting model parameters—lags, decay rates, control variables—until
results achieve desired statistical properties or match prior expectations. While this can
incorporate genuine domain knowledge, it introduces systematic risks.

.. warning::

   **Why Specification Shopping Is Dangerous**

   When you test multiple specifications and select based on results, you invalidate
   standard statistical inference. The reported confidence intervals don't reflect actual
   uncertainty. Worse, the process systematically selects for confirming rather than
   disconfirming evidence, creating models that look good but may be dangerously miscalibrated.

Common post-hoc adjustments—like zeroing out negative media effects—don't just violate
statistical principles. They systematically bias results upward and make downstream
optimization recommendations unreliable. When everyone uses the same biased methods,
an entire industry can be confidently wrong.


Core Principles
---------------

Validation as Standard Practice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where feasible, design holdout experiments that test model predictions against reality.
This creates a feedback loop distinguishing working models from non-working ones.

Uncertainty as a Feature
~~~~~~~~~~~~~~~~~~~~~~~~

Instead of point estimates implying false precision, we quantify and communicate uncertainty.
When confident, we say so. When not, we recommend experiments rather than papering over it
with specification choices.

A Toolbox, Not a Template
~~~~~~~~~~~~~~~~~~~~~~~~~

Different business questions require different tools. Attribution, incrementality, and
optimization questions aren't all best answered by the same model. We match methodology
to question.

Pre-Specified Analysis
~~~~~~~~~~~~~~~~~~~~~~

Define modeling decisions before seeing results. This reduces researcher degrees of
freedom and ensures that findings reflect data patterns rather than analyst choices.


What Better Looks Like
----------------------

Honest Communication
~~~~~~~~~~~~~~~~~~~~

Consider the difference between these two types of reporting:

**Good**: "We estimate TV ROI at 1.4 (1.2–1.6, 80% CI). This estimate is validated against
geo experiments and robust to specification choices."

**Also Good**: "Display ROI estimates are highly uncertain, ranging from 0.5 to 2.5 across
specifications. We cannot confidently recommend budget changes without additional data."

This kind of transparency builds trust. Clients can distinguish confident recommendations
from uncertain ones. They can make informed decisions about where to act immediately
versus where to invest in additional validation.


Who This Framework Is For
-------------------------

- **Marketing analysts** with statistical backgrounds who want to move from frequentist
  to Bayesian approaches
- **Data scientists** building production MMM systems who need proper uncertainty
  quantification
- **Measurement leaders** looking to transform organizational practices around
  specification shopping
- **Researchers** interested in causal inference applications to marketing problems
- **Decision-makers** who want to understand what honest marketing measurement looks like


Technical Foundation
--------------------

The framework is built on **PyMC-Marketing** for Bayesian modeling, with a complete
technical stack including FastAPI backends, Streamlit frontends, and comprehensive
testing infrastructure.

It supports sophisticated modeling scenarios including:

- Nested models with mediated causal pathways
- Multivariate outcomes with cross-correlations
- Principled variable selection that maintains causal validity

Key technical innovations include:

- Proper handling of geo-level random effects (which can't identify national media effects)
- Bayesian variable selection that distinguishes confounders from precision controls
- Extensive diagnostics following the Bayesian workflow framework from Gelman et al. (2020)

The framework is open source and designed for both individual use and organizational
adoption. Comprehensive documentation, mathematical foundations, and educational content
help teams understand not just *how* to use these methods but *why* they matter.


Getting Started
---------------

To get started with the framework:

1. Install the package using UV (recommended)::

       git clone https://github.com/redam94/mmm-framework.git
       cd mmm-framework
       uv sync

2. Check out the :doc:`api/index` for the complete API reference
3. Explore the :doc:`scientific_modeling` guide for the philosophical foundations

For questions or feedback, open an issue on
`GitHub <https://github.com/redam94/mmm-framework/issues>`_.
