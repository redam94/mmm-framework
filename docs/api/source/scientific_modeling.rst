Scientific Statistical Modeling
===============================

A principled approach to building, evaluating, and iterating on statistical models.
This guide establishes the philosophical foundations that underpin rigorous quantitative
analysis—applicable to Marketing Mix Models and beyond.

.. note::

   **Core Insight**: Scientific modeling is not about finding "the correct model." It is
   about building useful representations of reality, understanding their limitations, and
   honestly communicating what they can and cannot tell us about the questions we care about.


What is a Statistical Model?
----------------------------

A **statistical model** is a mathematical description of a data-generating process.
It specifies a family of probability distributions indexed by parameters, where different
parameter values correspond to different hypotheses about how the data arose.

Models are not photographs of reality—they are *useful simplifications*. A map of
New York City is not the city itself, but it can help you navigate. Similarly, a model
of sales response to advertising is not the actual cognitive and behavioral processes
involved, but it can help you make better decisions.

This perspective is liberating: we don't need to find the "true" model (which doesn't exist).
We need to find models that are useful for our purposes while being honest about their
limitations.


All Models Are Wrong
--------------------

    "All models are wrong, but some are useful."

    — George E.P. Box

Box's aphorism captures a fundamental truth. Every model makes simplifying assumptions that
do not perfectly match reality. The question is not "is this model true?" but rather:

- Is the model useful for the decisions I need to make?
- Are its predictions accurate enough for my purposes?
- Am I aware of its limitations and their implications?

A model that captures the main drivers of sales may be highly useful even if it ignores
subtle effects that contribute only 2% of variance. The key is knowing which simplifications
matter for your question and which don't.


The Generative Perspective
--------------------------

A model is **generative** if it can be used to simulate new data. Given parameter
values, you can run the model forward to produce synthetic observations. This is the key
test of understanding: if you cannot simulate data from your model, you do not fully
understand it.

The generative perspective offers several advantages:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Advantage
     - Description
   * - **Simulation**
     - You can generate fake data before seeing real data, checking whether your model
       can produce plausible outcomes
   * - **Prior predictive checks**
     - Sample from priors, push through the model, and examine implied predictions—catching
       implausible assumptions early
   * - **Posterior predictive checks**
     - After fitting, simulate new data from the posterior and compare to
       observations—assessing model adequacy
   * - **Calibration**
     - Test whether predictive intervals have nominal coverage on held-out data
   * - **Transparent assumptions**
     - Every assumption is explicit in the generative story—nothing hidden in "black box"
       algorithms


Questions Drive Models
----------------------

Scientific modeling begins with a question, not with data or techniques. The question
determines what model structure is appropriate, what data are needed, and how to evaluate
success.

**Descriptive Questions**
    "What patterns exist in the data?" Summarize, visualize, identify regularities.
    Less concerned with causation.

**Predictive Questions**
    "What will happen next?" Forecast future values. Model complexity traded against
    generalization.

**Causal Questions**
    "What would happen if we intervened?" Requires identifying assumptions and often
    experimental validation.

Marketing Mix Models typically aim to answer causal questions: "What is the effect of
advertising on sales?" This is the hardest type of question to answer from observational
data, and requires explicit assumptions about the data-generating process.


The Generative Story
--------------------

Every model should have a **generative story**: a narrative description of how you believe
the data came to be. For an MMM, this might be:

1. There is a baseline level of sales that would occur without any marketing.
2. This baseline varies over time due to trend and seasonality.
3. Marketing activities (TV, digital, etc.) increase sales above baseline.
4. Each channel has carryover effects (adstock) and diminishing returns (saturation).
5. External factors (weather, economy, competitors) create additional variation.
6. The observed sales = baseline + marketing effects + external effects + noise.

Writing down this story forces you to be explicit about your assumptions. Each element
of the story corresponds to a component of the mathematical model.


Model Components
----------------

Likelihood
~~~~~~~~~~

The probability of observed data given parameters. How is noise distributed?
Normal? Heavy-tailed? Count data?

Structural Model
~~~~~~~~~~~~~~~~

Functional relationships between variables. Linear? Nonlinear? Captures the
"systematic" part of the data-generating process.

Priors
~~~~~~

Probability distributions over unknown parameters encoding beliefs before
seeing data. Essential for Bayesian inference.

Choosing a Likelihood
~~~~~~~~~~~~~~~~~~~~~

The likelihood should match the nature of the outcome:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Outcome Type
     - Common Likelihood
     - Key Property
   * - Continuous, unbounded
     - Normal (Gaussian)
     - Symmetric, thin tails
   * - Continuous, positive
     - Log-normal, Gamma
     - Right-skewed, positive support
   * - Count data
     - Poisson, Negative binomial
     - Discrete, non-negative
   * - Binary
     - Bernoulli (via logit/probit)
     - 0/1 outcomes
   * - Proportions (0-1)
     - Beta
     - Bounded continuous


Iteration as Learning
---------------------

Model building is inherently iterative. We build, check, revise, and expand—learning about
both the data and the phenomenon through this cycle. This is not a failure of method;
it is the method.

Each iteration teaches us something:

- **Prior predictive failures** reveal that our encoded assumptions imply impossible outcomes
- **Computational failures** often indicate model misspecification or identification problems
- **Posterior predictive failures** show where the model fails to capture data patterns
- **Sensitivity analysis** reveals which conclusions are robust and which are fragile


Honest Iteration vs. Specification Shopping
-------------------------------------------

There is a crucial distinction between legitimate scientific iteration and problematic
"specification shopping." Both involve changing models, but they differ fundamentally
in what drives the changes.

**Honest Scientific Iteration** — Changes driven by:

- Diagnostic failures (divergences, poor mixing)
- Posterior predictive mismatches
- Domain knowledge about missing structure
- Pre-specified model expansion criteria

The goal is model improvement based on evidence of inadequacy.

**Specification Shopping** — Changes driven by:

- Coefficient has "wrong" sign
- Effect not statistically significant
- Results don't match expectations
- ROI falls below threshold

The goal is obtaining desired results, not model improvement.

.. warning::

   **The Winner's Curse**

   Even when a true effect exists, selecting the specification with the highest t-statistic
   biases estimates upward. The "winning" specification likely benefited from favorable noise
   in that particular sample. This is why specification-shopped effects often fail to
   replicate: you selected on noise, not signal.

Pre-Specification: The Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solution to specification shopping is **pre-specification**: commit to your model
structure before looking at results. This doesn't prevent iteration—you can still revise
models—but it requires documenting and justifying changes, making the distinction between
planned and exploratory analyses explicit.


Expanding Models
----------------

When a model fails predictive checks or sensitivity analysis reveals fragility, we may
need to expand it. Model expansion should be driven by the nature of the failure, not
by a desire for different results.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Observed Problem
     - Possible Expansion
   * - Residual autocorrelation
     - Add lagged terms, AR errors, or state-space dynamics
   * - Non-constant variance
     - Model heteroskedasticity, use robust likelihood
   * - Outliers in posterior predictive
     - Use heavy-tailed distribution, mixture model
   * - Group-level variation
     - Hierarchical/multilevel structure
   * - Non-linear patterns
     - Flexible basis functions, splines, saturation curves


Predictive Checking
-------------------

The primary tool for evaluating models is **predictive checking**: comparing model
predictions to observed data. There are two forms:

Prior Predictive Checks
~~~~~~~~~~~~~~~~~~~~~~~

Generate data using only priors (before fitting). Do the implied predictions look
plausible? Could the observed data have come from this prior predictive distribution?

.. code-block:: python

   # Prior predictive check
   with model:
       prior_pred = pm.sample_prior_predictive(samples=500)

   # Examine the implied distribution of predictions
   y_prior = prior_pred.prior_predictive["y"].values

   # Check: are these values plausible for sales data?
   # - All positive? (sales can't be negative)
   # - Reasonable range? (not implying billion-dollar weeks)
   # - Sensible variation? (not identical across all scenarios)

Posterior Predictive Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate replicated data from the posterior (after fitting). Compare statistics of
replicated data to observed data. Where does the model fail to reproduce patterns?

.. code-block:: python

   # Posterior predictive check
   with model:
       post_pred = pm.sample_posterior_predictive(trace)

   # Compare observed and replicated data
   y_rep = post_pred.posterior_predictive["y"].values
   y_obs = data["sales"].values

   # Visual checks
   # - Does distribution of y_rep match distribution of y_obs?
   # - Do time-series patterns match?
   # - Are extreme values reproduced?


Sensitivity Analysis
--------------------

Sensitivity analysis asks: *how much do conclusions change if we change assumptions?*
Conclusions that are robust to reasonable alternative assumptions are more credible than
those that depend on specific choices.

What to Vary
~~~~~~~~~~~~

- Prior distributions (especially on key parameters)
- Functional forms (linear vs. nonlinear, different saturation curves)
- Lag structures (different adstock parameters)
- Control variable sets (within theoretically justified bounds)

.. danger::

   Do NOT vary specifications until you get desired results.


External Validation
-------------------

The strongest test of a model is its performance on data it has never seen—especially
data generated by experimental intervention. For MMMs, this typically means comparing
model predictions to results from geo-lift experiments or randomized holdout tests.

.. note::

   **Why Experimental Validation Matters**

   Observational models can fit historical data well while producing biased causal estimates.
   Experimental validation provides ground truth against which to calibrate model predictions.
   Industry studies suggest two-thirds of uncalibrated MMM estimates require significant
   adjustment when compared to experimental results.


Starting Simple
---------------

A common mistake is to start with a complex model. Better practice:

- Start with the simplest model that could possibly work
- Add complexity only when diagnostics indicate it's needed
- Document each addition and why it was made
- Compare simpler and more complex models on held-out data

A simple model that you understand deeply is more valuable than a complex model
that produces mysteriously "better" results.


When to Stop
------------

Model building could continue forever. Practical stopping rules:

- **Predictive adequacy**: Posterior predictive checks pass for features that matter
- **Computational stability**: MCMC diagnostics are acceptable
- **Diminishing returns**: Additional complexity improves fit marginally
- **Resource constraints**: Time and computational budget exhausted

Importantly, stopping is not the same as claiming the model is "correct." It means
the model is adequate for current purposes, given current constraints.


Communicating Models
--------------------

How you communicate model results is as important as the modeling itself. Key principles:

- Report uncertainty honestly—credible intervals, not just point estimates
- Distinguish robust findings from fragile ones
- Explain assumptions and their implications for conclusions
- Acknowledge limitations without burying the signal in caveats
- Do NOT present specification-shopped results as definitive

.. note::

   **Uncertainty Is Information**

   Wide credible intervals are not a failure of analysis—they are honest communication that
   the data cannot distinguish between hypotheses. Wide credible intervals tell you "we need
   more data or experimentation to decide this confidently." That information is valuable:
   it prevents overconfident decisions and identifies where to invest in learning.


Summary: The Scientific Modeling Mindset
----------------------------------------

**Question First**
    Start with the decision, not the technique. Let the question drive model structure.

**Tell the Story**
    Every model should have a generative story. If you can't simulate data, you don't
    understand the model.

**Iterate Honestly**
    Model building is iterative—but changes should be driven by diagnostics, not desired
    results.

**Check Everything**
    Prior predictive, posterior predictive, diagnostics, sensitivity. Trust but verify.

**Quantify Uncertainty**
    All models are wrong. Honest uncertainty quantification tells us how wrong they might be.

**Communicate Clearly**
    Report what you know, what you don't, and what would change your conclusions.


Further Reading
---------------

- Gelman, A., et al. (2020). *Bayesian Workflow.* arXiv:2011.01808.
  `[Link] <https://arxiv.org/abs/2011.01808>`_
- Box, G. E. P. (1976). *Science and Statistics.* Journal of the American Statistical
  Association.
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press.
  `[Link] <https://xcelab.net/rm/statistical-rethinking/>`_
- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). *False-Positive Psychology.*
  Psychological Science. `[Link] <https://doi.org/10.1177/0956797611417632>`_
