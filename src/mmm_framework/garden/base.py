"""``CustomMMM`` — the recommended base class for Model Garden bespoke models.

Subclass this (or :class:`mmm_framework.BayesianMMM` directly) and override the
build / prior / transform hooks to customize. You inherit the full oracle
contract — ``fit`` / ``predict`` / ``sample_channel_contributions`` /
``compute_component_decomposition`` / serialization — so the model is
immediately runnable by the agent, gradable by the compatibility suite, and
round-trippable by :class:`~mmm_framework.serialization.MMMSerializer`.

This module imports ``BayesianMMM`` (and therefore the PyMC stack), so it is
kept out of the package ``__init__`` top level — import it lazily via
``from mmm_framework.garden import CustomMMM`` (resolved on demand).

Example
-------
>>> from mmm_framework.garden import CustomMMM
>>> class MyMMM(CustomMMM):
...     '''A bespoke model with a tighter adstock prior.'''
...     def _build_model(self):
...         model = super()._build_model()
...         # ... customise priors / structure here ...
...         return model
"""

from __future__ import annotations

from ..model import BayesianMMM
from .contract import GARDEN_CONTRACT_VERSION


class CustomMMM(BayesianMMM):
    """Base class for bespoke garden models (a thin, documented marker).

    It adds nothing to ``BayesianMMM`` except the contract-version stamp and a
    home for documentation — overriding hook methods (``_build_model``, prior
    construction, transforms) is how you customise. Keeping the standard
    constructor signature ``(panel, model_config, trend_config=...)`` is what
    lets the agent fit your model on any project's data by swapping the class.

    Bespoke configuration
    ---------------------
    To give your model its own **settable, defaulted, validated** parameters
    (instead of hard-coded class attributes) — e.g. a binomial awareness model's
    ``number_of_trials`` — set the class attribute ``CONFIG_SCHEMA`` to a
    ``pydantic.BaseModel`` subclass::

        class AwarenessParams(BaseModel):
            number_of_trials: int = Field(default=500, gt=0)
            awareness_retention: float = 0.75

        class MyAwarenessMMM(CustomMMM):
            CONFIG_SCHEMA = AwarenessParams
            def _build_model(self):
                n = self.model_params.number_of_trials  # validated + defaulted
                ...

    The agent/spec layer validates ``spec["model_params"]`` against the schema
    (applying defaults), passes it to the constructor, and the serializer
    round-trips it; its JSON Schema (``CONFIG_SCHEMA.model_json_schema()``) drives
    a params form in the UI. Likewise, declare a non-default observation family
    via ``model_config.likelihood`` (``spec["likelihood"]``) and read it in
    ``_build_model`` (e.g. ``pm.Binomial(n=n, p=sigmoid(mu), observed=self.y)``).
    See ``technical-docs/custom-model-config.md``.
    """

    #: Recorded into the registry manifest + serialized metadata so a consumer
    #: can detect the contract a model was authored against.
    GARDEN_CONTRACT_VERSION: str = GARDEN_CONTRACT_VERSION

    #: Model family kind, recorded in the manifest and used by the contract/compat
    #: to decide which gates apply. Default ``"mmm"`` (channels, spend,
    #: ``beta_<channel>`` params, channel read-ops). A **non-MMM** family (a CFA,
    #: latent-class model, …) overrides this to its own kind (e.g. ``"cfa"``) and
    #: is then exempt from the MMM-specific contract checks + compat tiers — it
    #: only needs to override ``_prepare_data`` / ``_build_model``, set
    #: ``channel_names = []``, expose a fitted posterior, and declare its own
    #: estimands via ``DEFAULT_ESTIMANDS``. See ``technical-docs/non-mmm-families.md``.
    __garden_model_kind__: str = "mmm"


__all__ = ["CustomMMM"]
