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
    """

    #: Recorded into the registry manifest + serialized metadata so a consumer
    #: can detect the contract a model was authored against.
    GARDEN_CONTRACT_VERSION: str = GARDEN_CONTRACT_VERSION


__all__ = ["CustomMMM"]
