"""A trivial bespoke garden model for the contract matrices (PR 0.1/0.2).

The point of the subclass is IDENTITY, not behaviour: the fingerprint case
must show the graph is byte-identical to the base model's, and the serializer
round-trip must reconstruct the SAME subclass rather than quietly demoting a
bespoke model to ``BayesianMMM`` on load.
"""

from mmm_framework.garden import CustomMMM


class ContractRoundTripMMM(CustomMMM):
    """Bespoke-in-name-only: exercises the garden identity path."""


GARDEN_MODEL = ContractRoundTripMMM
