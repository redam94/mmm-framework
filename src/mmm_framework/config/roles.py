"""Generalized variable-role taxonomy for the flexible :class:`Dataset` type.

:class:`DatasetRole` is a **superset** of :class:`~mmm_framework.config.enums.VariableRole`
(``config/enums.py``): it keeps the four MMM roles (target / predictor / control /
auxiliary) and adds the roles a non-MMM family needs — ``INDICATOR`` (a CFA/LCA
manifest variable), ``GROUP`` (a hierarchy key such as geography or segment),
``TIME``, ``OFFSET`` (a known exposure), ``WEIGHT`` (a per-observation regression
weight) and ``TRIALS`` (a binomial denominator).

It is a **new** enum rather than an edit to ``VariableRole`` because ``VariableRole``
is embedded in :class:`~mmm_framework.config.mff.MFFConfig`'s ``extra="forbid"``
Pydantic graph and is serialized into saved configs; changing its members risks
existing artifacts. The bidirectional bridge below keeps the MFF ↔ Dataset
adapters lossless for the four roles the two enums share.
"""

from __future__ import annotations

from enum import Enum

from .enums import VariableRole


class DatasetRole(str, Enum):
    """Role of a column in a :class:`~mmm_framework.dataset.Dataset`.

    The first four members map 1:1 to :class:`VariableRole`; the rest generalize
    the container to non-MMM families. ``str``-valued so a role serializes as its
    plain string (``"indicator"``) in JSON specs.
    """

    TARGET = "target"  # the modeled outcome (MMM: kpi)
    PREDICTOR = "predictor"  # a regressor with a causal role (MMM: media)
    CONTROL = "control"  # a nuisance regressor (MMM: control)
    INDICATOR = "indicator"  # a measured manifest variable (CFA/LCA item)
    GROUP = "group"  # a grouping / hierarchy key (geo, product, segment)
    TIME = "time"  # the time axis
    OFFSET = "offset"  # a known additive / exposure offset (e.g. log-exposure)
    WEIGHT = "weight"  # a per-observation regression weight
    TRIALS = "trials"  # a binomial denominator (n per observation)
    AUXILIARY = "auxiliary"  # allocation weights, etc. (mirrors VariableRole.AUXILIARY)


#: The substantive *measured* roles — what a family treats as observed data
#: (everything except structural columns like TIME / GROUP / WEIGHT / OFFSET /
#: TRIALS / AUXILIARY). A CFA/LCA reads exactly these as its indicator matrix.
OBSERVED_ROLES: tuple[DatasetRole, ...] = (
    DatasetRole.TARGET,
    DatasetRole.PREDICTOR,
    DatasetRole.CONTROL,
    DatasetRole.INDICATOR,
)


#: Lossless bridge for the four roles MFF and Dataset share. ``INDICATOR`` /
#: ``GROUP`` / ``TIME`` / ``OFFSET`` / ``WEIGHT`` / ``TRIALS`` have no MFF
#: equivalent (they only appear in genuinely non-MMM datasets).
MFF_ROLE_TO_DATASET: dict[VariableRole, DatasetRole] = {
    VariableRole.KPI: DatasetRole.TARGET,
    VariableRole.MEDIA: DatasetRole.PREDICTOR,
    VariableRole.CONTROL: DatasetRole.CONTROL,
    VariableRole.AUXILIARY: DatasetRole.AUXILIARY,
}

DATASET_ROLE_TO_MFF: dict[DatasetRole, VariableRole] = {
    v: k for k, v in MFF_ROLE_TO_DATASET.items()
}


__all__ = [
    "DatasetRole",
    "OBSERVED_ROLES",
    "MFF_ROLE_TO_DATASET",
    "DATASET_ROLE_TO_MFF",
]
