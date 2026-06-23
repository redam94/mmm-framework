"""The flexible :class:`Dataset` container — a role-tagged superset of
:class:`~mmm_framework.data_loader.PanelDataset`.

``PanelDataset`` is shaped around the MMM roles (``y`` / ``X_media`` /
``X_controls``); a non-MMM family that wants to treat every measured column as an
*indicator* (CFA, LCA) has to abuse it by concatenating those frames back together.
``Dataset`` fixes that: it holds one tidy, role-tagged ``table`` plus a
:class:`~mmm_framework.config.dataset.DatasetSchema`, exposes generic role
accessors (:meth:`frame_for`, :meth:`matrix`, :meth:`observed`), and *also* exposes
the MMM read-surface (``.y`` / ``.X_media`` / ``.X_controls`` / ``.coords``) as
derived views — so a ``Dataset`` is duck-type-droppable wherever a ``PanelDataset``
is read.

It sits **alongside** ``PanelDataset`` (additive): adapters :meth:`from_panel` and
:meth:`as_panel` convert losslessly for the MMM roles, and
``PanelDataset.as_dataset()`` is the reverse entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config.dataset import DatasetSchema
from .config.roles import OBSERVED_ROLES, DatasetRole
from .data_loader import PanelCoordinates, PanelDataset

if TYPE_CHECKING:
    from .config.mff import MFFConfig


@dataclass
class Dataset:
    """A generic, role-tagged data container (a superset of ``PanelDataset``).

    Parameters
    ----------
    table:
        One tidy frame whose columns are tagged by ``schema``. Indexed by the same
        ``MultiIndex`` / ``DatetimeIndex`` a ``PanelDataset`` uses.
    schema:
        The :class:`DatasetSchema` mapping columns → roles.
    index:
        The row index (kept for reconstruction / alignment).
    coords:
        The reused :class:`PanelCoordinates` (time / geo / product labels).
    config:
        The originating :class:`MFFConfig` when adapted from MFF (preserved for a
        lossless ``as_panel`` round-trip); ``None`` for a natively non-MMM dataset.
    stats:
        Optional per-column statistics (mirrors ``PanelDataset.media_stats``).
    """

    table: pd.DataFrame
    schema: DatasetSchema
    index: pd.MultiIndex | pd.DatetimeIndex
    coords: PanelCoordinates
    config: "MFFConfig | None" = None
    stats: dict[str, dict] = field(default_factory=dict)

    # ---- generic role accessors ------------------------------------------
    def columns_for(self, role: DatasetRole) -> list[str]:
        """Column names bound to ``role`` that are present in ``table``."""
        return [c for c in self.schema.names_for(role) if c in self.table.columns]

    def frame_for(self, role: DatasetRole) -> pd.DataFrame:
        """The sub-frame of all columns bound to ``role`` (declaration order)."""
        return self.table[self.columns_for(role)]

    def matrix(self, *roles: DatasetRole) -> np.ndarray:
        """A ``float64`` matrix of all columns across ``roles`` (role order)."""
        cols = [c for r in roles for c in self.columns_for(r)]
        return self.table[cols].to_numpy(dtype=np.float64)

    def observed(self) -> pd.DataFrame:
        """The measured columns (target / predictor / control / indicator).

        This is what a family that has no channel/outcome split — a CFA or LCA —
        reads as its manifest-indicator matrix, replacing the old
        ``pd.concat([y, X_media, X_controls])`` hack with a single role-aware call.
        Columns are returned in role order (so for an MMM-shaped panel this is
        ``[kpi, *media, *controls]`` — identical to the legacy concat order).
        """
        cols: list[str] = []
        for role in OBSERVED_ROLES:
            for c in self.columns_for(role):
                if c not in cols:
                    cols.append(c)
        return self.table[cols]

    # ---- MMM-shaped views (drop-in for PanelDataset readers) -------------
    @property
    def y(self) -> pd.Series:
        """The first ``TARGET`` column as a Series (empty if the dataset has none)."""
        targets = self.columns_for(DatasetRole.TARGET)
        if targets:
            return self.table[targets[0]]
        return pd.Series(index=self.index, dtype=float, name=None)

    @property
    def X_media(self) -> pd.DataFrame:
        """The ``PREDICTOR`` columns (MMM media)."""
        return self.frame_for(DatasetRole.PREDICTOR)

    @property
    def X_controls(self) -> pd.DataFrame | None:
        """The ``CONTROL`` columns, or ``None`` when there are none."""
        cols = self.columns_for(DatasetRole.CONTROL)
        return self.table[cols] if cols else None

    @property
    def media_stats(self) -> dict[str, dict]:
        return self.stats

    @property
    def n_obs(self) -> int:
        return len(self.table)

    @property
    def n_channels(self) -> int:
        return len(self.columns_for(DatasetRole.PREDICTOR))

    @property
    def n_controls(self) -> int:
        return len(self.columns_for(DatasetRole.CONTROL))

    @property
    def is_panel(self) -> bool:
        return self.coords.has_geo or self.coords.has_product

    # ---- adapters ---------------------------------------------------------
    def as_panel(self) -> PanelDataset:
        """View this dataset through the MMM ``PanelDataset`` surface.

        Lossy only for non-MMM roles (indicator / offset / weight / trials are not
        represented in the MMM view but remain in ``table``).
        """
        return PanelDataset(
            y=self.y,
            X_media=self.X_media,
            X_controls=self.X_controls,
            coords=self.coords,
            index=self.index,
            config=self.config or self.schema.to_mff(),
            media_stats=self.stats,
        )

    def retag(self, schema: DatasetSchema) -> "Dataset":
        """Return a copy of this dataset re-tagged with ``schema``.

        Every binding name in ``schema`` must be a column of ``table`` (raises
        otherwise). Used to apply an explicit role mapping (from a spec or a
        reloaded model) over the columns of an already-loaded table.
        """
        missing = [b.name for b in schema.bindings if b.name not in self.table.columns]
        if missing:
            raise ValueError(
                f"DatasetSchema names not present in the data columns: {missing}; "
                f"available columns are {list(self.table.columns)}."
            )
        return Dataset(
            table=self.table,
            schema=schema,
            index=self.index,
            coords=self.coords,
            config=self.config,
            stats=self.stats,
        )

    @classmethod
    def from_panel(cls, panel: PanelDataset) -> "Dataset":
        """Wrap an existing :class:`PanelDataset` as a :class:`Dataset`.

        No data motion beyond a column-wise concat of the (already aligned) MMM
        frames; the schema is derived from the panel's ``MFFConfig``.
        """
        frames: list[pd.DataFrame] = [panel.y.to_frame()]
        if panel.X_media is not None and panel.X_media.shape[1] > 0:
            frames.append(panel.X_media)
        if panel.X_controls is not None and panel.X_controls.shape[1] > 0:
            frames.append(panel.X_controls)
        table = pd.concat(frames, axis=1)
        schema = DatasetSchema.from_mff(panel.config)
        return cls(
            table=table,
            schema=schema,
            index=panel.index,
            coords=panel.coords,
            config=panel.config,
            stats=panel.media_stats,
        )

    def summary(self) -> str:
        role_counts = {
            r.value: len(self.columns_for(r))
            for r in DatasetRole
            if self.columns_for(r)
        }
        return (
            f"Dataset(n_obs={self.n_obs}, columns={list(self.table.columns)}, "
            f"roles={role_counts})"
        )


__all__ = ["Dataset"]
