"""Native loader for the flexible :class:`~mmm_framework.dataset.Dataset`.

This is the non-MFF front door: a **wide**, role-tagged table (a CSV / parquet
file or a DataFrame) plus a :class:`~mmm_framework.config.dataset.DatasetSchema`
becomes a :class:`Dataset` directly — no kpi/media/control classification. It is
how a genuinely non-MMM dataset (a CFA / LCA indicator matrix, a survey) is loaded
without riding an MMM panel. Geo / product panels keep using
:func:`mmm_framework.data_loader.load_mff`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config.dataset import DatasetSchema
from .dataset import Dataset


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read a wide table from disk by extension (csv / tsv / parquet / json)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(p, sep="\t")
    if suffix == ".json":
        return pd.read_json(p)
    return pd.read_csv(p)


def load_dataset(source: str | Path | pd.DataFrame, schema: DatasetSchema) -> Dataset:
    """Load a wide, role-tagged table into a :class:`Dataset` per ``schema``.

    Parameters
    ----------
    source:
        A path to a wide table (csv / tsv / parquet / json) or an in-memory
        DataFrame whose columns are tagged by ``schema``.
    schema:
        The :class:`DatasetSchema` mapping columns → roles. Every binding name must
        be a column of the table.
    """
    df = source if isinstance(source, pd.DataFrame) else _read_table(source)
    return Dataset.from_wide(df, schema)


__all__ = ["load_dataset"]
