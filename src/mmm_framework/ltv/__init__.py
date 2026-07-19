"""LTV / CLV modeling: RFM preprocessing + BG/NBD + Gamma-Gamma likelihoods.

``preprocess`` is pandas-only and imported eagerly; the pytensor likelihood
expressions live in :mod:`mmm_framework.ltv.likelihood` (import it explicitly —
kept out of this namespace so ``import mmm_framework.ltv`` stays light, the
same lazy-import convention as ``mmm_extensions``).
"""

from .kpi import clv_to_cac, new_customer_clv_series
from .preprocess import rfm_arrays, transactions_to_rfm

__all__ = [
    "transactions_to_rfm",
    "rfm_arrays",
    "new_customer_clv_series",
    "clv_to_cac",
]
