"""Compatibility shim: the DGPs moved to :mod:`mmm_framework.synth.dgp`.

The implementation now ships inside the package so the agent (and any user
code) can generate realistic synthetic data. This module re-exports the full
namespace — including private helpers like ``_geom_adstock`` / ``_ALPHA`` that
the tests and notebook builders reach into — so existing imports keep working.
"""

from mmm_framework.synth import dgp as _impl

_EXCLUDED = {
    "__name__",
    "__file__",
    "__package__",
    "__loader__",
    "__spec__",
    "__builtins__",
    "__cached__",
    "__path__",
}
globals().update({k: v for k, v in vars(_impl).items() if k not in _EXCLUDED})
del _impl
