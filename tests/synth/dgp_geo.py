"""Compatibility shim: the geo DGPs moved to :mod:`mmm_framework.synth.dgp_geo`.

Re-exports the full namespace (including private helpers) so existing
imports keep working; see :mod:`tests.synth.dgp` for the rationale.
"""

from mmm_framework.synth import dgp_geo as _impl

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
