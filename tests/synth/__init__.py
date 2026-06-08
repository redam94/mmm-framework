"""Structural-violation synthetic data for stress-testing the MMM.

The framework's existing synthetic fixtures are *well-specified*: data is drawn
from (essentially) the same generative family the model assumes, so a passing
recovery test only proves the sampler works when reality matches the model. Real
marketing data violates the model's structure in ways that bias the answer
*silently* -- the fit converges, the refutation suite passes, and the ROI is
wrong.

This package generates data that deliberately breaks each structural assumption
(see :mod:`tests.synth.dgp`) and scores how badly the model degrades and whether
the existing diagnostics catch it (see :mod:`tests.synth.harness`).
"""
