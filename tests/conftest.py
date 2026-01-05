"""
Pytest configuration and fixtures for MMM Framework tests.

Note: Most fixtures are defined in individual test files to avoid
import issues with the evolving mmm_framework API.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
