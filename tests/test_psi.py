import numpy as np

from psi import psi


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    assert psi(rng.lognormal(4, 1, 50_000), rng.lognormal(4, 1, 20_000)) < 0.01


def test_scaled_amounts_drift():
    rng = np.random.default_rng(0)
    assert psi(rng.lognormal(4, 1, 50_000), rng.lognormal(4, 1, 20_000) * 3) > 0.2


def test_missing_values_drift():
    live = np.where(np.arange(10_000) % 2 == 0, np.nan, np.arange(10_000.0))
    assert psi(np.arange(10_000.0), live) > 0.2
