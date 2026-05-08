import numpy as np

from src.regret_matching import regret_matching


def test_regret_matching_uniform_fallback():
    strategy = regret_matching(np.array([-2.0, 0.0, -1.0]))

    np.testing.assert_allclose(strategy, np.full(3, 1.0 / 3.0))


def test_regret_matching_normalizes_positive_regret():
    strategy = regret_matching(np.array([-1.0, 2.0, 6.0]))

    np.testing.assert_allclose(strategy, np.array([0.0, 0.25, 0.75]))
