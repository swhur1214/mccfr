import numpy as np


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Return the regret-matching strategy for one information set."""
    pos = np.maximum(regret, 0.0)
    total = pos.sum()
    if total > 0:
        return pos / total
    return np.full(len(regret), 1.0 / len(regret))
