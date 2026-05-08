import math

from src.kuhn_poker import KuhnPoker


def test_chance_deals_are_uniform():
    efg = KuhnPoker.efg()
    outcomes = efg[""]["outcomes"]

    assert len(outcomes) == 6
    assert sorted(outcome for outcome, _, _ in outcomes) == sorted(KuhnPoker.DEALS)
    assert math.isclose(sum(prob for _, _, prob in outcomes), 1.0)
    assert all(math.isclose(prob, 1.0 / 6.0) for _, _, prob in outcomes)


def test_terminal_utilities_are_zero_sum():
    efg = KuhnPoker.efg()

    terminals = [node for node in efg.values() if node["type"] == "TERMINAL"]
    assert terminals
    for node in terminals:
        assert sum(node["utility"]) == 0
