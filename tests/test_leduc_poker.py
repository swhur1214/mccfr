import math

from src.leduc_poker import LeducPoker
from src.util import collect_information_sets, exploitability, uniform_strategy


def test_leduc_private_deals_are_uniform():
    efg = LeducPoker.efg()
    outcomes = efg[""]["outcomes"]

    assert len(outcomes) == 30
    assert math.isclose(sum(prob for _, _, prob in outcomes), 1.0)
    assert all(math.isclose(prob, 1.0 / 30.0) for _, _, prob in outcomes)


def test_leduc_terminal_utilities_are_zero_sum():
    efg = LeducPoker.efg()
    terminals = [node for node in efg.values() if node["type"] == "TERMINAL"]

    assert terminals
    assert all(math.isclose(sum(node["utility"]), 0.0) for node in terminals)


def test_leduc_uniform_strategy_has_finite_exploitability():
    efg = LeducPoker.efg()
    actions, _ = collect_information_sets(efg)
    strategy = uniform_strategy(actions)

    expl = exploitability(efg, strategy)
    assert math.isfinite(expl)
    assert expl > 0.0
