import math

import numpy as np

from src.cfr import VanillaCFR
from src.external_sampling_mccfr import ExternalSamplingMCCFR
from src.kuhn_poker import KuhnPoker
from src.outcome_sampling_mccfr import OutcomeSamplingMCCFR
from src.util import collect_information_sets, expected_value, exploitability, uniform_strategy


def test_uniform_strategy_has_finite_exploitability():
    efg = KuhnPoker.efg()
    actions, _ = collect_information_sets(efg)
    strategy = uniform_strategy(actions)

    value = expected_value(efg, strategy)
    expl = exploitability(efg, strategy)

    assert math.isfinite(value)
    assert math.isfinite(expl)
    assert expl > 0


def test_vanilla_cfr_converges_on_kuhn_poker():
    efg = KuhnPoker.efg()
    trainer = VanillaCFR(efg)

    trainer.train(10_000)
    strategy = trainer.average_strategy()

    assert abs(expected_value(efg, strategy) - (-1.0 / 18.0)) < 0.01
    assert exploitability(efg, strategy) < 0.01


def test_external_sampling_mccfr_reduces_exploitability_and_visits_fewer_nodes():
    efg = KuhnPoker.efg()
    actions, _ = collect_information_sets(efg)
    uniform_expl = exploitability(efg, uniform_strategy(actions))

    exploitabilities = []
    node_visits = []
    for seed in (0, 1, 2):
        trainer = ExternalSamplingMCCFR(efg, seed=seed)
        trainer.train(100_000)
        exploitabilities.append(exploitability(efg, trainer.average_strategy()))
        node_visits.append(trainer.node_visits / trainer.iterations)

    full_cfr = VanillaCFR(efg)
    full_cfr.train(100)
    full_nodes_per_iteration = full_cfr.node_visits / full_cfr.iterations

    assert float(np.mean(exploitabilities)) < 0.05
    assert float(np.mean(exploitabilities)) < uniform_expl
    assert max(node_visits) < full_nodes_per_iteration


def test_outcome_sampling_mccfr_converges_on_kuhn_poker():
    efg = KuhnPoker.efg()
    trainer = OutcomeSamplingMCCFR(efg, seed=0, epsilon=0.6)

    trainer.train(100_000)
    strategy = trainer.average_strategy()

    assert abs(expected_value(efg, strategy) - (-1.0 / 18.0)) < 0.02
    assert exploitability(efg, strategy) < 0.02
