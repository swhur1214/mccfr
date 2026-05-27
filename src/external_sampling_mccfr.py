import numpy as np

try:
    from .regret_matching import regret_matching
    from .util import collect_information_sets, strategy_from_arrays
except ImportError:
    from regret_matching import regret_matching
    from util import collect_information_sets, strategy_from_arrays


class ExternalSamplingMCCFR:
    """External-sampling MCCFR for two-player zero-sum extensive-form games."""

    def __init__(self, efg: dict, seed: int | None = None):
        self._efg = efg
        self._rng = np.random.default_rng(seed)
        self._actions, self._players = collect_information_sets(efg)
        self._regret = {
            info_set: np.zeros(len(actions))
            for info_set, actions in self._actions.items()
        }
        self._strategy_sum = {
            info_set: np.zeros(len(actions))
            for info_set, actions in self._actions.items()
        }
        self.iterations = 0
        self.node_visits = 0

    def current_strategy(self) -> dict:
        return {
            info_set: {
                action: float(prob)
                for action, prob in zip(actions, regret_matching(self._regret[info_set]))
            }
            for info_set, actions in self._actions.items()
        }

    def average_strategy(self) -> dict:
        return strategy_from_arrays(self._actions, self._strategy_sum)

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            for player in (0, 1):
                self._external_sample("", update_player=player)
            self.iterations += 1

    def _external_sample(self, node_id: str, update_player: int) -> float:
        self.node_visits += 1
        node = self._efg[node_id]
        node_type = node["type"]

        if node_type == "TERMINAL":
            return float(node["utility"][update_player])

        if node_type == "CHANCE":
            outcomes = node["outcomes"]
            probs = np.array([prob for _, _, prob in outcomes], dtype=float)
            idx = int(self._rng.choice(len(outcomes), p=probs))
            _, child, _ = outcomes[idx]
            return self._external_sample(child, update_player)

        player = node["player"]
        info_set = node["information_set"]
        actions = [action for action, _ in node["actions"]]
        strategy = regret_matching(self._regret[info_set])

        if player != update_player:
            self._strategy_sum[info_set] += strategy
            idx = int(self._rng.choice(len(actions), p=strategy))
            _, child = node["actions"][idx]
            return self._external_sample(child, update_player)

        action_values = np.zeros(len(actions))
        node_value = 0.0
        for idx, (_, child) in enumerate(node["actions"]):
            action_values[idx] = self._external_sample(child, update_player)
            node_value += strategy[idx] * action_values[idx]

        self._regret[info_set] += action_values - node_value
        return float(node_value)
