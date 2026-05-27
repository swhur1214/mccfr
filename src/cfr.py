import numpy as np

try:
    from .regret_matching import regret_matching
    from .util import collect_information_sets, strategy_from_arrays
except ImportError:
    from regret_matching import regret_matching
    from util import collect_information_sets, strategy_from_arrays


class VanillaCFR:
    """Full-tree Counterfactual Regret Minimization baseline."""

    def __init__(self, efg: dict):
        self._efg = efg
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
                self._cfr("", update_player=player, reach=np.ones(2), chance_reach=1.0)
            self.iterations += 1

    def _cfr(
        self,
        node_id: str,
        update_player: int,
        reach: np.ndarray,
        chance_reach: float,
    ) -> float:
        self.node_visits += 1
        node = self._efg[node_id]
        node_type = node["type"]

        if node_type == "TERMINAL":
            return float(node["utility"][update_player])

        if node_type == "CHANCE":
            return sum(
                prob
                * self._cfr(child, update_player, reach, chance_reach * prob)
                for _, child, prob in node["outcomes"]
            )

        player = node["player"]
        info_set = node["information_set"]
        actions = [action for action, _ in node["actions"]]
        strategy = regret_matching(self._regret[info_set])

        action_values = np.zeros(len(actions))
        node_value = 0.0
        for idx, (_, child) in enumerate(node["actions"]):
            next_reach = reach.copy()
            next_reach[player] *= strategy[idx]
            action_values[idx] = self._cfr(
                child, update_player, next_reach, chance_reach
            )
            node_value += strategy[idx] * action_values[idx]

        if player == update_player:
            opponent = 1 - player
            counterfactual_reach = chance_reach * reach[opponent]
            self._regret[info_set] += counterfactual_reach * (action_values - node_value)
            self._strategy_sum[info_set] += reach[player] * strategy

        return float(node_value)
