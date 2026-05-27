import numpy as np

try:
    from .regret_matching import regret_matching
    from .util import collect_information_sets, strategy_from_arrays
except ImportError:
    from regret_matching import regret_matching
    from util import collect_information_sets, strategy_from_arrays


class OutcomeSamplingMCCFR:
    """Outcome-sampling MCCFR with epsilon exploration.

    This samples one terminal trajectory per update player. At update-player
    nodes, actions are sampled from q = epsilon * uniform + (1 - epsilon) *
    sigma, and the regret estimate uses the corresponding importance weights.
    """

    def __init__(self, efg: dict, seed: int | None = None, epsilon: float = 0.6):
        self._efg = efg
        self._rng = np.random.default_rng(seed)
        self._epsilon = epsilon
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
                self._episode(
                    "",
                    update_player=player,
                    my_reach=1.0,
                    opponent_reach=1.0,
                    sample_reach=1.0,
                )
            self.iterations += 1

    def _episode(
        self,
        node_id: str,
        update_player: int,
        my_reach: float,
        opponent_reach: float,
        sample_reach: float,
    ) -> float:
        self.node_visits += 1
        node = self._efg[node_id]
        node_type = node["type"]

        if node_type == "TERMINAL":
            return float(node["utility"][update_player])

        if node_type == "CHANCE":
            outcomes = node["outcomes"]
            probs = np.array([prob for _, _, prob in outcomes], dtype=float)
            idx = int(self._rng.choice(len(outcomes), p=probs))
            _, child, prob = outcomes[idx]
            return self._episode(
                child,
                update_player,
                my_reach,
                opponent_reach * prob,
                sample_reach * prob,
            )

        player = node["player"]
        info_set = node["information_set"]
        actions = [action for action, _ in node["actions"]]
        policy = regret_matching(self._regret[info_set])

        if player == update_player:
            uniform = np.full(len(actions), 1.0 / len(actions))
            sample_policy = self._epsilon * uniform + (1.0 - self._epsilon) * policy
        else:
            sample_policy = policy

        sampled_idx = int(self._rng.choice(len(actions), p=sample_policy))
        sampled_prob = float(sample_policy[sampled_idx])
        _, child = node["actions"][sampled_idx]

        if player == update_player:
            next_my_reach = my_reach * float(policy[sampled_idx])
            next_opponent_reach = opponent_reach
        else:
            next_my_reach = my_reach
            next_opponent_reach = opponent_reach * float(policy[sampled_idx])

        child_value = self._episode(
            child,
            update_player,
            next_my_reach,
            next_opponent_reach,
            sample_reach * sampled_prob,
        )

        child_values = np.zeros(len(actions))
        child_values[sampled_idx] = child_value / sampled_prob
        value_estimate = float(policy @ child_values)

        if player == update_player:
            weight = opponent_reach / sample_reach
            self._regret[info_set] += weight * (child_values - value_estimate)
            self._strategy_sum[info_set] += (my_reach / sample_reach) * policy

        return value_estimate
