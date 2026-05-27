from collections import deque

import numpy as np

try:
    from .kuhn_poker import KuhnPoker
except ImportError:
    from kuhn_poker import KuhnPoker


def collect_information_sets(efg: dict) -> tuple[dict, dict]:
    """Return action and owner maps keyed by information set."""
    actions = {}
    players = {}
    for node in efg.values():
        if node["type"] != "DECISION":
            continue
        info_set = node["information_set"]
        node_actions = [action for action, _ in node["actions"]]
        if info_set in actions and actions[info_set] != node_actions:
            raise ValueError(f"inconsistent actions for information set {info_set}")
        actions[info_set] = node_actions
        players[info_set] = node["player"]
    return actions, players


def uniform_strategy(actions_by_info_set: dict) -> dict:
    """Return a uniform behavioral strategy for every information set."""
    return {
        info_set: {action: 1.0 / len(actions) for action in actions}
        for info_set, actions in actions_by_info_set.items()
    }


def _strategy_prob(strategy: dict, info_set: str, action: str, actions: list[str]) -> float:
    if info_set not in strategy:
        return 1.0 / len(actions)
    return float(strategy[info_set].get(action, 0.0))


def _expected_value_recursive(efg: dict, strategy: dict, node_id: str = "") -> float:
    node = efg[node_id]
    node_type = node["type"]

    if node_type == "CHANCE":
        return sum(
            prob * _expected_value_recursive(efg, strategy, child)
            for _, child, prob in node["outcomes"]
        )

    if node_type == "DECISION":
        info_set = node["information_set"]
        actions = [action for action, _ in node["actions"]]
        return sum(
            _strategy_prob(strategy, info_set, action, actions)
            * _expected_value_recursive(efg, strategy, child)
            for action, child in node["actions"]
        )

    return float(node["utility"][0])


def expected_value(efg: dict, strategy: dict, player: int = 0) -> float:
    """Return the expected payoff of a behavioral strategy profile."""
    value0 = _expected_value_recursive(efg, strategy)
    return value0 if player == 0 else -value0


def behavior_to_sequence(strategy: dict, tfsdp: dict) -> dict:
    """Convert a behavioral strategy into sequence-form realization weights."""
    x = {sigma: 0.0 for sigma in tfsdp["Sigma"]}

    for info_set in tfsdp["J"]:
        actions = tfsdp["A"][info_set]
        parent = tfsdp["p"][info_set]
        parent_reach = 1.0 if parent is None else x[parent]
        for action in actions:
            prob = _strategy_prob(strategy, info_set, action, actions)
            x[(info_set, action)] = parent_reach * prob

    return x


def _get_nodes_top_down(tfsdp: dict) -> list:
    nodes = [*tfsdp["K"], *tfsdp["J"]]
    indegree = {node: 0 for node in nodes}
    children = {node: [] for node in nodes}

    for obs_point in tfsdp["K"]:
        for signal in tfsdp["S"][obs_point]:
            child = tfsdp["rho"][(obs_point, signal)]
            if child in indegree:
                indegree[child] += 1
                children[obs_point].append(child)

    for info_set in tfsdp["J"]:
        for action in tfsdp["A"][info_set]:
            child = tfsdp["rho"][(info_set, action)]
            if child in indegree:
                indegree[child] += 1
                children[info_set].append(child)

    queue = deque(node for node in nodes if indegree[node] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return order


def _traverse_tree_for_linear_utility(
    efg: dict,
    node_id: str,
    player: int,
    opponent_x: dict,
    l: dict,
    seq_player=None,
    seq_opp=None,
    chance_prob: float = 1.0,
) -> None:
    node = efg[node_id]
    node_type = node["type"]

    if node_type == "CHANCE":
        for _, child, prob in node["outcomes"]:
            _traverse_tree_for_linear_utility(
                efg, child, player, opponent_x, l, seq_player, seq_opp, chance_prob * prob
            )
        return

    if node_type == "DECISION":
        info_set = node["information_set"]
        acting_player = node["player"]
        for action, child in node["actions"]:
            next_seq = (info_set, action)
            if acting_player == player:
                _traverse_tree_for_linear_utility(
                    efg, child, player, opponent_x, l, next_seq, seq_opp, chance_prob
                )
            else:
                _traverse_tree_for_linear_utility(
                    efg, child, player, opponent_x, l, seq_player, next_seq, chance_prob
                )
        return

    if seq_player is not None:
        opp_reach = 1.0 if seq_opp is None else float(opponent_x[seq_opp])
        l[seq_player] += chance_prob * opp_reach * float(node["utility"][player])


def linear_utility(efg: dict, tfsdp: dict, player: int, opponent_x: dict) -> dict:
    l = {sigma: 0.0 for sigma in tfsdp["Sigma"]}
    _traverse_tree_for_linear_utility(efg, "", player, opponent_x, l)
    return l


def best_response_value(efg: dict, tfsdp: dict, player: int, opponent_x: dict) -> float:
    """Return the value of the player's best response to opponent_x."""
    l = linear_utility(efg, tfsdp, player, opponent_x)

    decision_points = set(tfsdp["J"])
    observation_points = set(tfsdp["K"])
    nodes = _get_nodes_top_down(tfsdp)

    values = {"T": 0.0}
    for node in reversed(nodes):
        if node in decision_points:
            values[node] = max(
                l[(node, action)] + values[tfsdp["rho"][(node, action)]]
                for action in tfsdp["A"][node]
            )
        elif node in observation_points:
            values[node] = sum(
                values[tfsdp["rho"][(node, signal)]]
                for signal in tfsdp["S"][node]
            )

    return values[""]


def best_response_value_behavior(efg: dict, strategy: dict, player: int) -> float:
    """Return an exact best-response value to a behavioral strategy profile.

    The recursion carries distributions over histories weighted only by chance
    and opponent reach. When the best-response player is to act, histories are
    grouped by information set so that one action is chosen for the whole
    information set, not separately at each hidden state.
    """

    def solve(items: list[tuple[str, float]]) -> float:
        terminal_value = 0.0
        continuation = []
        player_groups = {}

        for node_id, weight in items:
            if weight == 0.0:
                continue

            node = efg[node_id]
            node_type = node["type"]

            if node_type == "TERMINAL":
                terminal_value += weight * float(node["utility"][player])
                continue

            if node_type == "CHANCE":
                for _, child, prob in node["outcomes"]:
                    continuation.append((child, weight * prob))
                continue

            info_set = node["information_set"]
            actions = [action for action, _ in node["actions"]]
            if node["player"] == player:
                player_groups.setdefault(info_set, []).append((node_id, weight))
            else:
                for action, child in node["actions"]:
                    prob = _strategy_prob(strategy, info_set, action, actions)
                    continuation.append((child, weight * prob))

        value = terminal_value
        if continuation:
            value += solve(continuation)

        for group in player_groups.values():
            first_node = efg[group[0][0]]
            actions = [action for action, _ in first_node["actions"]]
            action_to_child = {
                node_id: {action: child for action, child in efg[node_id]["actions"]}
                for node_id, _ in group
            }

            best = -float("inf")
            for action in actions:
                child_items = [
                    (action_to_child[node_id][action], weight)
                    for node_id, weight in group
                ]
                best = max(best, solve(child_items))
            value += best

        return value

    return solve([("", 1.0)])


def exploitability(efg: dict, strategy: dict) -> float:
    """Return exploitability, i.e. NashConv / 2, for a 2p zero-sum EFG."""
    br0 = best_response_value_behavior(efg, strategy, player=0)
    br1 = best_response_value_behavior(efg, strategy, player=1)
    return 0.5 * (br0 + br1)


def format_strategy(strategy: dict, digits: int = 4) -> dict:
    """Return rounded strategy probabilities sorted by information-set name."""
    return {
        info_set: {
            action: round(float(prob), digits)
            for action, prob in action_probs.items()
        }
        for info_set, action_probs in sorted(strategy.items())
    }


def strategy_from_arrays(actions_by_info_set: dict, values: dict) -> dict:
    """Convert information-set arrays into a behavioral strategy dict."""
    strategy = {}
    for info_set, actions in actions_by_info_set.items():
        probs = np.asarray(values[info_set], dtype=float)
        total = probs.sum()
        if total <= 0:
            probs = np.full(len(actions), 1.0 / len(actions))
        else:
            probs = probs / total
        strategy[info_set] = {
            action: float(probs[idx]) for idx, action in enumerate(actions)
        }
    return strategy
