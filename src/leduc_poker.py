class LeducPoker:
    """Fixed-limit Leduc Hold'em with two suits and three ranks."""

    RANKS = ("J", "Q", "K")
    RANK_VALUE = {"J": 0, "Q": 1, "K": 2}
    DECK = tuple(f"{rank}{copy}" for rank in RANKS for copy in range(2))

    ANTE = 1
    BET_SIZES = (2, 4)

    @staticmethod
    def rank(card: str) -> str:
        return card[0]

    @staticmethod
    def efg(max_raises: int = 2) -> dict:
        """Return Leduc Poker as an extensive-form game tree.

        This follows the common benchmark rules: each player antes 1, each
        player receives one private card from a six-card deck, one public card
        is revealed after the first betting round, and the fixed bet sizes are
        2 in round 1 and 4 in round 2. Each round allows at most `max_raises`
        bet/raise actions.
        """
        efg = {"": {"type": "CHANCE", "outcomes": []}}

        def history_label(history: tuple[str, ...]) -> str:
            return ".".join(history) if history else "-"

        def node_id(
            cards: tuple[str, str],
            public: str | None,
            round_index: int,
            to_act: int,
            round_history: tuple[str, ...],
            full_history: tuple[tuple[str, ...], tuple[str, ...]],
        ) -> str:
            public_label = public if public is not None else "-"
            return (
                f"{cards[0]}:{cards[1]}|pub={public_label}|r={round_index}"
                f"|p={to_act}|h0={history_label(full_history[0])}"
                f"|h1={history_label(full_history[1])}"
                f"|cur={history_label(round_history)}"
            )

        def payoff(contrib: tuple[int, int], winner: int | None) -> list[float]:
            if winner is None:
                total = contrib[0] + contrib[1]
                u0 = total / 2.0 - contrib[0]
            elif winner == 0:
                u0 = float(contrib[1])
            else:
                u0 = -float(contrib[0])
            return [u0, -u0]

        def showdown(cards: tuple[str, str], public: str, contrib: tuple[int, int]):
            public_rank = LeducPoker.rank(public)
            ranks = [LeducPoker.rank(cards[0]), LeducPoker.rank(cards[1])]
            pairs = [rank == public_rank for rank in ranks]

            if pairs[0] and not pairs[1]:
                winner = 0
            elif pairs[1] and not pairs[0]:
                winner = 1
            elif LeducPoker.RANK_VALUE[ranks[0]] > LeducPoker.RANK_VALUE[ranks[1]]:
                winner = 0
            elif LeducPoker.RANK_VALUE[ranks[1]] > LeducPoker.RANK_VALUE[ranks[0]]:
                winner = 1
            else:
                winner = None

            return payoff(contrib, winner)

        def terminal_node(label: str, utility: list[float]) -> str:
            terminal_id = f"T|{label}"
            efg[terminal_id] = {"type": "TERMINAL", "utility": utility}
            return terminal_id

        def next_round_or_showdown(
            cards: tuple[str, str],
            public: str | None,
            round_index: int,
            contrib: tuple[int, int],
            full_history: tuple[tuple[str, ...], tuple[str, ...]],
        ) -> str:
            if round_index == 1:
                utility = showdown(cards, public, contrib)
                return terminal_node(
                    f"{cards[0]}:{cards[1]}|pub={public}|showdown"
                    f"|h0={history_label(full_history[0])}|h1={history_label(full_history[1])}",
                    utility,
                )

            chance_id = (
                f"{cards[0]}:{cards[1]}|public-chance"
                f"|h0={history_label(full_history[0])}"
            )
            if chance_id in efg:
                return chance_id

            efg[chance_id] = {"type": "CHANCE", "outcomes": []}
            remaining = [card for card in LeducPoker.DECK if card not in cards]
            for public_card in remaining:
                child = add_decision(
                    cards=cards,
                    public=public_card,
                    round_index=1,
                    to_act=0,
                    round_bets=(0, 0),
                    raises_made=0,
                    round_history=(),
                    full_history=(full_history[0], ()),
                    contrib=contrib,
                )
                efg[chance_id]["outcomes"].append(
                    (public_card, child, 1.0 / len(remaining))
                )
            return chance_id

        def add_decision(
            cards: tuple[str, str],
            public: str | None,
            round_index: int,
            to_act: int,
            round_bets: tuple[int, int],
            raises_made: int,
            round_history: tuple[str, ...],
            full_history: tuple[tuple[str, ...], tuple[str, ...]],
            contrib: tuple[int, int],
        ) -> str:
            state_id = node_id(cards, public, round_index, to_act, round_history, full_history)
            if state_id in efg:
                return state_id

            outstanding = max(round_bets) - round_bets[to_act]
            can_raise = raises_made < max_raises
            if outstanding == 0:
                actions = ["check"]
                if can_raise:
                    actions.append("bet")
            else:
                actions = ["fold", "call"]
                if can_raise:
                    actions.append("raise")

            private_rank = LeducPoker.rank(cards[to_act])
            public_rank = LeducPoker.rank(public) if public is not None else "-"
            info_set = (
                f"P{to_act}|card={private_rank}|pub={public_rank}|r={round_index}"
                f"|h0={history_label(full_history[0])}"
                f"|h1={history_label(full_history[1])}"
                f"|cur={history_label(round_history)}"
            )

            efg[state_id] = {
                "type": "DECISION",
                "player": to_act,
                "information_set": info_set,
                "actions": [],
            }

            for action in actions:
                child = apply_action(
                    cards,
                    public,
                    round_index,
                    to_act,
                    round_bets,
                    raises_made,
                    round_history,
                    full_history,
                    contrib,
                    action,
                )
                efg[state_id]["actions"].append((action, child))

            return state_id

        def apply_action(
            cards: tuple[str, str],
            public: str | None,
            round_index: int,
            to_act: int,
            round_bets: tuple[int, int],
            raises_made: int,
            round_history: tuple[str, ...],
            full_history: tuple[tuple[str, ...], tuple[str, ...]],
            contrib: tuple[int, int],
            action: str,
        ) -> str:
            other = 1 - to_act
            bet_size = LeducPoker.BET_SIZES[round_index]
            next_history = round_history + (action,)
            next_full = list(full_history)
            next_full[round_index] = next_history
            next_full = (next_full[0], next_full[1])

            if action == "fold":
                return terminal_node(
                    f"{cards[0]}:{cards[1]}|pub={public}|fold={to_act}"
                    f"|h0={history_label(next_full[0])}|h1={history_label(next_full[1])}",
                    payoff(contrib, other),
                )

            if action == "check":
                if len(round_history) > 0 and round_history[-1] == "check":
                    return next_round_or_showdown(
                        cards, public, round_index, contrib, next_full
                    )
                return add_decision(
                    cards,
                    public,
                    round_index,
                    other,
                    round_bets,
                    raises_made,
                    next_history,
                    next_full,
                    contrib,
                )

            next_round_bets = list(round_bets)
            next_contrib = list(contrib)

            if action == "bet":
                next_round_bets[to_act] += bet_size
                next_contrib[to_act] += bet_size
                return add_decision(
                    cards,
                    public,
                    round_index,
                    other,
                    tuple(next_round_bets),
                    raises_made + 1,
                    next_history,
                    next_full,
                    tuple(next_contrib),
                )

            if action == "raise":
                target = round_bets[other] + bet_size
                amount = target - round_bets[to_act]
                next_round_bets[to_act] += amount
                next_contrib[to_act] += amount
                return add_decision(
                    cards,
                    public,
                    round_index,
                    other,
                    tuple(next_round_bets),
                    raises_made + 1,
                    next_history,
                    next_full,
                    tuple(next_contrib),
                )

            if action == "call":
                amount = round_bets[other] - round_bets[to_act]
                next_round_bets[to_act] += amount
                next_contrib[to_act] += amount
                return next_round_or_showdown(
                    cards, public, round_index, tuple(next_contrib), next_full
                )

            raise ValueError(f"unknown action {action}")

        for card0 in LeducPoker.DECK:
            for card1 in LeducPoker.DECK:
                if card1 == card0:
                    continue
                child = add_decision(
                    cards=(card0, card1),
                    public=None,
                    round_index=0,
                    to_act=0,
                    round_bets=(0, 0),
                    raises_made=0,
                    round_history=(),
                    full_history=((), ()),
                    contrib=(LeducPoker.ANTE, LeducPoker.ANTE),
                )
                efg[""]["outcomes"].append((f"{card0}:{card1}", child, 1.0 / 30.0))

        return efg
