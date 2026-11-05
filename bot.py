"""Team owl: equity-driven bot.

Each decision estimates our win probability against one random opponent hand
by Monte Carlo, then uses pot odds (plus a safety margin, because real
opponents who bet are stronger than random) to bet, call or fold.
"""

from __future__ import annotations

import random
import time

from player_sdk.bot_interface import PokerBot
from player_sdk.models import Action, GameState

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]
RANK_OF = {r: i + 2 for i, r in enumerate(RANKS)}


def _straight_high(rank_set):
    """Highest card of a straight within a set of ranks, or 0."""
    if 14 in rank_set:
        rank_set = rank_set | {1}  # the ace also plays low
    for high in range(14, 4, -1):
        if all(r in rank_set for r in range(high - 4, high + 1)):
            return high
    return 0


def evaluate(cards):
    """Score 5-7 cards; a larger tuple is a better hand."""
    ranks = [RANK_OF[c[0]] for c in cards]
    by_suit = {}
    for c in cards:
        by_suit.setdefault(c[1].lower(), []).append(RANK_OF[c[0]])

    flush_ranks = None
    for suited in by_suit.values():
        if len(suited) >= 5:
            flush_ranks = sorted(suited, reverse=True)
    if flush_ranks:
        sf = _straight_high(set(flush_ranks))
        if sf:
            return (8, sf)

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # groups sorted by (count, rank) descending
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_n, top_r = groups[0][1], groups[0][0]

    if top_n == 4:
        kick = max(r for r in counts if r != top_r)
        return (7, top_r, kick)
    if top_n == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return (6, top_r, groups[1][0])
    if flush_ranks:
        return (5,) + tuple(flush_ranks[:5])
    st = _straight_high(set(ranks))
    if st:
        return (4, st)
    if top_n == 3:
        kicks = sorted((r for r in counts if r != top_r), reverse=True)[:2]
        return (3, top_r) + tuple(kicks)
    if top_n == 2:
        pairs = [r for r, n in groups if n == 2]
        if len(pairs) >= 2:
            p1, p2 = pairs[0], pairs[1]
            kick = max(r for r in counts if r != p1 and r != p2)
            return (2, p1, p2, kick)
        kicks = sorted((r for r in counts if r != top_r), reverse=True)[:3]
        return (1, top_r) + tuple(kicks)
    return (0,) + tuple(sorted(ranks, reverse=True)[:5])


def equity(hole, board, samples, deadline, k=1):
    """Share of pots we win (ties count half) against a random hand."""
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in known]
    need_board = 5 - len(board)
    score = 0.0
    done = 0
    for _ in range(samples):
        draw = random.sample(deck, 2 + need_board)
        opp = draw[:2]
        if k > 1 and board:
            # a bettor's range is stronger: keep the best of k random hands
            used = set(draw[2:])
            best = evaluate(opp + list(board))
            for _ in range(k - 1):
                pool = [c for c in deck if c not in used]
                cand = random.sample(pool, 2)
                sc = evaluate(cand + list(board))
                if sc > best:
                    best, opp = sc, cand
        full = list(board) + draw[2:]
        mine = evaluate(list(hole) + full)
        theirs = evaluate(opp + full)
        if mine > theirs:
            score += 1.0
        elif mine == theirs:
            score += 0.5
        done += 1
        if time.perf_counter() > deadline and done >= 20:
            break
    return score / done


class OwlBot(PokerBot):
    SAMPLES = 90

    def act(self, game_state: GameState) -> Action:
        start = time.perf_counter()
        hole = [c[0].upper() + c[1].lower() for c in game_state.hole_cards]
        board = [c[0].upper() + c[1].lower() for c in game_state.community_cards]
        to_call = game_state.call_amount
        pot = game_state.pot_size
        k = 1
        if board and to_call > 0:
            k = 2 if to_call / (pot + to_call) < 0.3 else 3
        eq = equity(hole, board, self.SAMPLES, start + 0.35, k)
        can_raise = game_state.can_raise or game_state.can_bet
        preflop = not board

        if to_call == 0:
            # Nobody has bet: bet for value, occasionally bluff a weak spot.
            bar = 0.60 if preflop else 0.64
            if can_raise and eq >= bar:
                frac = 1.0 if eq > 0.85 else (0.7 if eq > 0.75 else 0.5)
                return self._raise(game_state, frac)
            if can_raise and eq < 0.35 and not preflop and random.random() < 0.08:
                return self._raise(game_state, 0.5)
            return Action.call()

        # Facing a bet: price, plus a margin that grows with the bet size.
        price = to_call / (pot + to_call)
        margin = 0.03 + 0.07 * min(1.0, price * 2) if k > 1 else 0.05 + 0.15 * min(1.0, price * 2)
        if preflop and to_call <= (game_state.big_blind or 20):
            margin = 0.02  # blind-level price: defend widely

        if can_raise and eq >= 0.85 and (preflop or random.random() < 0.8):
            return self._raise(game_state, 0.8)
        if eq >= price + margin:
            return Action.call()
        return Action.fold()

    @staticmethod
    def _raise(game_state: GameState, frac: float) -> Action:
        low = game_state.min_raise_to or 1
        high = game_state.max_raise_to or low
        mine = game_state.my_bet or 0
        if not game_state.community_cards:
            bb = game_state.big_blind or 20
            target = max(3 * bb, mine + int(game_state.pot_size * frac) + game_state.call_amount)
        else:
            target = mine + game_state.call_amount + int(
                (game_state.pot_size + game_state.call_amount) * frac
            )
        target = max(min(target, high), min(low, high))
        return Action.raise_to(target)
