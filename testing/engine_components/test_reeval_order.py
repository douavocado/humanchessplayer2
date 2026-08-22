"""Guards the re-evaluation ordering knob (REEVAL_ORDER).

Why this exists rather than leaning on the parity harness: the golden-master
scenarios do reach this code path, but degenerately. `promotion_stop_incident`
runs with R=0 (nothing is re-evaluated, so all three orderings coincide) and
`resign_candidate` with R=1; `midgame_reference` has R >= N so the draw covers
everything. Parity therefore stayed green when the default flipped from
"random" to "human" -- which is evidence that the harness does not exercise the
lottery, NOT evidence that the change is inert. It is not: two 150-game sim
arms differ by ~+0.013 t1.

What the draw does: when there is not time to re-evaluate every root move, the
ones that miss out stay at depth_considered 0 and take a ~60cp penalty
(DEPTH_PENALTY x2 + ZERO_DEPTH_PENALTY). In a quiet position that dwarfs the
real eval spread between candidates, so missing the draw is effectively
disqualification. Measured on 115 real positions: quiet positions pick 8.46
root moves and re-evaluate 6.18, with the draw active in 63.2% of them.

So these tests pin the property the sweep actually relied on -- that the knob
selects genuinely different candidate sets -- so it cannot silently decay into
a no-op.
"""
import re
import unittest

import chess

from common.search_constants import REEVAL_ORDER, REEVAL_ORDERS
from engine import Engine
from engine_components import ponderer

# Quiet middlegame positions with enough legal moves that the re-evaluation
# budget cannot cover them all, which is the regime where ordering matters.
POSITIONS = [
    "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R2QKB1R w KQ - 0 9",
    "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 10",
    "rn1q1rk1/pbp1bppp/1p2pn2/3p4/2PP4/1PN1PN2/PB3PPP/R2QKB1R w KQ - 0 9",
]

_REEVAL_LINE = re.compile(r"Re-evaluating moves: \[(.*?)\](?: \(\w+ order\))? with depth")
_RESULTS_LINE = re.compile(r"Re-evaluated evals with depth considered statistics are: \{(.*?)\} \n")
_QUOTED = re.compile(r"'([^']+)'")


def _info(fen):
    return {"side": chess.Board(fen).turn, "fens": [fen], "last_moves": [],
            "self_clock_times": [45.0], "opp_clock_times": [45.0],
            "self_initial_time": 60.0, "opp_initial_time": 60.0}


def _reevaluated(engine, fen):
    """The set of moves this engine chose to re-evaluate, from its own log."""
    engine.update_info(_info(fen))
    engine.log = ""
    engine.make_move(log=False, seed=4242)
    m = _REEVAL_LINE.search(engine.log)
    return m.group(1) if m else None


def _shortlist_and_executed(engine, fen):
    """The candidate order the engine logged, and the order it actually
    searched them in.

    The results dict is written in loop order and logged before the
    None-filtering, so its key order is the executed order - including the
    moves the budget never reached, which land there as [None, 0]."""
    engine.update_info(_info(fen))
    engine.log = ""
    engine.make_move(log=False, seed=4242)
    shortlist = _REEVAL_LINE.search(engine.log)
    results = _RESULTS_LINE.search(engine.log)
    if shortlist is None or results is None:
        return None, None
    return _QUOTED.findall(shortlist.group(1)), _QUOTED.findall(results.group(1))


class TestReevalOrder(unittest.TestCase):

    def test_default_is_human(self):
        """The shipped default. "human" was chosen because uniformly random
        sampling of which candidates to calculate is not a defensible model of
        human thought -- a player calculates the moves that look plausible."""
        self.assertEqual(REEVAL_ORDER, "human")

    def test_invalid_order_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Engine(log_file=None, reeval_order="sideways")

    def test_orderings_select_different_candidates(self):
        """The knob must actually bite. If every ordering picked the same set
        the sweep that justified the default flip measured nothing."""
        seen = {}
        engines = {}
        try:
            for order in REEVAL_ORDERS:
                engines[order] = Engine(log_file=None, reeval_order=order)
                seen[order] = [_reevaluated(engines[order], fen)
                               for fen in POSITIONS]
        finally:
            for e in engines.values():
                e.close_engines()

        covered = [i for i in range(len(POSITIONS))
                   if all(seen[o][i] is not None for o in REEVAL_ORDERS)]
        self.assertTrue(covered,
                        "no test position reached the human-move path; the "
                        "fixtures no longer exercise the draw")
        for a, b in (("random", "human"), ("random", "eval"), ("human", "eval")):
            self.assertTrue(
                any(seen[a][i] != seen[b][i] for i in covered),
                f"{a!r} and {b!r} chose identical candidates on every position "
                f"-- the ordering knob has become a no-op")


class TestReevalSequence(unittest.TestCase):
    """reeval_sequence is the last thing standing between the ordering the
    knob computes and the order the budget is actually spent in.

    It used to be an unconditional random.shuffle, so the ordering never
    survived the handoff: reeval_order chose the shortlist and a uniform
    draw chose who on it got calculated. Live consequence, 2026-08-22: Nxa1
    at p=0.888 was shortlisted first, shuffled last, never reached, and lost
    the comparison on the depth-0 penalty (~1.1 pawn).
    """

    class _FakeEngine:
        def __init__(self, order):
            self.reeval_order = order

    def test_human_and_eval_orderings_are_preserved(self):
        moves = ["c2a1", "c2d4", "c2b4", "d8b6"]
        for order in ("human", "eval"):
            self.assertEqual(
                ponderer.reeval_sequence(self._FakeEngine(order), moves), moves,
                f"{order!r} ordering did not survive into the search order")

    def test_random_still_draws(self):
        """"random" is the setting that asks for the lottery, so it keeps it."""
        moves = [f"a{i}a{i}" for i in range(1, 9)]
        engine = self._FakeEngine("random")
        draws = {tuple(ponderer.reeval_sequence(engine, moves)) for _ in range(50)}
        self.assertGreater(len(draws), 1, "'random' ordering stopped shuffling")
        for draw in draws:
            self.assertEqual(sorted(draw), sorted(moves))

    def test_callers_list_is_not_mutated(self):
        """The main call site logs the shortlist before handing it over, so an
        in-place shuffle would make that log line describe an order that was
        never searched. That is what hid the bug in the live logs."""
        moves = [f"a{i}a{i}" for i in range(1, 9)]
        original = list(moves)
        for order in REEVAL_ORDERS:
            ponderer.reeval_sequence(self._FakeEngine(order), moves)
            self.assertEqual(moves, original)

    def test_missing_attribute_falls_back_to_the_module_default(self):
        class _Bare:
            pass
        moves = ["c2a1", "c2d4"]
        result = ponderer.reeval_sequence(_Bare(), moves)
        self.assertEqual(sorted(result), sorted(moves))


class TestOrderReachesTheSearch(unittest.TestCase):
    """End-to-end counterpart to TestReevalSequence: the order the engine
    logs must be the order it spends the budget in."""

    def test_logged_order_is_the_executed_order(self):
        checked = 0
        for order in ("human", "eval"):
            engine = Engine(log_file=None, reeval_order=order)
            try:
                for fen in POSITIONS:
                    shortlist, executed = _shortlist_and_executed(engine, fen)
                    if shortlist is None or len(shortlist) < 2:
                        continue
                    self.assertEqual(
                        shortlist, executed,
                        f"{order!r}: logged {shortlist} but searched {executed}")
                    checked += 1
            finally:
                engine.close_engines()
        self.assertTrue(checked, "no position exercised the re-evaluation "
                                 "loop with more than one candidate")


if __name__ == "__main__":
    unittest.main()
