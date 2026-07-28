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

# Quiet middlegame positions with enough legal moves that the re-evaluation
# budget cannot cover them all, which is the regime where ordering matters.
POSITIONS = [
    "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R2QKB1R w KQ - 0 9",
    "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 10",
    "rn1q1rk1/pbp1bppp/1p2pn2/3p4/2PP4/1PN1PN2/PB3PPP/R2QKB1R w KQ - 0 9",
]

_REEVAL_LINE = re.compile(r"Re-evaluating moves: \[(.*?)\] with depth")


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


if __name__ == "__main__":
    unittest.main()
