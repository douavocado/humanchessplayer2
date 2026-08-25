"""Guards PONDER_MIN_ROOT_MOVES: the ponder must never shortlist one move.

ponder() picks our candidate replies for a position it is preparing for by
slicing the altered NN ranking to `[:search_width]`, then hands them to
re_evaluate. At search_width 1 that shortlist holds a single move, so the
argmax is that move by construction and the Stockfish call inside re_evaluate
is asked "which of these one moves is best". The NN's top pick is cached and,
because a recognised position is answered the moment it appears, played
instantly with no check_obvious_move and no re-evaluation.

The fixture is the 2026-08-25 game, move 13. We had just played dxc6+ and
pondered Black's replies; for ...bxc6 the cached answer was Bxc6+, a bishop
for a pawn (+569 -> +233) out of a won position. The raw net had it right
(Bd3 0.2276 ahead of Bxc6+ 0.1505) and the alteration's capture and check
multipliers flipped the order, which is precisely when a second opinion earns
its keep: replayed at the ponder's own budget, root_moves ['Bxc6+'] returns
Bxc6+ and ['Bxc6+', 'Bd3'] returns Bd3.
"""
import unittest

import chess

from common.search_constants import PONDER_MIN_ROOT_MOVES

# White to move, after 12. dxc6+ bxc6. Bxc6+ is the blunder; Bd3/Be2/Bc4/Ba4
# all keep the extra piece.
BXC6_FEN = "r4b1r/p2kpppp/2p2n2/1B6/4P3/8/PP3PPP/R1BNK2R w KQ - 0 13"
BXC6_BLUNDER = "b5c6"


class TestPonderRootFloor(unittest.TestCase):
    def test_floor_is_at_least_two(self):
        # One candidate cannot be a choice, whatever else is tuned.
        self.assertGreaterEqual(PONDER_MIN_ROOT_MOVES, 2)

    def test_floor_only_raises_narrow_widths(self):
        # It is a floor, not an override: a position that already searches
        # wide must keep its width.
        for search_width in (1, 2, 3, 8):
            self.assertEqual(max(PONDER_MIN_ROOT_MOVES, search_width),
                             max(search_width, 2))


class TestBlunderIsRejectedGivenAChoice(unittest.TestCase):
    """The engine half: given two candidates, Stockfish drops the bishop sac.

    Uses the ponder's own per-position budget rather than a deep search, so
    this pins that the floor is sufficient at the time the ponder actually
    has, not merely that a strong engine disagrees.
    """

    @classmethod
    def setUpClass(cls):
        import chess.engine
        from common.constants import PATH_TO_STOCKFISH
        cls.engine = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH)
        cls.board = chess.Board(BXC6_FEN)

    @classmethod
    def tearDownClass(cls):
        cls.engine.quit()

    def _pick(self, ucis):
        info = self.engine.analyse(
            self.board, chess.engine.Limit(time=0.02),
            root_moves=[chess.Move.from_uci(u) for u in ucis])
        return info["pv"][0].uci()

    def test_one_candidate_returns_the_blunder(self):
        # What actually ran. Not a criticism of Stockfish: it was given no
        # alternative, so this is the shape of the bug, not a search failure.
        self.assertEqual(self._pick([BXC6_BLUNDER]), BXC6_BLUNDER)

    def test_two_candidates_reject_the_blunder(self):
        self.assertNotEqual(self._pick([BXC6_BLUNDER, "b5d3"]), BXC6_BLUNDER)

    def test_any_sound_bishop_move_beats_it(self):
        # The floor takes whichever move the NN ranked second, so the fix must
        # not depend on that move being Bd3 specifically.
        for alternative in ("b5d3", "b5e2", "b5c4", "b5a4"):
            self.assertNotEqual(self._pick([BXC6_BLUNDER, alternative]),
                                BXC6_BLUNDER,
                                msg="blunder survived against {}".format(alternative))
