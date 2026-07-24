"""Regression test: a queued premove drawn on the board is not a position.

Live chess.com game, 2026-07-25. We played Nd7xe5 and queued Bg7xe5, expecting
White to recapture on e5. White instead played exf7+, so the premove was never
legal - but chess.com had already *drawn* it, with our bishop standing on our
own knight's square:

    scraped   rnb1k2r/ppp1pp1p/4P1p1/4b3/2B5/5N2/PPP2PPP/RNBK3R
              (bishop on e5, g7 empty, knight gone)

That board never existed on the server and could not exist. It linked to
nothing, so the client wiped its fen history, adopted the phantom as truth,
and asked the engine to move from it. The engine returned Be5-g7 - correct for
the phantom, illegal on the real board - and the move-registration check had
to catch it twice before the next scan resynced.

The existing confirmation re-capture cannot help here: the drawing is stable
for as long as the premove sits there, so it reproduces perfectly and looks
like a settled position rather than a transient misread. The client instead
records what the site will be drawing when it queues a premove, and discards a
frame that matches it.
"""
import unittest

import chess

from common.utils import premove_render_placement

# The live incident, exactly as logged.
FEN_BEFORE_OUR_MOVE = "rnb1k2r/pppnppbp/4P1p1/8/2B5/5N2/PPP2PPP/RNBK3R b kq - 0 10"
OUR_MOVE = "d7e5"          # Nd7xe5
OUR_PREMOVE = "g7e5"       # Bg7xe5, expecting a recapture that never came
SCRAPED_PHANTOM = "rnb1k2r/ppp1pp1p/4P1p1/4b3/2B5/5N2/PPP2PPP/RNBK3R"
REAL_BOARD_AFTER = "rnb1k2r/ppp1pPbp/6p1/4n3/2B5/5N2/PPP2PPP/RNBK3R"


class PremoveRenderPlacementTest(unittest.TestCase):

    def test_reproduces_the_live_phantom_board(self):
        self.assertEqual(
            premove_render_placement(FEN_BEFORE_OUR_MOVE, OUR_MOVE, OUR_PREMOVE),
            SCRAPED_PHANTOM)

    def test_phantom_is_not_the_real_position(self):
        """The whole point: the drawn board differs from what the server has."""
        self.assertNotEqual(SCRAPED_PHANTOM, REAL_BOARD_AFTER)

    def test_phantom_would_not_link_from_the_previous_position(self):
        """Which is why it reached the history-wiping branch at all."""
        board = chess.Board(FEN_BEFORE_OUR_MOVE)
        reachable = set()
        for first in board.legal_moves:
            board.push(first)
            reachable.add(board.board_fen())
            for second in board.legal_moves:
                board.push(second)
                reachable.add(board.board_fen())
                board.pop()
            board.pop()
        self.assertNotIn(SCRAPED_PHANTOM, reachable)

    def test_renders_a_premove_that_captures_our_own_piece(self):
        """
        The premove is applied as a drawing, not a chess move - python-chess
        would refuse this outright, but it is what the site puts on screen.
        """
        placement = premove_render_placement(FEN_BEFORE_OUR_MOVE, OUR_MOVE, OUR_PREMOVE)
        board = chess.Board(placement + " b - - 0 1")
        self.assertEqual(board.piece_at(chess.E5), chess.Piece(chess.BISHOP, chess.BLACK))
        self.assertIsNone(board.piece_at(chess.G7))

    def test_ordinary_premove_renders_as_the_resulting_board(self):
        """A premove that is simply legal draws the position it would create."""
        start = chess.STARTING_FEN
        placement = premove_render_placement(start, "e2e4", "e7e5")
        expected = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        self.assertEqual(placement, expected.board_fen())

    def test_returns_none_without_a_premove(self):
        self.assertIsNone(premove_render_placement(chess.STARTING_FEN, "e2e4", None))
        self.assertIsNone(premove_render_placement(chess.STARTING_FEN, "e2e4", ""))

    def test_returns_none_when_the_from_square_is_empty(self):
        """Nothing to draw, so nothing to guard against."""
        self.assertIsNone(premove_render_placement(chess.STARTING_FEN, "e2e4", "e5e6"))

    def test_malformed_input_does_not_raise(self):
        """This runs inside the scan loop; it must never break a scan."""
        self.assertIsNone(premove_render_placement(chess.STARTING_FEN, "e2e4", "zz99"))
        self.assertIsNone(premove_render_placement("not a fen", "e2e4", "e7e5"))

    def test_predicts_nothing_when_our_own_move_was_not_legal(self):
        """
        A wrong prediction is worse than none: it could match a genuine scan
        and discard it. Our move must be a move that was really available.
        """
        self.assertIsNone(
            premove_render_placement(chess.STARTING_FEN, "a1a8", "e7e5"))
        # right move, wrong side to move
        self.assertIsNone(
            premove_render_placement(chess.STARTING_FEN, "e7e5", "e2e4"))


if __name__ == "__main__":
    unittest.main()
