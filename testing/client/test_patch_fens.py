"""Tests for common.utils.patch_fens, the scan-to-scan move linker.

Regression for the 2026-07-25 chess.com sessions: 24 link failures over 19
games, roughly 1.3 per game, every one on a board that was perfectly real.
Each failure made clients/mp_original.py wipe its whole fen and last_moves
history and start again from a single position.

The cause was the candidate filter, which only tried moves leaving a square
whose *occupancy* changed. Our own premoves systematically defeat it: after
f1g2 the castling rook lands back on f1, so f1 never reads as vacated and
the true first move was never generated at all. Seven of the 24 were exactly
that shape, and in the worst case the search produced no candidate moves.

Dropping the filter fixes those, but a plain full-width depth-3 search took
up to 6.4 seconds on the real positions - unaffordable in a 1+0 game inside
a ~14 scans/second loop. So the search is pruned on the placement diff (one
ply closes at most four squares) and deepened one ply at a time, which also
makes the shortest link win rather than the first one stumbled upon.

The fen pairs below are the recorded prev_fen/now_fen of real failures,
taken from logs/sessions/2026-07-25_*/errors/linking_move_error_*.
"""
import time
import unittest

import chess

from common.utils import patch_fens


def _position(*moves, start=chess.STARTING_FEN):
    board = chess.Board(start)
    for uci in moves:
        board.push_uci(uci)
    return board.fen()


def _link(*moves, start=chess.STARTING_FEN):
    """(fen_before, fen_after) for a sequence played from `start`."""
    return start, _position(*moves, start=start)


class TestPatchFensPremoveSequences(unittest.TestCase):
    """The live failures: three plies where a vacated square is refilled."""

    def test_castling_refills_the_bishops_square(self):
        # 12:09:04, and six more of the same shape. We play Bg2, the
        # opponent replies, our queued O-O puts a rook back on f1 - so f1
        # never reads as vacated and f1g2 was never tried.
        before = "rnbqkbnr/pp1p1ppp/4p3/2p5/8/5NP1/PPPPPP1P/RNBQKB1R w KQkq - 0 3"
        after = "r1bqkbnr/pp1p1ppp/2n1p3/2p5/8/5NP1/PPPPPPBP/RNBQ1RK1 b kq - 0 3"
        moves, fens = patch_fens(before, after)
        self.assertEqual(moves, ["f1g2", "b8c6", "e1g1"])
        self.assertEqual(chess.Board(fens[-1]).board_fen(),
                         chess.Board(after).board_fen())
        self.assertEqual(chess.Board(fens[-1]).turn, chess.BLACK)

    def test_fianchetto_refills_the_pawns_square(self):
        # 11:42:52, 11:57:43, 12:18:36 - the black-side mirror: the g7 pawn
        # steps up and our premoved bishop takes its place.
        before = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        after = "rnbqk1nr/ppppppbp/6p1/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2"
        moves, _ = patch_fens(before, after)
        self.assertEqual(moves, ["g7g6", "d2d4", "f8g7"])

    def test_king_returns_to_the_square_it_left(self):
        # 11:36:10: our king steps out and the premove steps it straight
        # back, so neither of our two moves shows up in the diff at all -
        # only the opponent's reply does. The transit square is a guess (we
        # actually played g2f3), the opponent's move and the final position
        # are not.
        before = "6k1/3r1pbp/6p1/1P4P1/7P/8/R5K1/8 w - - 1 38"
        after = "5bk1/3r1p1p/6p1/1P4P1/7P/8/R5K1/8 b - - 0 38"
        res = patch_fens(before, after)
        self.assertIsNotNone(res)
        moves, fens = res
        self.assertEqual(len(moves), 3)
        self.assertEqual(moves[1], "g7f8")
        self.assertEqual(chess.Board(fens[-1]).board_fen(),
                         chess.Board(after).board_fen())
        self.assertEqual(chess.Board(fens[-1]).turn, chess.BLACK)

    def test_recapture_refills_the_captured_square(self):
        # 12:01:17, the slowest of the 24: Qxd1, Nxd1, Bxc3 - d1 is emptied
        # and refilled, c3 changes owner rather than occupancy.
        before = "rnb1k1nr/ppp1ppbp/6p1/8/3qP3/2N2N2/PPP3PP/R1BQKB1R b KQkq - 1 6"
        after = "rnb1k1nr/ppp1pp1p/6p1/8/4P3/2b2N2/PPP3PP/R1BNKB1R w KQkq - 0 7"
        moves, _ = patch_fens(before, after)
        self.assertEqual(moves, ["d4d1", "c3d1", "g7c3"])


class TestPatchFensPrefersShortestLink(unittest.TestCase):
    """Callers rank candidates on ply count, so the link must be minimal."""

    def test_single_ply_not_dressed_up_as_three(self):
        moves, _ = patch_fens(*_link("e2e4"))
        self.assertEqual(moves, ["e2e4"])

    def test_knight_move_is_one_ply_not_a_shuffle(self):
        # Nf3 is also reachable as Ng1-h3 ... h3-f3 style transit lines with
        # an opponent move wedged in; depth-first search can return one of
        # those, iterative deepening cannot.
        moves, _ = patch_fens(*_link("g1f3"))
        self.assertEqual(moves, ["g1f3"])

    def test_identical_positions_link_with_no_moves(self):
        fen = "r1bqkbnr/ppp2p1p/2np2p1/4p3/2P5/2NP2P1/PP2PPBP/R1BQK1NR b KQkq - 0 5"
        moves, fens = patch_fens(fen, fen)
        self.assertEqual(moves, [])
        self.assertEqual(fens, [fen])

    def test_same_placement_other_side_to_move_is_not_a_zero_ply_link(self):
        """Turn is part of the terminating test - the unreadable-turn
        fallback ranks the two turn hypotheses on exactly that."""
        white = "r1bqkbnr/ppp2p1p/2np2p1/4p3/2P5/2NP2P1/PP2PPBP/R1BQK1NR w KQkq - 0 6"
        flipped = chess.Board(white)
        flipped.turn = chess.BLACK
        self.assertIsNone(patch_fens(white, flipped.fen()))


class TestPatchFensLimits(unittest.TestCase):

    def test_castling_alone_is_not_pruned(self):
        """One ply changes four squares; the prune must allow exactly that."""
        opening = _position("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5")
        moves, _ = patch_fens(*_link("e1g1", start=opening))
        self.assertEqual(moves, ["e1g1"])

    def test_en_passant_alone_is_not_pruned(self):
        """Three squares change - the captured pawn is not on the to-square."""
        opening = _position("e2e4", "a7a6", "e4e5", "d7d5")
        moves, _ = patch_fens(*_link("e5d6", start=opening))
        self.assertEqual(moves, ["e5d6"])

    def test_promotion_is_linked(self):
        opening = "8/1P6/8/8/8/8/6k1/4K3 w - - 0 1"
        moves, _ = patch_fens(*_link("b7b8q", start=opening))
        self.assertEqual(moves, ["b7b8q"])

    def test_depth_limit_is_respected(self):
        before, after = _link("e2e4", "e7e5", "g1f3", "b8c6")
        self.assertIsNone(patch_fens(before, after, depth_lim=3))
        self.assertEqual(len(patch_fens(before, after, depth_lim=4)[0]), 4)

    def test_impossible_position_is_rejected_rather_than_invented(self):
        """11:41:45: a pawn vanished from a scan with no capture that could
        explain it. A misread must stay unlinkable."""
        before = "8/4P3/kP6/5p1p/p4P1P/P7/1K6/8 b - - 0 56"
        after = "8/4P3/kP6/7p/p4P1P/P7/1K6/8 b - - 0 56"
        self.assertIsNone(patch_fens(before, after))

    def test_worst_real_case_stays_inside_the_scan_budget(self):
        """The unpruned full-width search took 6.4s on this pair, which on
        its own is a tenth of a 1+0 game."""
        before = "rnb1k1nr/ppp1ppbp/6p1/8/3qP3/2N2N2/PPP3PP/R1BQKB1R b KQkq - 1 6"
        after = "rnb1k1nr/ppp1pp1p/6p1/8/4P3/2b2N2/PPP3PP/R1BNKB1R w KQkq - 0 7"
        start = time.time()
        res = patch_fens(before, after)
        elapsed = time.time() - start
        self.assertIsNotNone(res)
        self.assertLess(elapsed, 0.25)


if __name__ == "__main__":
    unittest.main()
