"""Regression test for cross-checking a linked scan against the last-move
highlights.

From the 2026-08-22 game (session 2026-08-22_22-00-44, t=140.2s): black
played 19...Rae8 and the scan caught the rook mid-slide, over b8. a8b8 is a
legal move, so patch_fens linked it happily and the frame was adopted with
no second look - the confirmation re-capture only ran when linking *failed*.
The engine then chose 20.Rxf8+ from a board whose e8 was empty; Bxe8 (worth
~0.7 pawn there) was never in its move list, appearing instead as the quiet
move 'Be8' with probability 0.0003.

The highlights read the same frame independently and marked a8 and e8. A
piece in flight does not carry its highlight with it, which is what makes
them a usable check on a single-frame adoption.
"""
import unittest

import chess

from common.utils import highlight_squares_to_chess, highlights_contradict_move


def img_squares(bottom, *names):
    """Image-space indices (0-63, row-major from the board crop's top-left)
    for the named chess squares, in the given orientation."""
    out = []
    for name in names:
        square = chess.parse_square(name)
        if bottom == "w":
            out.append(chess.square_mirror(square))
        else:
            out.append(63 - chess.square_mirror(square))
    return out


class TestHighlightSquareMapping(unittest.TestCase):

    def test_round_trip_white_at_bottom(self):
        names = ["a8", "e8", "e1", "g1", "d4"]
        squares = {chess.parse_square(n) for n in names}
        self.assertEqual(
            highlight_squares_to_chess(img_squares("w", *names), "w"), squares)

    def test_round_trip_black_at_bottom(self):
        names = ["a8", "e8", "e1", "g1", "d4"]
        squares = {chess.parse_square(n) for n in names}
        self.assertEqual(
            highlight_squares_to_chess(img_squares("b", *names), "b"), squares)

    def test_orientation_actually_differs(self):
        self.assertNotEqual(img_squares("w", "a8"), img_squares("b", "a8"))


class TestHighlightsContradictMove(unittest.TestCase):

    def test_mid_slide_misread_is_caught(self):
        # The live failure: rook caught over b8 during 19...Rae8.
        self.assertTrue(highlights_contradict_move(
            img_squares("w", "a8", "e8"), "a8b8", "w"))

    def test_real_move_agrees(self):
        self.assertFalse(highlights_contradict_move(
            img_squares("w", "a8", "e8"), "a8e8", "w"))

    def test_caught_from_black_side_too(self):
        self.assertTrue(highlights_contradict_move(
            img_squares("b", "a8", "e8"), "a8b8", "b"))
        self.assertFalse(highlights_contradict_move(
            img_squares("b", "a8", "e8"), "a8e8", "b"))

    def test_wrong_orientation_is_not_silently_accepted(self):
        # Reading a white-at-bottom frame as black-at-bottom must not agree.
        self.assertTrue(highlights_contradict_move(
            img_squares("w", "a8", "e8"), "a8e8", "b"))


class TestNoSignalIsNotDisagreement(unittest.TestCase):
    """Only an unambiguous pair counts. Everything else must read as silence,
    or a blank highlight reading would discard real positions."""

    def test_nothing_detected(self):
        self.assertFalse(highlights_contradict_move([], "a8b8", "w"))

    def test_single_square_left_by_a_premove_overlay(self):
        # On Lichess a queued premove paints over one of the pair.
        self.assertFalse(highlights_contradict_move(
            img_squares("w", "a8"), "a8b8", "w"))

    def test_more_than_a_pair(self):
        self.assertFalse(highlights_contradict_move(
            img_squares("w", "a8", "e8", "d4"), "a8b8", "w"))


class TestSpecialMovesHighlightTheirOwnPair(unittest.TestCase):
    """Both sites highlight the moving piece's own from/to squares, so the
    plain uci squares are the right comparison even when other squares
    change occupancy."""

    def test_castling(self):
        board = chess.Board(
            "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        move = board.parse_uci("e1g1")
        self.assertTrue(board.is_castling(move))
        self.assertFalse(highlights_contradict_move(
            img_squares("w", "e1", "g1"), "e1g1", "w"))

    def test_en_passant(self):
        board = chess.Board(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        move = board.parse_uci("e5f6")
        self.assertTrue(board.is_en_passant(move))
        self.assertFalse(highlights_contradict_move(
            img_squares("w", "e5", "f6"), "e5f6", "w"))

    def test_promotion(self):
        self.assertFalse(highlights_contradict_move(
            img_squares("w", "b7", "b8"), "b7b8q", "w"))


if __name__ == "__main__":
    unittest.main()
