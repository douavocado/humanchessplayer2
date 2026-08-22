"""Regression: our own premove hides the square that says whose turn it is.

Lichess draws a queued premove as a near-opaque overlay on both of its
squares. We almost always premove with the piece we have just moved, so the
premove's origin *is* the last move's destination - and that destination is
the only square of the last-move pair that carries a piece, hence the only
one check_turn_from_last_moved can read a colour off. Covered up, the scan
sees a lone empty origin square, tallies nothing, and returns None.

Measured on the 2026-08-22 19:47 session: 30 turn-detection failures across
6 games, every one of them exactly one detected last-move square, every one
of them empty, and every one accompanied by exactly one premove pair. The
chess.com sessions either side of it had 0 and 1. Each failure also cost a
synchronous 4K screenshot on the scan path, pushing per-move detection
overhead from a flat ~190ms to 320-500ms in a 1+0 game.

Two things had to hold for the recovery to work:
  - the premove overlay shades have to be in the premove colour table (the
    old guesses matched the light one by luck and missed the dark one, so
    half of every pair was invisible), and
  - they must stay OUT of the last-move table, since the two mean opposite
    things (see test_highlight_colours.py).
"""
import unittest

import chess
import numpy as np

from chessimage.image_scrape_utils import (
    _HIGHLIGHT_TOLERANCE,
    STEP,
    _build_highlight_colours,
    _build_premove_colours,
    _mover_colour_count,
    check_turn_from_last_moved,
)

# Measured off 30 real 4K Lichess frames from the incident session (BGR).
# Each shade appeared on one square parity only, and identically whether a
# last-move highlight or the plain board sat underneath - the overlay is
# opaque.
PREMOVE_LIGHT = [134, 138, 133]
PREMOVE_DARK = [90, 87, 64]

# The same frames' plain board and last-move colours, for separation checks.
BOARD_LIGHT = [186, 246, 242]
BOARD_DARK = [99, 145, 105]
LAST_MOVE_LIGHT = [189, 207, 174]
LAST_MOVE_DARK = [138, 147, 94]


def _matches(colour, table):
    diff = np.abs(np.array(colour, dtype=np.int16) - table)
    return bool(np.any(np.all(diff <= _HIGHLIGHT_TOLERANCE, axis=1)))


def _sep(a, b):
    return np.abs(np.array(a, dtype=np.int16) - np.array(b, dtype=np.int16)).max()


class LichessPremoveColoursTest(unittest.TestCase):

    def test_both_overlay_shades_are_in_the_premove_table(self):
        """The dark shade used to be missed, losing half of every pair."""
        table = _build_premove_colours()
        self.assertTrue(_matches(PREMOVE_LIGHT, table))
        self.assertTrue(_matches(PREMOVE_DARK, table))

    def test_overlay_shades_are_not_in_the_last_move_table(self):
        """A queued premove must never register as a move that happened."""
        table = _build_highlight_colours()
        self.assertFalse(_matches(PREMOVE_LIGHT, table))
        self.assertFalse(_matches(PREMOVE_DARK, table))

    def test_overlay_shades_are_separable_from_board_and_last_move(self):
        for pm_name, pm in (("light", PREMOVE_LIGHT), ("dark", PREMOVE_DARK)):
            for other_name, other in (("board light", BOARD_LIGHT),
                                      ("board dark", BOARD_DARK),
                                      ("last-move light", LAST_MOVE_LIGHT),
                                      ("last-move dark", LAST_MOVE_DARK)):
                with self.subTest(premove=pm_name, other=other_name):
                    self.assertGreater(
                        _sep(pm, other), _HIGHLIGHT_TOLERANCE,
                        f"Lichess premove {pm_name} is within tolerance of {other_name}")


class TurnFromPremoveMarksTest(unittest.TestCase):
    """
    The incident position, 2026-08-22 19:53:51 (game 5, we are black).

    We played Qc8-e6 and queued the premove e6-f5. The board-image scan found
    exactly one last-move square - c8, the empty origin - because e6 was under
    the premove overlay. It is white to move, but the scraped fen always
    claims white, so the correct answer is "the turn as read is already right".
    """

    # As scraped: placement only, turn always claimed as white.
    FEN = "6k1/7p/p3q1p1/1p3p2/3Q2P1/P1P2P1P/1P3K2/8 w - - 0 1"
    BOTTOM = "b"

    def _raw(self, name):
        """Square index in board-image space (we are black, so board flipped)."""
        return 63 - chess.square_mirror(chess.parse_square(name))

    def _board_img(self, last_move_squares, premove_squares):
        """
        Paint a board image the way Lichess would: plain squares, last-move
        highlights, and the premove overlay on top of whatever it covers.
        """
        img = np.zeros((8 * STEP, 8 * STEP, 3), dtype=np.uint8)
        for raw in range(64):
            col, row = raw % 8, raw // 8
            light = (row + col) % 2 == 0
            if raw in premove_squares:
                colour = PREMOVE_LIGHT if light else PREMOVE_DARK
            elif raw in last_move_squares:
                colour = LAST_MOVE_LIGHT if light else LAST_MOVE_DARK
            else:
                colour = BOARD_LIGHT if light else BOARD_DARK
            img[row * STEP:(row + 1) * STEP, col * STEP:(col + 1) * STEP] = colour
        return img

    def test_lone_empty_origin_says_nothing_on_its_own(self):
        board = chess.Board(self.FEN)
        self.assertEqual(
            _mover_colour_count(board, [self._raw("c8")], self.BOTTOM), 0)

    def test_premove_origin_recovers_the_turn(self):
        # c8 = the last move's origin, left visible; e6 = its destination,
        # covered by the premove origin; f5 = the premove destination.
        img = self._board_img(
            last_move_squares={self._raw("c8"), self._raw("e6")},
            premove_squares={self._raw("e6"), self._raw("f5")})
        self.assertIs(
            check_turn_from_last_moved(self.FEN, img, self.BOTTOM), True)

    def test_unmasked_board_reads_the_same_turn(self):
        """The recovery must agree with what the plain highlights would say."""
        img = self._board_img(
            last_move_squares={self._raw("c8"), self._raw("e6")},
            premove_squares=set())
        self.assertIs(
            check_turn_from_last_moved(self.FEN, img, self.BOTTOM), True)

    def test_no_premove_marks_leaves_it_unreadable(self):
        """Without the premove pair there is nothing to recover from."""
        img = self._board_img(
            last_move_squares={self._raw("c8")},
            premove_squares=set())
        self.assertIsNone(
            check_turn_from_last_moved(self.FEN, img, self.BOTTOM))

    def test_recovered_turn_is_the_one_that_links(self):
        """
        Ground truth: only one of the two turn readings can be reached from
        the previously tracked position by a legal move.
        """
        from common.utils import patch_fens
        prev = "2q3k1/7p/p5p1/1p3p2/3Q2P1/P1P2P1P/1P3K2/8 b - - 0 35"
        white_to_move = chess.Board(self.FEN)
        black_to_move = chess.Board(self.FEN)
        black_to_move.turn = chess.BLACK
        self.assertIsNotNone(patch_fens(prev, white_to_move.fen()))
        self.assertIsNone(patch_fens(prev, black_to_move.fen()))


if __name__ == "__main__":
    unittest.main()
