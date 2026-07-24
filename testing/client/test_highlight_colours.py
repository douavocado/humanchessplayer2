"""Regression test for the highlight colour table in chessimage.

The 2026-07-12 game-4 flag traced back to FALLBACK_COLOUR_SCHEME's
highlight entries being channel-swapped (RGB instead of BGR like the rest
of the scheme): the swapped highlight_dark landed within matching tolerance
of Lichess's *selected-square* olive tint, so a stale selection left by a
cancelled premove was read as a last-move highlight on every scan and
poisoned turn detection for the rest of the game.

The colour table must match the real last-move highlight colours and must
NOT match the selected-square tint.
"""
import unittest

import numpy as np

from chessimage.image_scrape_utils import _build_highlight_colours, _HIGHLIGHT_TOLERANCE

# Measured from live-session debug screenshots (BGR).
LAST_MOVE_LIGHT = [205, 209, 177]   # pale cyan on light square
LAST_MOVE_DARK = [144, 151, 100]    # teal on dark square
SELECTED_DARK = [109, 161, 139]     # olive: selected piece on dark square
SELECTED_DARK_B = [112, 165, 143]   # second sample of the same tint


def _matches(colour, table):
    diff = np.abs(np.array(colour, dtype=np.int16) - table)
    return bool(np.any(np.all(diff <= _HIGHLIGHT_TOLERANCE, axis=1)))


class TestHighlightColourTable(unittest.TestCase):

    def setUp(self):
        self.table = _build_highlight_colours()

    def test_real_last_move_highlights_match(self):
        self.assertTrue(_matches(LAST_MOVE_LIGHT, self.table))
        self.assertTrue(_matches(LAST_MOVE_DARK, self.table))

    def test_selected_square_tint_does_not_match(self):
        # The incident false positive: c6 with a stale selection highlight.
        self.assertFalse(_matches(SELECTED_DARK, self.table))
        self.assertFalse(_matches(SELECTED_DARK_B, self.table))


# --- chess.com -----------------------------------------------------------
# Measured from auto_calibration/offline_screenshots/chess_com (BGR). The
# calibration profiles themselves are gitignored, so these assert the
# invariant a profile's colour scheme has to satisfy rather than reading one.
CC_LAST_MOVE_LIGHT = [154, 246, 244]   # yellow on the cream square
CC_LAST_MOVE_DARK = [94, 202, 188]     # yellow on the green square
CC_LIGHT_SQUARE = [211, 236, 235]
CC_DARK_SQUARE = [90, 148, 123]


class TestChessComHighlightSeparation(unittest.TestCase):
    """
    chess.com's highlights must be separable from its plain board.

    A live game stalled because the chess.com profile carried Lichess's teal
    highlight values: no last-move highlight ever matched, so as black
    set_game concluded it was still White's turn, produced a FEN with the
    wrong side to move, failed to link it to the starting position, and then
    waited for a turn that never came while the clock ran out.

    The board colours are the thing to stay clear of - a highlight within
    tolerance of an ordinary square would mark the whole board as moved.
    """

    def _sep(self, a, b):
        return np.abs(np.array(a, dtype=np.int16) - np.array(b, dtype=np.int16)).max()

    def test_highlights_are_distinct_from_plain_squares(self):
        for hl_name, hl in (("light", CC_LAST_MOVE_LIGHT), ("dark", CC_LAST_MOVE_DARK)):
            for sq_name, sq in (("light", CC_LIGHT_SQUARE), ("dark", CC_DARK_SQUARE)):
                with self.subTest(highlight=hl_name, square=sq_name):
                    self.assertGreater(
                        self._sep(hl, sq), _HIGHLIGHT_TOLERANCE,
                        f"chess.com {hl_name} highlight is within tolerance of the "
                        f"{sq_name} square colour")

    def test_lichess_highlights_do_not_match_chess_com_squares(self):
        """
        _build_highlight_colours always appends the Lichess fallbacks, so they
        travel with every profile. They must not light up a chess.com board.
        """
        table = _build_highlight_colours()
        self.assertFalse(_matches(CC_LIGHT_SQUARE, table))
        self.assertFalse(_matches(CC_DARK_SQUARE, table))

    def test_the_two_chess_com_highlights_are_distinguishable(self):
        self.assertGreater(
            self._sep(CC_LAST_MOVE_LIGHT, CC_LAST_MOVE_DARK), _HIGHLIGHT_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
