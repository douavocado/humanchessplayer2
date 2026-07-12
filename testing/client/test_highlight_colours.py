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


if __name__ == "__main__":
    unittest.main()
