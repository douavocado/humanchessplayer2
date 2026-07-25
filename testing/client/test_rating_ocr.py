"""Tests for reading player ratings off the screen.

Every one of the 19 chess.com games on 2026-07-25 logged "Detected ratings:
Opponent: None, Self: None", so the engine spent all of them with the
rating-dependent behaviour (the premove snap boost in engine.py, the resign
logic in engine_components/decision_logic.py) falling back to neutral.

Two independent faults, one per layer:

  * geometry - the rating box was placed beside the *clock*, which is where
    Lichess keeps player info. chess.com puts the player line above and
    below the board and writes the rating inline after the username, so a
    fixed narrow box lands wherever that username happens to end. The
    opponent crop was reading "53)".

  * parsing - capture_rating did int() on the whole OCR string, which cannot
    parse "(1453)" however well the crop is placed.

The word samples below are real tesseract output from
logs/sessions/2026-07-25_*/errors/*_fullscreen_*.png.
"""
import unittest
from unittest.mock import MagicMock
import sys

for mod in ("fastgrab", "pytesseract", "matplotlib", "matplotlib.pyplot"):
    sys.modules.setdefault(mod, MagicMock())

from auto_calibration.coordinate_calculator import CoordinateCalculator
from chessimage.image_scrape_utils import (
    RATING_BRACKETED_MIN_CONFIDENCE,
    RATING_MIN_CONFIDENCE,
    rating_from_words,
)


def _words(line, confidence=90):
    """OCR words for a line, all at the same confidence."""
    return [(word, confidence) for word in line.split()]


class TestRatingFromWords(unittest.TestCase):

    def test_chess_com_inline_rating(self):
        self.assertEqual(rating_from_words(_words("squishypup (1453) a6 o000")), 1453)

    def test_username_ending_in_digits_is_not_the_rating(self):
        """'AlbertXu2010 (1446)' - the brackets are what disambiguates."""
        self.assertEqual(rating_from_words(_words("AlbertXu2010 (1446) ae ones D")), 1446)

    def test_username_digits_alone_are_not_a_rating(self):
        """Tesseract split the name; nothing bracketed, so nothing to trust."""
        self.assertIsNone(rating_from_words(_words("analysis 2013 Ee cece")))

    def test_misread_bracket_glyphs_still_count_as_brackets(self):
        """Real reads: '{1429)' and '(1440;' at this font size."""
        self.assertEqual(rating_from_words(_words("Urmat1012Sal {1429) = a0")), 1429)
        self.assertEqual(rating_from_words(_words("Besteverk (1440; 0000")), 1440)

    def test_lichess_bare_number(self):
        """Lichess crops the rating on its own, so the whole word is it."""
        self.assertEqual(rating_from_words([("2427", 96)]), 2427)

    def test_bare_number_needs_the_higher_confidence_floor(self):
        self.assertEqual(rating_from_words([("2427", RATING_MIN_CONFIDENCE)]), 2427)
        self.assertIsNone(rating_from_words([("2427", RATING_MIN_CONFIDENCE - 1)]))

    def test_bracketed_number_is_trusted_lower(self):
        floor = RATING_BRACKETED_MIN_CONFIDENCE
        self.assertEqual(rating_from_words([("name", 90), ("(2427)", floor)]), 2427)
        self.assertIsNone(rating_from_words([("name", 90), ("(2427)", floor - 1)]))

    def test_ambiguous_bare_numbers_are_refused(self):
        """The post-game layout puts rating and trophy rank on one line; with
        no brackets there is nothing to choose between them."""
        self.assertIsNone(rating_from_words([("2348", 90), ("1234", 90)]))

    def test_debris_out_of_range_is_ignored(self):
        """Real misreads that used to get through: '111', '400', '198'."""
        self.assertIsNone(rating_from_words(_words("analysis @0] 3 3 11 1]", confidence=37)))
        self.assertIsNone(rating_from_words([("99", 95)]))
        self.assertIsNone(rating_from_words([("40000", 95)]))

    def test_brackets_win_over_a_bare_candidate(self):
        self.assertEqual(rating_from_words([("2013", 95), ("(2381)", 80)]), 2381)

    def test_trailing_question_mark_is_stripped(self):
        """Provisional ratings; behaviour kept from the original parser."""
        self.assertEqual(rating_from_words([("2427?", 96)]), 2427)

    def test_empty_line(self):
        self.assertIsNone(rating_from_words([]))
        self.assertIsNone(rating_from_words([("", -1), ("   ", -1)]))


class TestRatingGeometry(unittest.TestCase):
    """Where the crop goes - the other half of the failure."""

    BOARD = {'x': 785, 'y': 185, 'size': 1872}   # the live 3840x2160 profile

    def _ratings(self, site, board=None):
        board = board or self.BOARD
        calculator = CoordinateCalculator(site=site)
        calculator.set_board(board)
        calculator.set_clocks({'bottom_clock': {'play': {'x': 2505, 'y': 2101,
                                                         'width': 160, 'height': 32}}})
        return calculator.calculate_all()['rating']

    def test_chess_com_rows_bracket_the_board(self):
        rating = self._ratings("chess_com")
        board_top = self.BOARD['y']
        board_bottom = self.BOARD['y'] + self.BOARD['size']
        self.assertLess(rating['opp_white']['y'] + rating['opp_white']['height'], board_top)
        self.assertGreater(rating['own_white']['y'], board_bottom)

    def test_chess_com_crop_is_a_whole_name_row(self):
        """Wide enough for a long username plus the rating - the old 70px box
        cut 'MitchellVanDerStruys (2404)' off well before the number."""
        rating = self._ratings("chess_com")['opp_white']
        self.assertGreater(rating['width'], 8 * rating['height'])

    def test_chess_com_does_not_swap_rows_by_colour(self):
        """We are always the bottom player there, whatever colour we have."""
        rating = self._ratings("chess_com")
        self.assertEqual(rating['opp_white'], rating['opp_black'])
        self.assertEqual(rating['own_white'], rating['own_black'])

    def test_chess_com_geometry_scales_with_the_board(self):
        full = self._ratings("chess_com")
        half = self._ratings("chess_com", {'x': 392, 'y': 92, 'size': 936})
        self.assertAlmostEqual(full['opp_white']['width'] / self.BOARD['size'],
                               half['opp_white']['width'] / 936, places=2)

    def test_lichess_still_swaps_rows_by_colour(self):
        """Lichess moves the player panels with the board orientation, and
        that behaviour must not change."""
        rating = self._ratings("lichess")
        self.assertEqual(rating['opp_white'], rating['own_black'])
        self.assertEqual(rating['own_white'], rating['opp_black'])
        self.assertNotEqual(rating['opp_white'], rating['own_white'])

    def test_unknown_site_falls_back_to_lichess_geometry(self):
        self.assertEqual(self._ratings("some_new_site"), self._ratings("lichess"))


if __name__ == "__main__":
    unittest.main()
