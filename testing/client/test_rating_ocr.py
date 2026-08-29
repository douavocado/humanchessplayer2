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

A third fault, found 2026-08-23: the crop parsed fine and was placed fine, but
a single global Otsu threshold over a crop that also holds the board's file
watermark and the player avatar is not stable. "(1803)" read as "(1203)", the
value was cached for the whole game, and every move was then paced against a
587-point rating gap that did not exist. Covered by TestRatingVotes,
TestDeclutter and TestRatingFrameRegression below.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import sys

import cv2
import numpy as np

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAVE_TESSERACT = True
except Exception:
    HAVE_TESSERACT = False

_mocked = ["fastgrab", "matplotlib", "matplotlib.pyplot"]
if not HAVE_TESSERACT:
    _mocked.append("pytesseract")
for mod in _mocked:
    sys.modules.setdefault(mod, MagicMock())

from auto_calibration.coordinate_calculator import CoordinateCalculator
from chessimage import image_scrape_utils
from chessimage.image_scrape_utils import (
    RATING_BRACKETED_MIN_CONFIDENCE,
    RATING_MIN_CONFIDENCE,
    _drop_non_glyph_blobs,
    _is_dropped_digit,
    capture_rating_agreed,
    combine_rating_votes,
    rating_from_words,
    read_rating,
)

FRAMES = os.path.join(os.path.dirname(__file__), "frames")


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


class TestRatingVotes(unittest.TestCase):
    """Agreement, and what counts as agreement."""

    def test_a_single_read_is_never_enough(self):
        self.assertIsNone(combine_rating_votes([1803]))
        self.assertIsNone(combine_rating_votes([1803, None, None]))

    def test_two_agreeing_reads_win(self):
        self.assertEqual(combine_rating_votes([1803, None, 1803]), 1803)

    def test_a_split_read_abstains(self):
        """The 2026-08-23 crop: plain Otsu said 1203, the other passes 1803.
        With a third pass it resolves; with only the two it must not guess."""
        self.assertIsNone(combine_rating_votes([1203, 1803]))
        self.assertEqual(combine_rating_votes([1203, 1803, 1803]), 1803)

    def test_a_dropped_digit_backs_the_full_read(self):
        """'(1643)' coming back as '643' is the commonest misread of this row,
        and it still parses as a legal rating, so it used to compete."""
        self.assertEqual(combine_rating_votes([1643, 643, None]), 1643)
        self.assertEqual(combine_rating_votes([643, 1643, 643]), 1643)
        self.assertEqual(combine_rating_votes([171, 1771, 171]), 1771)
        self.assertEqual(combine_rating_votes([179, 1779, 179]), 1779)

    def test_a_truncation_alone_still_needs_corroboration(self):
        self.assertIsNone(combine_rating_votes([171, None, None]))

    def test_same_length_reads_never_merge(self):
        """Only shorter-into-longer folds, so 1586 and 1588 stay rivals."""
        self.assertIsNone(combine_rating_votes([1568, 1588, 1586]))

    def test_dropped_digit_recognition(self):
        self.assertTrue(_is_dropped_digit(643, 1643))
        self.assertTrue(_is_dropped_digit(790, 1790))
        self.assertTrue(_is_dropped_digit(171, 1771))
        self.assertFalse(_is_dropped_digit(1203, 1803))
        self.assertFalse(_is_dropped_digit(643, 1543))
        self.assertFalse(_is_dropped_digit(1643, 643))


class TestDeclutter(unittest.TestCase):
    """Removing what pollutes the threshold, without removing the text."""

    def _row(self, glyph_height, extras=()):
        img = np.zeros((33, 420), dtype=np.uint8)
        for i in range(12):
            x = 10 + i * 12
            img[8:8 + glyph_height, x:x + 6] = 255
        for (x, y, w, h) in extras:
            img[y:y + h, x:x + w] = 255
        return img

    def test_tall_clutter_is_dropped(self):
        """The board file letter behind the row spans the whole crop; the
        avatar is not far off. Neither is a character of the info line."""
        watermark = (293, 0, 21, 33)
        avatar = (350, 0, 23, 24)
        cleaned = _drop_non_glyph_blobs(self._row(10, [watermark, avatar]))
        self.assertEqual(cleaned[:, 290:].max(), 0)

    def test_the_text_survives(self):
        cleaned = _drop_non_glyph_blobs(self._row(10, [(293, 0, 21, 33)]))
        self.assertEqual(cleaned[:, :200].sum(), self._row(10)[:, :200].sum())

    def test_a_bare_number_crop_is_left_alone(self):
        """Lichess crops the rating on its own, so every blob is a digit and
        they fill most of the crop. Sizes are judged against each other, not
        against the crop, so nothing here looks oversized."""
        digits = np.zeros((24, 40), dtype=np.uint8)
        for i in range(4):
            digits[4:20, 2 + i * 9:8 + i * 9] = 255
        self.assertEqual(_drop_non_glyph_blobs(digits).sum(), digits.sum())

    def test_an_empty_crop_is_returned_unchanged(self):
        blank = np.zeros((33, 420), dtype=np.uint8)
        self.assertEqual(_drop_non_glyph_blobs(blank).sum(), 0)


class TestRatingAgreement(unittest.TestCase):
    """capture_rating_agreed: the frames have to agree, not just the passes."""

    def _reads(self, sequence):
        """capture_rating answers from `sequence`, one entry per frame."""
        frames = iter(sequence)
        current = {}

        def fake(side, position, single_pass=False):
            if position == "start":
                current["value"] = next(frames)
            return current["value"]

        return fake

    def test_later_frames_only_confirm(self):
        """The first frame is read every way; the rest cost one pass each,
        because our clock is running while set_game does this."""
        passes = []

        def fake(side, position, single_pass=False):
            passes.append(single_pass)
            return 1803

        with patch.object(image_scrape_utils, "capture_rating", side_effect=fake):
            self.assertEqual(capture_rating_agreed("own"), 1803)
        self.assertEqual(passes, [False, True])

    def test_two_agreeing_frames_are_accepted(self):
        with patch.object(image_scrape_utils, "capture_rating",
                          side_effect=self._reads([1803, 1803])):
            self.assertEqual(capture_rating_agreed("own"), 1803)

    def test_a_frame_that_disagrees_forces_a_third(self):
        with patch.object(image_scrape_utils, "capture_rating",
                          side_effect=self._reads([1203, 1803, 1803])):
            self.assertEqual(capture_rating_agreed("own"), 1803)

    def test_frames_that_never_settle_return_none(self):
        """None costs nothing - decision_logic just skips the rating factor.
        A wrong rating costs every move time in the game."""
        with patch.object(image_scrape_utils, "capture_rating",
                          side_effect=self._reads([1203, 1803, 1503])):
            self.assertIsNone(capture_rating_agreed("own"))

    def test_an_unreadable_row_returns_none(self):
        with patch.object(image_scrape_utils, "capture_rating",
                          side_effect=self._reads([None, None, None])):
            self.assertIsNone(capture_rating_agreed("own"))

    def test_the_playing_layout_is_still_the_fallback(self):
        """Lichess moves the player panels with the board orientation."""
        asked = []

        def fake(side, position, single_pass=False):
            asked.append(position)
            return 2427 if position == "playing" else None

        with patch.object(image_scrape_utils, "capture_rating", side_effect=fake):
            self.assertEqual(capture_rating_agreed("own"), 2427)
        self.assertIn("playing", asked)


@unittest.skipUnless(HAVE_TESSERACT, "needs a real tesseract binary")
class TestRatingFrameRegression(unittest.TestCase):
    """The actual 2026-08-23 crop, straight off the screen.

    logs/sessions/2026-08-23_19-46-40/errors/insane_scraped_fen_fullscreen_
    19-47-11_307.png, own-rating row. Kept as a fixture because session logs
    are pruned to 7 days. Three later frames of the same game read 1803: the
    digits are byte-identical between them and this one, and the only
    difference in the whole crop is 206 pixels of avatar in the corner.
    """

    def _crop(self):
        path = os.path.join(FRAMES, "rating_row_1803_misread_as_1203.png")
        crop = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(crop, "missing fixture " + path)
        return crop

    def test_the_row_reads_1803(self):
        self.assertEqual(read_rating(self._crop()), 1803)

    def test_one_global_threshold_alone_still_reads_1203(self):
        """The fault this fixture exists for; if tesseract ever stops making
        this mistake the fixture has lost its point and should be replaced."""
        _, binary = cv2.threshold(self._crop(), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(binary,
                                         output_type=pytesseract.Output.DICT,
                                         config='--oem 3 --psm 7')
        single = rating_from_words(zip(data['text'], (int(c) for c in data['conf'])))
        self.assertEqual(single, 1203)


if __name__ == "__main__":
    unittest.main()
