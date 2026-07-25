"""Regression test: a chess.com "new game" must be on a game page.

Two live false positives, both on 2026-07-25: the bot announced a new game as
soon as the Play Online page was opened, and again while it sat in the seek
queue ("Searching.."). Neither screen is a game, but both look like one to
every signal chess.com's layout leaves us - the lobby renders a preview board
at the starting position beside a clock showing the selected time control, and
that clock never moves between game states, so neither the board nor the clock
carries the answer. The bot then "played" into a board nobody was moving.

The green Start Game button was the only lobby discriminator, and it is not
one: it is *absent* while a seek is running, which is exactly the second
screenshot. The fix is positive evidence of a game page instead - the
move-navigation bar under the moves list, which exists on every game page and
on no lobby screen.

Screenshots are gitignored, so as with the other chess.com client tests these
assert the invariant the fix has to satisfy, using bands synthesised from the
greys measured on the corpus, rather than reading the images. Measured over
that corpus with the real detector: 0.063-0.135 of the band on every game page
(playing, ended, aborted), 0.000 on every lobby screen.
"""
import unittest

import chess
import numpy as np

import sites.chess_com as chess_com
from sites.chess_com import (
    GAME_CONTROLS_BOX,
    GAME_CONTROLS_MIN_FRACTION,
    START_GAME_SEARCH_BOX,
    ChessComSite,
)

# Greys measured in the nav-bar band at 3840x2160: panel background between
# the buttons, the button fill itself, and the arrow glyphs on them.
PANEL_GREY = 32
BUTTON_GREY = 42
GLYPH_GREY = 200

BAND_W, BAND_H = 480, 62


def _band(fill):
    return np.full((BAND_H, BAND_W, 3), fill, dtype=np.uint8)


def _lobby_band():
    """The lobby panel: flat, no controls."""
    return _band(37)


def _nav_bar_band():
    """Five buttons with arrow glyphs, as the game page draws them."""
    band = _band(PANEL_GREY)
    for i in range(5):
        x = 8 + i * 96
        band[6:56, x:x + 88] = BUTTON_GREY
        # the glyph: roughly the 6% of the band that is genuinely bright
        band[22:40, x + 36:x + 52] = GLYPH_GREY
    return band


class _StubCapture:
    """Stands in for the screen: one band, whatever region is asked for."""

    def __init__(self, band):
        self.band = band
        self.requests = []

    def capture(self, box):
        self.requests.append(box)
        x, y, w, h = [int(v) for v in box]
        out = np.zeros((h, w, 4), dtype=np.uint8)
        source = self.band[:h, :w]
        out[:source.shape[0], :source.shape[1], :3] = source
        return out


class GameControlsBandTest(unittest.TestCase):
    """The band test itself, against synthesised panels."""

    def _visible(self, band, capture_cls=_StubCapture):
        site = ChessComSite()
        original = chess_com.SCREEN_CAPTURE
        chess_com.SCREEN_CAPTURE = capture_cls(band)
        try:
            return site._game_controls_visible()
        finally:
            chess_com.SCREEN_CAPTURE = original

    def test_nav_bar_reads_as_a_game_page(self):
        self.assertIs(self._visible(_nav_bar_band()), True)

    def test_flat_lobby_panel_reads_as_not_a_game_page(self):
        """The reported false positives: Play Online, and mid-seek."""
        self.assertIs(self._visible(_lobby_band()), False)

    def test_a_lighter_theme_is_read_relatively_not_absolutely(self):
        """
        Content is measured against the panel it sits on, so a theme that
        lightens both must not turn the lobby into a game page.
        """
        self.assertIs(self._visible(_band(150)), False)
        light_nav = _nav_bar_band().astype(np.int16) + 110
        self.assertIs(self._visible(np.clip(light_nav, 0, 255).astype(np.uint8)), True)

    def test_capture_failure_is_unknown_rather_than_absent(self):
        """
        None, not False: a window narrower than the calibration must fall
        back to the old behaviour, not stop the bot finding games at all.
        """
        class _Broken(_StubCapture):
            def capture(self, box):
                raise RuntimeError("out of bounds")

        class _Clipped(_StubCapture):
            def capture(self, box):
                return super().capture(box)[:, :10]

        self.assertIsNone(self._visible(_nav_bar_band(), _Broken))
        self.assertIsNone(self._visible(_nav_bar_band(), _Clipped))

    def test_threshold_sits_below_the_measured_game_page_range(self):
        """0.063 was the weakest game page in the corpus; 0.000 every lobby."""
        self.assertLess(GAME_CONTROLS_MIN_FRACTION, 0.063)
        self.assertGreater(GAME_CONTROLS_MIN_FRACTION, 0.0)

    def test_band_sits_below_the_board_and_beside_it(self):
        """
        Board-relative like the rest of the panel geometry: to the right of
        the eight files, and below the board's bottom edge where the moves
        list ends.
        """
        x, y, w, h = GAME_CONTROLS_BOX
        self.assertGreater(x, 8.0)
        self.assertGreater(y, 7.5)
        self.assertLess(y, 8.5)
        # and clear of the Start Game search box, which is the same column
        # higher up the panel - they must not be measuring the same pixels
        self.assertGreater(y, START_GAME_SEARCH_BOX[1] + START_GAME_SEARCH_BOX[3])


class _ScriptedSite(ChessComSite):
    """detect_new_game with everything but the page test forced to 'yes'."""

    def __init__(self, controls, start_button=None, modal=False):
        super().__init__()
        self._controls = controls
        self._start_button = start_button
        self._modal = modal

    def _game_controls_visible(self):
        return self._controls

    def _find_start_game_button(self):
        return self._start_button

    def game_over_screen_visible(self):
        return self._modal


class DetectNewGameTest(unittest.TestCase):
    """The whole decision, with the screen scripted around it."""

    def setUp(self):
        self._saved = {name: getattr(chess_com, name) for name in
                       ("read_clock", "capture_bottom_clock", "capture_board",
                        "get_fen_from_image")}
        chess_com.read_clock = lambda img: 60
        chess_com.capture_bottom_clock = lambda state="play": np.zeros((4, 4, 3), np.uint8)
        chess_com.capture_board = lambda: np.zeros((8, 8, 3), np.uint8)
        chess_com.get_fen_from_image = lambda img, bottom="w", fast_mode=True: chess.STARTING_FEN

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(chess_com, name, value)

    def test_finds_a_game_on_a_game_page(self):
        site = _ScriptedSite(controls=True)
        self.assertEqual(site.detect_new_game(expected_time=60), 60)

    def test_refuses_the_lobby_even_though_clock_and_board_agree(self):
        """
        The regression. Everything else about the Play Online page and the
        seek queue says "new game": full clock, starting position, no result
        modal. Only the missing game controls say otherwise.
        """
        site = _ScriptedSite(controls=False)
        self.assertIsNone(site.detect_new_game(expected_time=60))

    def test_refuses_the_lobby_while_searching_with_no_start_button(self):
        """
        Mid-seek the Start Game button is gone, so the old guard passed the
        screen straight through - this is the second live false positive.
        """
        site = _ScriptedSite(controls=False, start_button=None)
        self.assertIsNone(site.detect_new_game(expected_time=60))

    def test_still_refuses_the_new_game_panel_by_its_start_button(self):
        site = _ScriptedSite(controls=False, start_button=(100, 200))
        self.assertIsNone(site.detect_new_game(expected_time=60))

    def test_unknown_page_state_does_not_block_detection(self):
        site = _ScriptedSite(controls=None)
        self.assertEqual(site.detect_new_game(expected_time=60), 60)

    def test_result_modal_still_wins(self):
        site = _ScriptedSite(controls=True, modal=True)
        self.assertIsNone(site.detect_new_game(expected_time=60))

    def test_a_board_that_is_not_start_like_is_still_refused(self):
        board = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            board.push_san(san)
        chess_com.get_fen_from_image = lambda img, bottom="w", fast_mode=True: board.fen()
        site = _ScriptedSite(controls=True)
        self.assertIsNone(site.detect_new_game(expected_time=60))


if __name__ == "__main__":
    unittest.main()
