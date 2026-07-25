"""Regression test: the chess.com seek sequence follows the real page flow.

The first live seek attempt never reached the time-control grid. The sequence
was built from a screenshot in which hovering the left-nav "Play" entry opens
a submenu containing "Play Online", so the second click went to that submenu
position. Clicking "Play" does not hover it - it navigates to /play, and the
submenu leaves with the old page. The Play Online that is actually on screen
by then is a card on the right-hand panel, on the far side of the board.

Screenshots are gitignored, so as with the highlight colour table these
assert the invariant a working sequence has to satisfy - measured from
find_new_game_play_menu.png (the /play panel) and find_new_game_2/3.png (the
New Game panel) - rather than reading the images.
"""
import unittest

from sites.chess_com import (
    ChessComSite,
    PLAY_ONLINE_OFFSET,
    START_GAME_FALLBACK_OFFSET,
    START_GAME_SEARCH_BOX,
    TIME_CONTROL_DROPDOWN_OFFSET,
)

# The "Play Online" card on /play, in board steps from the board origin.
# Measured at 3840x2160: x 1404-1639, y 120-169 in a 1999-wide render of it.
CARD_X = (8.17, 10.10)
CARD_Y = (0.19, 0.60)


class _Recorder:
    """Stands in for SiteActions, recording the click sequence."""

    def __init__(self, panel_found):
        # what _find_start_game_button returns, in call order
        self._panel_found = list(panel_found)
        self.clicks = []
        self.sleeps = []
        self.logs = []

    def click(self, x, y, **kwargs):
        self.clicks.append((round(x, 1), round(y, 1)))

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def log(self, message):
        self.logs.append(message)

    def next_panel_result(self):
        if len(self._panel_found) > 1:
            return self._panel_found.pop(0)
        return self._panel_found[0]


class _StubbedSite(ChessComSite):
    """ChessComSite with the screen replaced by a scripted panel answer."""

    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder
        self.panel_probes = 0

    def _find_start_game_button(self):
        self.panel_probes += 1
        return self._recorder.next_panel_result()


class PlayOnlineOffsetTest(unittest.TestCase):

    def test_play_online_is_on_the_right_hand_panel(self):
        """The regression itself: it was at -1.71, over the left nav."""
        self.assertGreater(PLAY_ONLINE_OFFSET[0], 8.0)

    def test_play_online_lands_inside_the_measured_card(self):
        x, y = PLAY_ONLINE_OFFSET
        self.assertTrue(CARD_X[0] < x < CARD_X[1], f"x offset {x} outside the card")
        self.assertTrue(CARD_Y[0] < y < CARD_Y[1], f"y offset {y} outside the card")

    def test_play_online_shares_the_panel_column_with_the_new_game_controls(self):
        """
        Both panels occupy the same right-hand column, so a Play Online
        offset far from it would mean one of the two was mismeasured.
        """
        self.assertLess(abs(PLAY_ONLINE_OFFSET[0] - TIME_CONTROL_DROPDOWN_OFFSET[0]), 0.5)

    def test_play_online_sits_within_the_start_game_search_box(self):
        """
        Which is why the skip guard exists: on the New Game panel that same
        point is between the time-control dropdown (0.25) and Start Game
        (0.52), so clicking it there would hit a control rather than nothing.
        """
        bx, by, bw, bh = START_GAME_SEARCH_BOX
        self.assertTrue(bx <= PLAY_ONLINE_OFFSET[0] <= bx + bw)
        self.assertTrue(by <= PLAY_ONLINE_OFFSET[1] <= by + bh)
        self.assertLess(TIME_CONTROL_DROPDOWN_OFFSET[1], PLAY_ONLINE_OFFSET[1])
        self.assertLess(PLAY_ONLINE_OFFSET[1], START_GAME_FALLBACK_OFFSET[1])


class SeekSequenceTest(unittest.TestCase):

    def _run(self, panel_found, time_control="1+0"):
        rec = _Recorder(panel_found)
        site = _StubbedSite(rec)
        ok = site.start_new_game(rec, time_control)
        return ok, rec, site

    def test_clicks_play_online_when_the_panel_is_not_up(self):
        # not up on the first probe, up once clicked
        ok, rec, _site = self._run([None, (100, 200)])
        self.assertTrue(ok)
        expected = ChessComSite._at(PLAY_ONLINE_OFFSET)
        self.assertIn((round(expected[0], 1), round(expected[1], 1)), rec.clicks)

    def test_full_sequence_is_five_clicks(self):
        """Play, Play Online, dropdown, the control, Start Game."""
        ok, rec, _site = self._run([None, (100, 200)])
        self.assertTrue(ok)
        self.assertEqual(len(rec.clicks), 5)

    def test_skips_play_online_when_already_on_the_new_game_panel(self):
        ok, rec, _site = self._run([(100, 200)])
        self.assertTrue(ok)
        expected = ChessComSite._at(PLAY_ONLINE_OFFSET)
        self.assertNotIn((round(expected[0], 1), round(expected[1], 1)), rec.clicks)
        self.assertEqual(len(rec.clicks), 4)

    def test_gives_up_rather_than_clicking_blind_when_the_panel_never_loads(self):
        """
        Every remaining click is a fixed position. On a page that is not the
        New Game panel they land on whatever is there - on /play itself the
        dropdown offset is the Play Online card - so stopping is the only
        answer that cannot start the wrong kind of game.
        """
        ok, rec, site = self._run([None])
        self.assertFalse(ok)
        self.assertEqual(len(rec.clicks), 2)   # Play, Play Online, then stop
        self.assertTrue(any("never appeared" in m for m in rec.logs))
        self.assertGreater(site.panel_probes, 2, "should have retried, not probed once")

    def test_unsupported_time_control_is_refused_before_any_click(self):
        ok, rec, _site = self._run([(100, 200)], time_control="5+3")
        self.assertFalse(ok)
        self.assertEqual(rec.clicks, [])
        self.assertTrue(any("not on the default grid" in m for m in rec.logs))


if __name__ == "__main__":
    unittest.main()
