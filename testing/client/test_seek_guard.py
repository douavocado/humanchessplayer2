"""Regression test: the pre-seek "is a game still live?" guard.

`new_game()` refuses to click through the lobby while a game is being played,
because those clicks land on the running game. The test it used was Lichess's
- a clock readable at a live clock position, with a visible end-of-game screen
as the escape hatch - and it was applied to every site.

On chess.com that test is never false. There is one clock position, it never
moves, and a finished game goes on showing the time it ended with; the only
thing standing between it and a refused seek was the result modal, which
chess.com swaps for its analysis panel a few seconds after the game ends.
Every session on 2026-07-25 shows the consequence:

    GAME 1 ENDED
    [ERROR] Tried to seek a new game but a live game appears to be on screen
    [WARN ] Game 2 skipped, seeking again
    [INFO ] chess.com: found Start Game button at (2939, 308)   <- the retry

The bot only ever sought on the retry after the new-game wait had timed out.
Measured over all 57 `new_game_blocked_live_game` screenshots in logs/: the
modal dark fraction was 0.000-0.126, every one below the 0.20 modal
threshold, i.e. no modal was on screen in any of them.

So the question moved to the site, and chess.com answers it with the one
unambiguous statement of liveness it has: is a clock *ticking*.
"""
import unittest

import sites.chess_com as chess_com
from sites.chess_com import CLOCK_TICK_MAX_DROP, ChessComSite
from sites.lichess import LichessSite


class _Clocks:
    """Scripted clock readings: one list of (bottom, top) per capture round."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.round = 0

    def install(self, module):
        module.capture_bottom_clock = lambda state="play": ("bottom", state)
        module.capture_top_clock = lambda state="play": ("top", state)
        module.read_clock = self._read
        module.time.sleep = self._advance

    def _read(self, handle):
        which, _state = handle
        pair = self.rounds[min(self.round, len(self.rounds) - 1)]
        return pair[0 if which == "bottom" else 1]

    def _advance(self, seconds):
        self.round += 1


class _Site(ChessComSite):
    def __init__(self, modal=False):
        super().__init__()
        self._modal = modal

    def game_over_screen_visible(self):
        return self._modal


class ChessComSeekGuardTest(unittest.TestCase):

    def setUp(self):
        self._saved = {name: getattr(chess_com, name) for name in
                       ("capture_bottom_clock", "capture_top_clock", "read_clock")}
        self._saved_sleep = chess_com.time.sleep

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(chess_com, name, value)
        chess_com.time.sleep = self._saved_sleep

    def _guard(self, rounds, modal=False):
        _Clocks(rounds).install(chess_com)
        return _Site(modal=modal).live_game_on_screen()

    def test_frozen_clock_on_a_finished_board_allows_seeking(self):
        """The regression: readable, unchanging, no modal - and not a game."""
        self.assertIsNone(self._guard([(3, 2), (3, 2)]))

    def test_ticking_clock_blocks_seeking(self):
        reason = self._guard([(171, 175), (170, 175)])
        self.assertIsNotNone(reason)
        self.assertIn("ticking", reason)

    def test_the_opponents_clock_counts_as_ticking(self):
        """Playing black before our first move, only their clock moves."""
        reason = self._guard([(60, 58), (60, 57)])
        self.assertIsNotNone(reason)

    def test_a_visible_result_modal_still_allows_seeking_without_waiting(self):
        """The modal is conclusive, so the tick wait must not even happen."""
        clocks = _Clocks([(171, 175), (170, 175)])
        clocks.install(chess_com)
        self.assertIsNone(_Site(modal=True).live_game_on_screen())
        self.assertEqual(clocks.round, 0, "waited on a screen already known to be over")

    def test_no_readable_clock_allows_seeking(self):
        self.assertIsNone(self._guard([(None, None), (None, None)]))

    def test_a_jump_larger_than_a_tick_is_not_a_tick(self):
        """
        A big drop is a different clock - a new game, or a misread - not the
        one-second countdown of a game in progress.
        """
        jump = CLOCK_TICK_MAX_DROP + 10
        self.assertIsNone(self._guard([(180, 180), (180 - jump, 180)]))

    def test_a_clock_going_up_is_not_a_tick(self):
        """Clocks only count down; increment is applied at move time."""
        self.assertIsNone(self._guard([(100, 100), (103, 100)]))


class LichessSeekGuardTest(unittest.TestCase):
    """The inherited behaviour must be exactly what the client used to do."""

    def setUp(self):
        import sites.base as base
        import chessimage.image_scrape_utils as isu
        self._isu = isu
        self._saved = (isu.capture_bottom_clock, isu.read_clock)
        self._base = base

    def tearDown(self):
        self._isu.capture_bottom_clock, self._isu.read_clock = self._saved

    def _guard(self, readings, modal=False):
        """readings: clock state -> value."""
        self._isu.capture_bottom_clock = lambda state="play": state
        self._isu.read_clock = lambda state: readings.get(state)

        class _L(LichessSite):
            def game_over_screen_visible(self):
                return modal

        return _L().live_game_on_screen()

    def test_readable_live_clock_blocks_seeking(self):
        reason = self._guard({"play": 57})
        self.assertIsNotNone(reason)
        self.assertIn("play", reason)

    def test_reports_the_state_that_matched(self):
        reason = self._guard({"start1": 60})
        self.assertIn("start1", reason)

    def test_end_screen_allows_seeking_over_a_readable_clock(self):
        self.assertIsNone(self._guard({"play": 57}, modal=True))

    def test_no_clock_anywhere_allows_seeking(self):
        self.assertIsNone(self._guard({}))

    def test_every_live_state_is_probed(self):
        """Lichess moves its clock between layouts, so one state is not enough."""
        for state in LichessSite.live_clock_states:
            with self.subTest(state=state):
                self.assertIsNotNone(self._guard({state: 60}))


if __name__ == "__main__":
    unittest.main()
