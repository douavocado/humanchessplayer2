"""Regression test: a clock misread must not be mistaken for a berserk.

The client infers "this player berserked" purely from the clock reading, and
acting on it halves the assumed initial time for the rest of the game, which
mis-paces every remaining move. The old test was a single reading below half
the initial time inside the first five moves.

That threshold is proportional, so the window a misread has to land in grows
with the clock. At 1+0 a misread has to reach 30s; at 3+0 it only has to
reach 90s, and a "3:00" scanned as "1:00" clears that easily. In
logs/sessions/2026-08-16_19-51-54 exactly that happened on move 1 of two
3+0 games out of three:

    self_clock_times: [180, 60]
    Detected to have BESERKED, reducting self initial time from 180 to 90.0

The same code also ran on chess.com, which has no berserk at all, so every
reading there could only ever have been a misread.

These tests pin the three conditions _detect_berserk now requires: the site
has berserk, the readings are consecutive, and the clock is reachable from a
halved one in the wall-clock time the game has been running.
"""
import sys
import time
import unittest
from unittest.mock import MagicMock

# clients.mp_original instantiates Engine() and CustomCursor() at module
# level; stub the display/torch-bound modules so the import works headless.
for mod in ("engine", "common.custom_cursor", "pyautogui"):
    sys.modules.setdefault(mod, MagicMock())

import clients.mp_original as mp  # noqa: E402


class _Site:
    def __init__(self, supports_berserk):
        self.supports_berserk = supports_berserk


class BerserkDetectionTest(unittest.TestCase):

    def setUp(self):
        self._site = mp.SITE
        self._game_info = dict(mp.GAME_INFO)

    def tearDown(self):
        mp.SITE = self._site
        mp.GAME_INFO.clear()
        mp.GAME_INFO.update(self._game_info)

    def _detect(self, clocks, initial, supports_berserk=True, elapsed=30.0):
        mp.SITE = _Site(supports_berserk)
        mp.GAME_INFO["game_start_wall"] = None if elapsed is None else time.time() - elapsed
        return mp._detect_berserk(clocks, initial, "Own")

    def test_logged_3plus0_misread_is_not_a_berserk(self):
        """The real failure: 3:00 read as 1:00 two seconds into a 3+0 game."""
        self.assertFalse(self._detect([180, 60], 180, elapsed=2.0))

    def test_real_berserk_is_still_detected(self):
        """A 1+0 arena game berserked to 30s, a few moves in."""
        self.assertTrue(self._detect([29, 28], 60, elapsed=20.0))

    def test_chess_com_never_berserks(self):
        """chess.com has no berserk, so a halved clock there is a misread."""
        self.assertFalse(self._detect([29, 28], 60, supports_berserk=False, elapsed=20.0))

    def test_single_low_reading_is_not_enough(self):
        """One bad frame between good ones must not halve the clock.

        Wall-clock plausible here (100s into the game), so this isolates the
        consecutive-readings requirement.
        """
        self.assertFalse(self._detect([120, 60], 180, elapsed=100.0))

    def test_clock_below_what_the_game_could_have_burnt(self):
        """Two low readings, but far too early for a halved clock to be there.

        Isolates the wall-clock check: both readings are consecutive and
        under half of 180, and only a persistent misread can produce them
        two seconds into the game.
        """
        self.assertFalse(self._detect([60, 59], 180, elapsed=2.0))

    def test_without_a_start_time_consecutive_readings_still_decide(self):
        """No wall-clock anchor (an older game record) falls back safely."""
        self.assertTrue(self._detect([29, 28], 60, elapsed=None))
        self.assertFalse(self._detect([180, 60], 180, elapsed=None))

    def test_missing_initial_time_is_not_a_berserk(self):
        self.assertFalse(self._detect([29, 28], None))
        self.assertFalse(self._detect([29, 28], 0))

    def test_unreadable_clock_is_not_a_berserk(self):
        self.assertFalse(self._detect([None, 28], 60, elapsed=20.0))


if __name__ == "__main__":
    unittest.main()
