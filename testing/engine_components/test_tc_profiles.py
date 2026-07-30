"""Guards the time-control phase-envelope table.

The load-bearing test is `test_legacy_reproduces_current_formula`. The 60+0
calibration is the only one this repo has validated, and Task 2 replaces the
inline envelope arithmetic with a table lookup. If LEGACY is not bit-for-bit
the old formula, every 1+0 measurement silently changes meaning.
"""
import unittest
from typing import ClassVar

from common.tc_profiles import LEGACY, TC_PROFILES, apply_envelope, resolve_tc


def _legacy_expected(base, phase, initial_time):
    """The arithmetic decision_logic.py used before the table existed."""
    if phase == "opening":
        return (base ** 0.2) / 2
    if phase == "midgame":
        return base * (1.7 if initial_time > 60 else 1.4)
    return base * 0.7


class TestLegacyPinning(unittest.TestCase):

    GRID: ClassVar = [(b, t) for b in (0.1, 0.5, 1.92, 5.48, 15.12, 40.0)
                      for t in (30.0, 60.0, 61.0, 90.0, 180.0, 600.0)]

    def test_legacy_reproduces_current_formula(self):
        for base, t in self.GRID:
            for phase in ("opening", "midgame", "endgame"):
                with self.subTest(base=base, t=t, phase=phase):
                    self.assertEqual(
                        apply_envelope(LEGACY, base, phase, t),
                        _legacy_expected(base, phase, t))

    def test_midgame_branch_is_strictly_above_60(self):
        """The legacy branch is `> 60`, so 60 itself takes the 1.4 path."""
        self.assertAlmostEqual(apply_envelope(LEGACY, 1.0, "midgame", 60.0), 1.4)
        self.assertAlmostEqual(apply_envelope(LEGACY, 1.0, "midgame", 60.1), 1.7)


class TestResolution(unittest.TestCase):

    def test_unkeyed_control_resolves_to_legacy(self):
        """Exact-match-else-legacy: an unfitted control must keep today's
        behaviour rather than borrowing a neighbouring row's fit."""
        for t in (30.0, 60.0, 90.0, 300.0, 600.0):
            if t not in TC_PROFILES:
                self.assertIs(resolve_tc(t), LEGACY)

    def test_keyed_control_resolves_to_its_row(self):
        for t, prof in TC_PROFILES.items():
            self.assertIs(resolve_tc(t), prof)

    def test_legacy_reports_no_fitted_clock(self):
        self.assertIsNone(LEGACY.fitted_at)


if __name__ == "__main__":
    unittest.main()
