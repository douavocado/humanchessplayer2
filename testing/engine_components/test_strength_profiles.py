"""Tests for the target-rating dial (common/strength_profiles.py).

The dial maps a requested rating onto the knob vector that reproduces that
human rating band's measured behaviour (cheat_detection/runs/elo_progression).
Two properties matter most and both are about *honesty*, not mechanism:

  * It must not promise precision that was never measured. A 150-game arm
    resolves top-1 match to ~0.005 and adjacent 100-Elo bands differ by 0.005,
    so the real resolution is ~200-300 Elo -- hence a small set of supported
    levels rather than an arbitrary integer, and 2700 vs 2750 must give the
    same bot.
  * It must only move knobs whose rating mapping has actually been fitted.
    Today that is `quickness` alone; everything else stays at its module
    default until measured, and `CALIBRATED_KNOBS` says so in code rather than
    in a comment.
"""
import unittest

from common.constants import QUICKNESS
from common.strength_profiles import (
    CALIBRATED_KNOBS,
    STRENGTH_LEVELS,
    resolve,
    snap_rating,
)


class TestSnapping(unittest.TestCase):

    def test_supported_levels_are_returned_unchanged(self):
        for lvl in STRENGTH_LEVELS:
            self.assertEqual(snap_rating(lvl), lvl)

    def test_nearby_requests_collapse_to_one_level(self):
        """The spec's own example: the dial cannot tell 2700 from 2750."""
        self.assertEqual(snap_rating(2700), snap_rating(2750))

    def test_requests_below_and_above_the_table_clamp(self):
        self.assertEqual(snap_rating(1200), min(STRENGTH_LEVELS))
        self.assertEqual(snap_rating(4000), max(STRENGTH_LEVELS))


class TestResolve(unittest.TestCase):

    def test_returns_only_calibrated_knobs(self):
        """An uncalibrated knob must not be silently moved -- a dial that
        guesses at mappings it never fitted is worse than one that admits
        it only controls pace."""
        for lvl in STRENGTH_LEVELS:
            self.assertEqual(set(resolve(lvl)) - {"effective_rating"},
                             set(CALIBRATED_KNOBS))

    def test_quickness_is_monotone_in_rating(self):
        """Higher rating = faster play = lower quickness (bigger is slower)."""
        qs = [resolve(lvl)["quickness"] for lvl in sorted(STRENGTH_LEVELS)]
        self.assertEqual(qs, sorted(qs, reverse=True))

    def test_reports_the_level_actually_used(self):
        """The caller must be able to see the snap from the API rather than
        by reading the design doc."""
        self.assertEqual(resolve(2750)["effective_rating"], snap_rating(2750))

    def test_anchors_match_the_fitted_relationship(self):
        """quickness must invert the Phase A fit emt = 0.1964*q + 0.6234
        onto each level's target mean emt; if this drifts, the dial is
        calibrated against a relationship nobody measured."""
        targets = {2200: 1.26, 2450: 1.19, 2700: 1.065, 2850: 1.00}
        for lvl, emt in targets.items():
            q = resolve(lvl)["quickness"]
            self.assertAlmostEqual(0.1964 * q + 0.6234, emt, places=2,
                                   msg=f"level {lvl} does not hit its target emt")


class TestEngineWiring(unittest.TestCase):

    def test_default_engine_is_untouched(self):
        from engine import Engine
        eng = Engine(log_file=None)
        try:
            self.assertIsNone(eng.target_rating)
            self.assertEqual(eng.quickness, QUICKNESS)
        finally:
            eng.close_engines()

    def test_target_rating_sets_the_calibrated_knobs(self):
        from engine import Engine
        eng = Engine(log_file=None, target_rating=2850)
        try:
            self.assertEqual(eng.quickness, resolve(2850)["quickness"])
        finally:
            eng.close_engines()

    def test_explicit_knob_beats_target_rating(self):
        """Precedence is explicit arg > target_rating > module constant;
        every existing sweep depends on the first of those."""
        from engine import Engine
        eng = Engine(log_file=None, target_rating=2850, quickness=9.9)
        try:
            self.assertEqual(eng.quickness, 9.9)
        finally:
            eng.close_engines()


if __name__ == "__main__":
    unittest.main()
