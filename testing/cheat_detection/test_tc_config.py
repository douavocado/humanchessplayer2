"""Guards the time-control derivation of the analyser's clock thresholds.

The load-bearing test here is `test_bullet_values_are_unchanged`. Both
derivations were chosen to reproduce the previously-hardcoded 60+0 constants
exactly (2.0s and 10.0s); if either drifts, every baseline, report and tracked
band table in the repo silently changes meaning, because they were all built
against those two numbers.
"""
import unittest

from cheat_detection.config import (
    LONG_THINK_FRACTION,
    TIME_PRESSURE_FRACTION,
    AnalysisConfig,
)


class TestTimeControlDerivation(unittest.TestCase):

    def test_bullet_values_are_unchanged(self):
        """The regression guarantee: at 60+0 the derived thresholds equal the
        constants every existing result in the repo was computed against."""
        cfg = AnalysisConfig()
        self.assertEqual(cfg.initial_time, 60.0)
        self.assertEqual(cfg.long_think_secs, 2.0)
        self.assertEqual(cfg.time_pressure_secs, 10.0)

    def test_fractions_are_the_documented_ones(self):
        self.assertAlmostEqual(LONG_THINK_FRACTION, 1 / 30)
        self.assertAlmostEqual(TIME_PRESSURE_FRACTION, 1 / 6)

    def test_derives_at_three_minutes(self):
        cfg = AnalysisConfig(initial_time=180.0)
        self.assertAlmostEqual(cfg.long_think_secs, 6.0)
        self.assertAlmostEqual(cfg.time_pressure_secs, 30.0)

    def test_instant_move_secs_is_absolute(self):
        """A one-second move means the same thing at any control -- it is a
        human motor-and-decision floor, not a share of the clock."""
        self.assertEqual(AnalysisConfig(initial_time=180.0).instant_move_secs,
                         AnalysisConfig(initial_time=60.0).instant_move_secs)

    def test_setting_initial_time_after_construction_redirives(self):
        """analyze.py's _config_from_args setattrs onto an already-built cfg.
        If the thresholds were snapshotted rather than derived on read, that
        path would silently keep bullet values at another time control."""
        cfg = AnalysisConfig()
        cfg.initial_time = 180.0
        self.assertAlmostEqual(cfg.long_think_secs, 6.0)
        self.assertAlmostEqual(cfg.time_pressure_secs, 30.0)

    def test_explicit_override_beats_derivation(self):
        cfg = AnalysisConfig(initial_time=180.0,
                             long_think_secs_override=4.5,
                             time_pressure_secs_override=12.0)
        self.assertEqual(cfg.long_think_secs, 4.5)
        self.assertEqual(cfg.time_pressure_secs, 12.0)


if __name__ == "__main__":
    unittest.main()
