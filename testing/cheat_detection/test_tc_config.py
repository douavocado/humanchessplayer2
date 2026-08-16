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


class TestParseTcSeconds(unittest.TestCase):
    """parse_tc_seconds lives in pgn_loader (the natural layering: a CLI
    depends on the loader, not the reverse) and is re-exported from analyze
    for backward compatibility. Both import paths must behave identically."""

    def test_parses_base_and_increment(self):
        from cheat_detection.analyze import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("180+0"), 180.0)
        self.assertEqual(parse_tc_seconds("60+0"), 60.0)

    def test_increment_is_optional(self):
        from cheat_detection.analyze import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("300"), 300.0)

    def test_increment_does_not_change_the_base(self):
        """initial_time is the base clock; the increment is not folded in."""
        from cheat_detection.analyze import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("180+2"), 180.0)

    def test_rejects_garbage(self):
        from cheat_detection.analyze import parse_tc_seconds
        for bad in ("", "-", "?", "blitz", "3+0min"):
            with self.assertRaises(ValueError):
                parse_tc_seconds(bad)

    def test_importable_from_pgn_loader(self):
        """The real home of the function post-move."""
        from cheat_detection.pgn_loader import parse_tc_seconds
        self.assertEqual(parse_tc_seconds("180+0"), 180.0)

    def test_analyze_and_pgn_loader_export_the_same_function(self):
        """analyze.parse_tc_seconds is a re-export, not a second definition
        that could drift from pgn_loader's."""
        from cheat_detection.analyze import parse_tc_seconds as from_analyze
        from cheat_detection.pgn_loader import parse_tc_seconds as from_loader
        self.assertIs(from_analyze, from_loader)

    def test_elo_progression_imports_from_pgn_loader_not_analyze(self):
        """Regression: elo_progression previously imported parse_tc_seconds
        from .analyze, forcing it to pull in baseline/orchestrate/parallel/
        report just to parse a string. It should depend on pgn_loader
        directly now."""
        import inspect

        from cheat_detection import elo_progression
        source = inspect.getsource(elo_progression)
        self.assertNotIn("from .analyze import parse_tc_seconds", source)
        self.assertIs(elo_progression.parse_tc_seconds,
                      __import__("cheat_detection.pgn_loader", fromlist=["parse_tc_seconds"]).parse_tc_seconds)


class TestConfigFromArgs(unittest.TestCase):
    """Verify _config_from_args preserves fetch behaviour when --tc is omitted."""

    def test_config_without_tc_uses_default(self):
        """When --tc is omitted or None, initial_time stays at the default 60.0.

        This is critical for the 'run' command: args.tc must be None so that
        fetch_user_games() streams all games unfiltered; meanwhile the config
        uses the default initial_time for threshold derivation.
        """
        from cheat_detection.analyze import _config_from_args

        class Args:
            depth = None
            multipv = None
            threads = None
            hash_mb = None
            workers = None
            flag_pvalue = None
            test_mode = None
            tc = None

        cfg = _config_from_args(Args())
        self.assertEqual(cfg.initial_time, 60.0)

    def test_run_subcommand_tc_has_no_default_in_the_real_parser(self):
        """The actual regression: an earlier implementer removed 'run's
        pre-existing `--tc` (no default) in favour of a shared one defaulting
        to "60+0", silently changing `run`'s fetch behaviour from "fetch all
        clocks" to "drop every non-60+0 game" (fetch_user_games treats
        time_control=None as unfiltered). This drives the *real* argparse
        parser, unlike the hand-built Args stub above, so it would catch a
        re-addition of default="60+0" on `run`'s --tc.
        """
        from cheat_detection.analyze import build_parser

        args = build_parser().parse_args([
            "run", "--user", "someone", "--rating", "2300", "2600",
            "--baseline", "baseline.json",
        ])
        self.assertIsNone(args.tc)

    def test_config_with_tc_parses_it(self):
        """When --tc is provided, _config_from_args parses it to initial_time."""
        from cheat_detection.analyze import _config_from_args

        class Args:
            depth = None
            multipv = None
            threads = None
            hash_mb = None
            workers = None
            flag_pvalue = None
            test_mode = None
            tc = "180+0"

        cfg = _config_from_args(Args())
        self.assertEqual(cfg.initial_time, 180.0)


if __name__ == "__main__":
    unittest.main()
