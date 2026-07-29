"""--tc / --allow-tc-mismatch on the four analysis CLIs that lacked them.

player_dispersion.py, mistake_impact.py, bucket_diagnostic.py and
emt_buckets.py each build an AnalysisConfig in their own main() with no way
to set the time control -- on a 180+0 corpus, player_dispersion in
particular goes through iter_units and would hard-raise
TimeControlMismatchError with no escape hatch. Each module now exposes
build_parser() so the parser can be exercised without running the (Stockfish-
requiring) analysis body.
"""
import unittest

from cheat_detection import (
    bucket_diagnostic,
    emt_buckets,
    mistake_impact,
    player_dispersion,
)


class TestCliTcFlags(unittest.TestCase):

    def test_player_dispersion_tc_flag(self):
        args = player_dispersion.build_parser().parse_args([
            "--pgn", "x.pgn", "--rating", "2500", "2800",
            "--out", "out.json", "--tc", "180+0",
        ])
        self.assertEqual(args.tc, "180+0")
        self.assertFalse(args.allow_tc_mismatch)

    def test_player_dispersion_tc_default_is_60(self):
        args = player_dispersion.build_parser().parse_args([
            "--pgn", "x.pgn", "--rating", "2500", "2800", "--out", "out.json",
        ])
        self.assertEqual(args.tc, "60+0")

    def test_player_dispersion_allow_tc_mismatch_flag(self):
        args = player_dispersion.build_parser().parse_args([
            "--pgn", "x.pgn", "--rating", "2500", "2800", "--out", "out.json",
            "--allow-tc-mismatch",
        ])
        self.assertTrue(args.allow_tc_mismatch)

    def test_mistake_impact_tc_flag(self):
        args = mistake_impact.build_parser().parse_args([
            "--pgn", "x.pgn", "--tc", "180+0",
        ])
        self.assertEqual(args.tc, "180+0")

    def test_mistake_impact_tc_default_is_60(self):
        args = mistake_impact.build_parser().parse_args(["--pgn", "x.pgn"])
        self.assertEqual(args.tc, "60+0")
        self.assertFalse(args.allow_tc_mismatch)

    def test_bucket_diagnostic_tc_flag(self):
        args = bucket_diagnostic.build_parser().parse_args([
            "--bot-pgn", "b.pgn", "--bot-player", "Bot",
            "--human-pgn", "h.pgn", "--human-rating", "2500", "2800",
            "--tc", "180+0",
        ])
        self.assertEqual(args.tc, "180+0")

    def test_bucket_diagnostic_tc_default_is_60(self):
        args = bucket_diagnostic.build_parser().parse_args([
            "--bot-pgn", "b.pgn", "--bot-player", "Bot",
            "--human-pgn", "h.pgn", "--human-rating", "2500", "2800",
        ])
        self.assertEqual(args.tc, "60+0")
        self.assertFalse(args.allow_tc_mismatch)

    def test_emt_buckets_tc_flag(self):
        args = emt_buckets.build_parser().parse_args([
            "--pgn", "x.pgn", "--tc", "180+0",
        ])
        self.assertEqual(args.tc, "180+0")

    def test_emt_buckets_tc_default_is_60(self):
        args = emt_buckets.build_parser().parse_args(["--pgn", "x.pgn"])
        self.assertEqual(args.tc, "60+0")
        self.assertFalse(args.allow_tc_mismatch)


class TestCliConfigWiring(unittest.TestCase):
    """The parsed --tc must actually reach AnalysisConfig.initial_time, the
    same way analyze.py:_config_from_args does it -- not just be accepted
    and ignored."""

    def test_player_dispersion_main_builds_cfg_with_initial_time(self):
        from unittest import mock

        from cheat_detection.config import AnalysisConfig

        captured = {}

        def fake_build(pgn, rating_band, min_units, max_games, cfg=None):
            captured["cfg"] = cfg
            return {"rating_band": list(rating_band), "min_units": min_units,
                    "n_players": 0, "features": {}}

        argv = ["--pgn", "x.pgn", "--rating", "2500", "2800",
                "--out", "/dev/null", "--tc", "180+0", "--allow-tc-mismatch"]
        with mock.patch.object(player_dispersion, "build", side_effect=fake_build), \
             mock.patch("sys.argv", ["player_dispersion.py"] + argv), \
             mock.patch("builtins.open", mock.mock_open()):
            player_dispersion.main()

        cfg = captured["cfg"]
        self.assertIsInstance(cfg, AnalysisConfig)
        self.assertEqual(cfg.initial_time, 180.0)
        self.assertFalse(cfg.strict_tc)

    def test_mistake_impact_main_builds_cfg_with_initial_time(self):
        from types import SimpleNamespace
        from unittest import mock

        captured = {}
        fake_move = SimpleNamespace(
            phase="middlegame", emt=1.0, is_blunder=False, wc_loss=0.01,
            kind=None, ply=4, sharpness=0.1, ambiguity=1,
        )

        def fake_collect(pgn_path, cfg, **kwargs):
            captured["cfg"] = cfg
            return [fake_move], [(fake_move, 10)], 1

        argv = ["--pgn", "x.pgn", "--tc", "180+0"]
        with mock.patch.object(mistake_impact, "collect", side_effect=fake_collect), \
             mock.patch("sys.argv", ["mistake_impact.py"] + argv):
            mistake_impact.main()

        cfg = captured["cfg"]
        self.assertEqual(cfg.initial_time, 180.0)
        self.assertTrue(cfg.strict_tc)  # --allow-tc-mismatch not passed

    def test_bucket_diagnostic_main_builds_cfg_with_initial_time(self):
        from unittest import mock

        captured = []

        def fake_collect(pgn_path, cfg, **kwargs):
            captured.append(cfg)
            return []

        argv = ["--bot-pgn", "b.pgn", "--bot-player", "Bot",
                "--human-pgn", "h.pgn", "--human-rating", "2500", "2800",
                "--tc", "180+0"]
        with mock.patch.object(bucket_diagnostic, "collect_bucket_units",
                              side_effect=fake_collect), \
             mock.patch("sys.argv", ["bucket_diagnostic.py"] + argv):
            bucket_diagnostic.main()

        self.assertTrue(captured)
        for cfg in captured:
            self.assertEqual(cfg.initial_time, 180.0)

    def test_emt_buckets_main_builds_cfg_with_initial_time(self):
        from unittest import mock

        captured = {}

        class FakeAnalyzer:
            def __init__(self, cfg):
                captured["cfg"] = cfg

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def flush(self):
                pass

        argv = ["--pgn", "x.pgn", "--tc", "180+0"]
        with mock.patch.object(emt_buckets, "iter_games", return_value=iter([])), \
             mock.patch.object(emt_buckets, "EngineAnalyzer", FakeAnalyzer), \
             mock.patch("sys.argv", ["emt_buckets.py"] + argv):
            emt_buckets.main()

        self.assertEqual(captured["cfg"].initial_time, 180.0)


if __name__ == "__main__":
    unittest.main()
