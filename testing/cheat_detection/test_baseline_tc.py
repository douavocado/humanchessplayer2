"""Baseline.initial_time: prerequisite for coexisting 60+0/180+0 baselines.

Once a 180+0 human band table exists alongside the 60+0 one, reporting a
180+0 bot against a 60+0 baseline would silently compare long_think_rate
measured at a 6s threshold against one measured at 2s. Baseline.initial_time
records what a baseline was built at so that mismatch is at least detectable
(cmd_report warns; see test_analyze_report_tc_warning.py).
"""
import os
import unittest
from unittest import mock

from cheat_detection.baseline import Baseline
from cheat_detection.config import AnalysisConfig

_REAL_LEGACY_BASELINE = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cheat_detection", "baselines", "bullet_1plus0_2300_2600.json",
)


class TestBaselineInitialTimeRoundTrip(unittest.TestCase):

    def test_legacy_real_baseline_on_disk_loads_as_none(self):
        """A real, pre-existing baseline JSON (built before this field
        existed) must keep loading, with initial_time reading as None --
        never modified, read-only fixture."""
        path = os.path.abspath(_REAL_LEGACY_BASELINE)
        self.assertTrue(os.path.exists(path), f"missing fixture: {path}")
        baseline = Baseline.from_json(path)
        self.assertIsNone(baseline.initial_time)
        # Sanity: it's a real, populated baseline, not an empty stub.
        self.assertGreater(baseline.n_units, 0)

    def test_synthetic_legacy_json_without_field_loads_as_none(self):
        """Belt-and-suspenders: a minimal hand-built legacy-shaped JSON (no
        initial_time, no values, no filters key at all) still loads."""
        import json
        import tempfile
        payload = {
            "rating_band": [2300, 2600],
            "n_units": 3,
            "stats": {"t1_rate": {"mean": 0.4, "std": 0.1, "n": 3}},
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "legacy.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            baseline = Baseline.from_json(path)
        self.assertIsNone(baseline.initial_time)
        self.assertIsNone(baseline.values)
        self.assertIsNone(baseline.filters)

    def test_initial_time_round_trips(self):
        import tempfile
        baseline = Baseline(
            rating_band=(2500, 2800),
            n_units=5,
            stats={"t1_rate": {"mean": 0.4, "std": 0.1, "n": 5}},
            initial_time=180.0,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "baseline.json")
            baseline.to_json(path)
            loaded = Baseline.from_json(path)
        self.assertEqual(loaded.initial_time, 180.0)

    def test_default_baseline_construction_has_none_initial_time(self):
        """Constructing a Baseline without passing initial_time (as any code
        written before this field existed would) must not require the arg."""
        baseline = Baseline(rating_band=(2300, 2600), n_units=1, stats={})
        self.assertIsNone(baseline.initial_time)


class TestBuildBaselineThreadsInitialTime(unittest.TestCase):

    def test_build_baseline_sets_initial_time_from_config(self):
        """build_baseline must set Baseline.initial_time from cfg.initial_time
        (no live Stockfish/PGN needed -- collect_units is stubbed out)."""
        from cheat_detection import baseline as baseline_mod

        with mock.patch.object(baseline_mod, "collect_units", return_value=[]):
            cfg = AnalysisConfig(initial_time=180.0)
            b = baseline_mod.build_baseline("unused.pgn", (2500, 2800), cfg)
        self.assertEqual(b.initial_time, 180.0)

    def test_build_baseline_at_default_60_sets_60(self):
        from cheat_detection import baseline as baseline_mod

        with mock.patch.object(baseline_mod, "collect_units", return_value=[]):
            cfg = AnalysisConfig()
            b = baseline_mod.build_baseline("unused.pgn", (2500, 2800), cfg)
        self.assertEqual(b.initial_time, 60.0)


if __name__ == "__main__":
    unittest.main()
