"""cmd_report warns (stderr, not error) on a baseline/report initial_time
mismatch -- reporting a 180+0 bot against a 60+0 baseline would otherwise
silently compare long_think_rate at different thresholds with nothing to
catch it. A legacy (initial_time=None) baseline must not raise, only note
that it predates the field.
"""
import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from cheat_detection import analyze
from cheat_detection.baseline import Baseline


def _args(tc="60+0"):
    class Args:
        pgn = "unused.pgn"
        baseline = "unused.json"
        player = None
        max_games = None
        min_moves = 10
        out_md = None
        out_json = None
        depth = multipv = threads = hash_mb = workers = flag_pvalue = None
        test_mode = None
        opponent_rating = None
        rating_diff = None
        allow_tc_mismatch = False
    a = Args()
    a.tc = tc
    return a


class TestReportTcWarning(unittest.TestCase):

    def _run_with_baseline(self, baseline: Baseline, tc: str):
        buf = io.StringIO()
        with mock.patch.object(Baseline, "from_json", return_value=baseline), \
             mock.patch.object(analyze, "collect_units", return_value=[]), \
             redirect_stderr(buf):
            analyze.cmd_report(_args(tc=tc))
        return buf.getvalue()

    def test_mismatched_initial_time_warns_with_both_values(self):
        baseline = Baseline(rating_band=(2500, 2800), n_units=1, stats={},
                            initial_time=60.0)
        out = self._run_with_baseline(baseline, tc="180+0")
        self.assertIn("WARNING", out)
        self.assertIn("60", out)
        self.assertIn("180", out)

    def test_matching_initial_time_does_not_warn(self):
        baseline = Baseline(rating_band=(2500, 2800), n_units=1, stats={},
                            initial_time=60.0)
        out = self._run_with_baseline(baseline, tc="60+0")
        self.assertNotIn("WARNING", out)

    def test_legacy_none_baseline_warns_as_predating_not_mismatching(self):
        baseline = Baseline(rating_band=(2500, 2800), n_units=1, stats={},
                            initial_time=None)
        out = self._run_with_baseline(baseline, tc="180+0")
        self.assertIn("WARNING", out)
        self.assertIn("predates", out.lower())
        # Must not word this as a disagreement -- there is nothing to compare.
        self.assertNotIn("disagrees", out.lower())

    def test_legacy_none_baseline_does_not_raise(self):
        """A None (legacy) baseline must never break the report -- only warn."""
        baseline = Baseline(rating_band=(2500, 2800), n_units=1, stats={},
                            initial_time=None)
        try:
            self._run_with_baseline(baseline, tc="60+0")
        except (ValueError, TypeError, AttributeError) as e:  # pragma: no cover - failure path
            self.fail(f"legacy baseline raised: {e!r}")


if __name__ == "__main__":
    unittest.main()
