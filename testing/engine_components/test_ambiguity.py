"""Unit test for position ambiguity, the trigger for the intuition snap gate.

Ambiguity is "how many of the engine's candidate moves are about as good as
the best one" -- 1 means the position has a single right answer (a tactical
shot to recognise), >= 2 means several near-equal tries to choose between.
`decision_logic.get_time_taken` splits the snap gate on it: humans snap
positions they can *recognise* far more often as they get stronger
(instant rate in sharp-forced positions +13.2pp from 2100-2299 to 2800+,
vs +6.1pp in quiet ones -- see docs/position-conditioned-human-likeness.md).

The number MUST match what cheat_detection measures, or every sweep of the
snap-gate deltas is tuned against a quantity we cannot observe: the window
here and `AnalysisConfig.ambiguity_wc_window` are the same 0.05, and both
compare with `<=` so the boundary is inclusive on both sides. That agreement
is what this test is really protecting.
"""
import unittest

from cheat_detection.config import AnalysisConfig
from common.search_constants import AMBIGUITY_WC_WINDOW
from engine_components.state import compute_ambiguity


class TestComputeAmbiguity(unittest.TestCase):

    def test_none_scan_is_unknown(self):
        """A failed sharpness scan leaves ambiguity unknown, not 1.

        `compute_sharpness` sets sharpness_scan = None when the scan raises;
        the gate treats None as "apply no split", degrading to the plain
        per-game gate rather than silently claiming a forced position.
        """
        self.assertIsNone(compute_ambiguity(None))

    def test_empty_scan_is_unknown(self):
        """No candidates carried a pv -- same unknown case as a failed scan."""
        self.assertIsNone(compute_ambiguity({}))

    def test_single_candidate(self):
        self.assertEqual(compute_ambiguity({"e2e4": 0.55}), 1)

    def test_one_clear_best_is_forced(self):
        """A tactical shot: the best move is far ahead of every alternative."""
        scan = {"e2e4": 0.90, "d2d4": 0.40, "g1f3": 0.35, "b1c3": 0.20}
        self.assertEqual(compute_ambiguity(scan), 1)

    def test_all_equal_is_maximally_ambiguous(self):
        scan = {"e2e4": 0.50, "d2d4": 0.50, "g1f3": 0.50, "b1c3": 0.50}
        self.assertEqual(compute_ambiguity(scan), 4)

    def test_counts_only_those_inside_the_window(self):
        """Two moves within 0.05 of the best, two clearly worse."""
        scan = {"e2e4": 0.60, "d2d4": 0.57, "g1f3": 0.40, "b1c3": 0.10}
        self.assertEqual(compute_ambiguity(scan), 2)

    def test_comfortably_inside_the_window_counts(self):
        best = 0.60
        scan = {"e2e4": best, "d2d4": best - AMBIGUITY_WC_WINDOW * 0.5}
        self.assertEqual(compute_ambiguity(scan), 2)

    def test_comfortably_outside_the_window_does_not_count(self):
        best = 0.60
        scan = {"e2e4": best, "d2d4": best - AMBIGUITY_WC_WINDOW * 2}
        self.assertEqual(compute_ambiguity(scan), 1)

    def test_agrees_with_the_analyser_at_the_boundary(self):
        """Near the cutoff, engine and analyser must make the *same* call.

        Not "the boundary is inclusive" in the idealised sense -- `0.60-0.05`
        is 0.5499999... in binary float, so a candidate constructed as
        `best - window` actually lands 4e-17 *outside* and does not count.
        cheat_detection performs the identical float comparison
        (`best_wc - wc <= cfg.ambiguity_wc_window`), so both agree; that
        agreement, not the arithmetic ideal, is the property worth pinning.
        """
        window = AnalysisConfig().ambiguity_wc_window
        best = 0.60
        for offset in (0.0, window * 0.999, window, window * 1.001, window * 2):
            wcs = [best, best - offset]
            scan = {"e2e4": wcs[0], "d2d4": wcs[1]}
            expected = sum(1 for wc in wcs if best - wc <= window)
            self.assertEqual(compute_ambiguity(scan), expected,
                             f"disagreed with the analyser at offset {offset!r}")

    def test_window_matches_the_analyser(self):
        """The engine and cheat_detection must gate/measure on one window."""
        self.assertEqual(AMBIGUITY_WC_WINDOW, AnalysisConfig().ambiguity_wc_window)


if __name__ == "__main__":
    unittest.main()
