"""The band table's long-think column must follow --tc.

INSTANT_SECS/LONG_THINK_SECS were module-level snapshots of a default config,
so before this change the 180+0 table would have counted "long thinks" against
bullet's 2.0s threshold -- a wrong number in a tracked calibration target that
nothing downstream would have caught.
"""
import unittest
from typing import ClassVar

from cheat_detection.config import AnalysisConfig
from cheat_detection.elo_progression import _stats
from cheat_detection.features import MoveFeatures


def _move(emt):
    return MoveFeatures(
        ply=0, phase="middlegame", rank=1, within_topk=True,
        matched_top1=True, matched_top2=True, matched_top3=True,
        cp_loss=0.0, wc_loss=0.0, ambiguity=1, sharpness=0.0,
        n_legal=30, is_blunder=False, emt=emt, clock_before=100.0,
    )


class TestStatsFollowConfig(unittest.TestCase):
    """Move times of 1.5s and 4.0s: at 60+0 (threshold 2.0s) one is a long
    think; at 180+0 (threshold 6.0s) neither is."""

    MOVES: ClassVar = [_move(1.5), _move(4.0)]

    def test_bullet_threshold(self):
        s = _stats(self.MOVES, AnalysisConfig(initial_time=60.0))
        self.assertAlmostEqual(s["long_think_rate"], 0.5)

    def test_three_minute_threshold(self):
        s = _stats(self.MOVES, AnalysisConfig(initial_time=180.0))
        self.assertAlmostEqual(s["long_think_rate"], 0.0)

    def test_instant_rate_is_unaffected_by_time_control(self):
        moves = [_move(0.5), _move(4.0)]
        for t in (60.0, 180.0):
            s = _stats(moves, AnalysisConfig(initial_time=t))
            self.assertAlmostEqual(s["instant_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
