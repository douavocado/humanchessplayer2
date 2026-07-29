"""The corpus-clock guard.

CLAUDE.md's corpus policy is to pin one exact time control, because mixing
clocks muddies every timing feature. This makes that policy enforceable rather
than remembered -- the failure it prevents is a silently blended population,
which produces numbers that look fine and mean nothing.
"""
import os
import tempfile
import unittest

from cheat_detection.config import AnalysisConfig
from cheat_detection.parallel import collect_units
from cheat_detection.pgn_loader import (
    GameRecord,
    TimeControlMismatchError,
    check_time_control,
)


def _game(base_secs, tc="180+0"):
    return GameRecord(white="a", black="b", white_elo=2500, black_elo=2500,
                      time_control=tc, base_secs=base_secs, increment=0,
                      result="1-0", moves=[])


class TestCheckTimeControl(unittest.TestCase):

    def test_match_passes(self):
        self.assertTrue(check_time_control(_game(180), 180.0))

    def test_mismatch_raises_when_strict(self):
        with self.assertRaises(TimeControlMismatchError):
            check_time_control(_game(60, "60+0"), 180.0)

    def test_error_names_both_controls(self):
        """The message has to be actionable -- which corpus, which --tc."""
        with self.assertRaises(TimeControlMismatchError) as ctx:
            check_time_control(_game(60, "60+0"), 180.0)
        msg = str(ctx.exception)
        self.assertIn("60", msg)
        self.assertIn("180", msg)

    def test_mismatch_skips_when_not_strict(self):
        self.assertFalse(check_time_control(_game(60, "60+0"), 180.0,
                                            strict=False))

    def test_unknown_time_control_passes(self):
        """A missing or unparseable header cannot be checked, so it must not
        block analysis -- absence of evidence is not a mismatch."""
        self.assertTrue(check_time_control(_game(None, "-"), 180.0))

    def test_bullet_corpus_against_default_config_passes(self):
        """The existing 60+0 corpora must keep working untouched."""
        self.assertTrue(check_time_control(_game(60, "60+0"), 60.0))


_TWO_GAME_PGN = """[Event "?"]
[White "a"]
[Black "b"]
[TimeControl "60+0"]
[Result "1-0"]

1. e4 {[%clk 0:00:58]} 1... e5 {[%clk 0:00:59]} 2. Nf3 {[%clk 0:00:57]} 2... Nc6 {[%clk 0:00:58]} 1-0

[Event "?"]
[White "c"]
[Black "d"]
[TimeControl "60+0"]
[Result "0-1"]

1. d4 {[%clk 0:00:58]} 1... d5 {[%clk 0:00:59]} 2. c4 {[%clk 0:00:57]} 2... e6 {[%clk 0:00:58]} 0-1
"""


class TestParallelGuardConsistency(unittest.TestCase):
    """The parallel path (workers > 1) must enforce the same corpus-clock
    guard as the sequential path.

    Before this fix, `collect_units` raised `TimeControlMismatchError` at
    `workers=1` (delegating to `pipeline.iter_units`, which checks) but
    silently produced a plausible-looking, meaningless report at
    `workers>1` (`_worker` iterated games directly with no check). A unit
    test on `check_time_control` alone can't see this seam -- it lives at
    the `collect_units` entry point, not inside the function being tested.
    """

    def _write_pgn(self, tmpdir: str) -> str:
        path = os.path.join(tmpdir, "corpus.pgn")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_TWO_GAME_PGN)
        return path

    def _cfg(self, tmpdir: str, workers: int) -> AnalysisConfig:
        # cache_dir redirected to the temp dir so the test doesn't write into
        # the repo's real (shared, accumulating) engine-eval cache.
        cfg = AnalysisConfig(initial_time=180.0, cache_dir=tmpdir)
        cfg.workers = workers
        return cfg

    def test_raises_at_workers_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = self._write_pgn(tmpdir)
            cfg = self._cfg(tmpdir, workers=1)
            with self.assertRaises(TimeControlMismatchError):
                collect_units(pgn_path, cfg)

    def test_raises_at_workers_2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pgn_path = self._write_pgn(tmpdir)
            cfg = self._cfg(tmpdir, workers=2)
            with self.assertRaises(TimeControlMismatchError):
                collect_units(pgn_path, cfg)


if __name__ == "__main__":
    unittest.main()
