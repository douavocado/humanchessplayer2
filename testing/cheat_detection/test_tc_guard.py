"""The corpus-clock guard.

CLAUDE.md's corpus policy is to pin one exact time control, because mixing
clocks muddies every timing feature. This makes that policy enforceable rather
than remembered -- the failure it prevents is a silently blended population,
which produces numbers that look fine and mean nothing.
"""
import unittest

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


if __name__ == "__main__":
    unittest.main()
