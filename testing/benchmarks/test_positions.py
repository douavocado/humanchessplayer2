"""The benchmark corpus has to stay usable without an engine to check it.

Cross-device comparisons are only meaningful if every machine measures the
same positions, so the corpus is hardcoded. That makes a bad edit silent:
an illegal FEN, or one where the side to move is already in check, would
either be skipped at runtime (quietly shrinking the corpus, so two devices
compare different subsets) or crash Stockfish outright -- the failure mode
common/utils.py:scraped_fen_sanity_issues exists to prevent.

Fast and engine-free on purpose: this runs with testing/engine_components,
not the parity harness.
"""
import pathlib
import unittest

import chess

from benchmarks.compute import _PONDER_SECS
from benchmarks.positions import POSITIONS, info_for
from common.board_information import phase_of_game
from common.utils import scraped_fen_sanity_issues


class TestBenchmarkPositions(unittest.TestCase):

    def test_every_fen_is_legal_and_playable(self):
        for fen, label in POSITIONS:
            with self.subTest(label=label):
                board = chess.Board(fen)
                self.assertTrue(board.is_valid(), f"{label}: invalid position")
                self.assertGreater(board.legal_moves.count(), 0,
                                   f"{label}: no legal move")

    def test_side_to_move_is_not_in_check(self):
        """A position already in check drags in the engine's evasion paths,
        which is a different (and far narrower) workload than the quiet
        search the benchmark is meant to time."""
        for fen, label in POSITIONS:
            with self.subTest(label=label):
                self.assertFalse(chess.Board(fen).is_check(),
                                 f"{label}: side to move is in check")

    def test_no_position_would_be_refused_by_the_engine(self):
        """calculate_analytics raises InvalidPositionError rather than let
        Stockfish segfault. turn_reliable=True is the engine's own setting."""
        for fen, label in POSITIONS:
            with self.subTest(label=label):
                issues = scraped_fen_sanity_issues(chess.Board(fen),
                                                   turn_reliable=True)
                self.assertFalse(issues, f"{label}: {issues}")

    def test_all_three_phases_are_covered(self):
        """The engine loads a different MoveScorer per phase and sizes its
        search differently, so a single-phase corpus would misreport the mix
        a real game incurs."""
        phases = {phase_of_game(chess.Board(fen)) for fen, _ in POSITIONS}
        self.assertEqual(phases, {"opening", "midgame", "endgame"})

    def test_labels_are_unique(self):
        labels = [label for _, label in POSITIONS]
        self.assertEqual(len(labels), len(set(labels)))

    def test_info_for_matches_the_position_turn(self):
        """update_info asserts turn == side; a mismatch would fail every
        measurement rather than just skewing it."""
        for fen, label in POSITIONS:
            with self.subTest(label=label):
                info = info_for(fen)
                self.assertEqual(info["side"], chess.Board(fen).turn)
                self.assertEqual(info["fens"][-1], fen)


class TestPonderLogContract(unittest.TestCase):
    """compute_floor subtracts the elective ponder from make_move_total, and
    it finds that ponder by parsing engine.py's own log line. If the wording
    drifts the regex quietly matches nothing, the subtraction becomes zero,
    and the floor silently equals make_move_total -- a wrong number rather
    than an error. Cheaper to pin the string than to run an Engine."""

    def test_engine_still_logs_the_ponder_duration_we_parse(self):
        engine_py = pathlib.Path(__file__).resolve().parents[2] / "engine.py"
        source = engine_py.read_text(encoding="utf-8")
        self.assertIn("Took {} seconds for pondering", source,
                      "engine.py no longer logs the line benchmarks/compute.py "
                      "parses to separate elective ponder from compute floor")

    def test_the_regex_matches_a_realistic_log_line(self):
        log = ("Have enough time to ponder for he next position. \n"
               "Took 0.4213 seconds for pondering. \n"
               "Took 1.5 seconds for pondering. \n")
        self.assertEqual([float(x) for x in _PONDER_SECS.findall(log)],
                         [0.4213, 1.5])


if __name__ == "__main__":
    unittest.main()
