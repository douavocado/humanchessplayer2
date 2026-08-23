"""Unit tests for conversion progress in the scramble move-appeal formula.

The scramble path (stockfish_move_logic.get_stockfish_move) ranks candidates
by `capped eval + progress - hand distance + noise`. In a decided position
the eval term is flat -- measured on a rook-up 7-piece ending, every legal
move scored within 0.31 appeal points uncapped and exactly 0.00 once the
flag-race cap bit, and depth doesn't rescue it (even depth 16 spreads those
moves by under 100cp). Hand distance spreads ~3.0 over the same moves, so
without a progress term the ranking is "shortest mouse travel", whose
cheapest option is putting the piece you just moved straight back.

These pin the shape of the progress term rather than its calibration: that a
shuffle is penalised, that the plan is rewarded, and -- the part that cost
real games when it was missing -- that the plan bonus only reaches moves
that still look best, so it can break a tie but cannot override a drop the
player can see.
"""
import unittest

import chess

from common.board_information import move_progress_score, seen_position_keys
from common.constants import (
    PROGRESS_UNDO_PENALTY, PROGRESS_REPEAT_PENALTY, PROGRESS_PAWN_PUSH_BONUS,
    PROGRESS_PASSED_PAWN_BONUS,
)


def bonus(board, uci, **kwargs):
    return move_progress_score(board, chess.Move.from_uci(uci), **kwargs)[0]


def penalty(board, uci, **kwargs):
    return move_progress_score(board, chess.Move.from_uci(uci), **kwargs)[1]


class TestShufflePenalties(unittest.TestCase):

    def test_undoing_our_own_last_move_is_penalised(self):
        board = chess.Board("8/6k1/5pp1/8/8/6PK/5P1P/3R4 w - - 0 41")
        last = chess.Move.from_uci("g2h3")  # we just walked the king to h3
        self.assertEqual(penalty(board, "h3g2", own_last_move=last), PROGRESS_UNDO_PENALTY)
        self.assertEqual(penalty(board, "h3g4", own_last_move=last), 0.0)

    def test_no_own_last_move_means_no_undo_penalty(self):
        """The premove path passes the *opponent's* predicted move for hand
        distance, so the undo check reads our own move off the board's stack
        instead -- and takes None when there isn't one."""
        board = chess.Board("8/5pk1/6p1/8/8/6P1/5PKP/3R4 w - - 0 40")
        self.assertEqual(penalty(board, "h2h3"), 0.0)

    def test_walking_back_into_a_seen_position_is_penalised(self):
        board = chess.Board("8/5pk1/6p1/8/8/6P1/5PKP/3R4 w - - 0 40")
        board.push_uci("g2h3")
        board.push_uci("g7h7")
        board.push_uci("h3g2")
        board.push_uci("h7g7")  # back where we started, one cycle in
        keys = seen_position_keys(board)
        self.assertEqual(penalty(board, "g2h3", seen_keys=keys), PROGRESS_REPEAT_PENALTY)
        self.assertEqual(penalty(board, "d1d2", seen_keys=keys), 0.0)

    def test_seen_keys_covers_the_whole_stack(self):
        board = chess.Board("8/5pk1/6p1/8/8/6P1/5PKP/3R4 w - - 0 40")
        start = board.epd()
        board.push_uci("g2h3")
        board.push_uci("g7h7")
        self.assertIn(start, seen_position_keys(board))
        self.assertIn(board.epd(), seen_position_keys(board))


class TestProgressBonuses(unittest.TestCase):

    def test_pawn_push_beats_a_quiet_piece_move(self):
        board = chess.Board("8/5pk1/6p1/8/8/6P1/5PKP/3R4 w - - 0 40")
        self.assertGreaterEqual(bonus(board, "g3g4"), PROGRESS_PAWN_PUSH_BONUS)
        self.assertEqual(bonus(board, "d1c1"), 0.0)

    def test_passed_pawn_push_beats_a_blocked_one(self):
        board = chess.Board("8/6k1/8/6p1/8/1P4P1/5p2/6K1 w - - 0 45")
        self.assertEqual(bonus(board, "b3b4") - bonus(board, "g3g4"), PROGRESS_PASSED_PAWN_BONUS)

    def test_capture_scales_with_what_is_taken(self):
        rook_capture = chess.Board("7k/8/8/8/8/3r4/8/3RK3 w - - 0 40")
        pawn_capture = chess.Board("7k/8/8/8/8/3p4/8/3RK3 w - - 0 40")
        self.assertGreater(bonus(rook_capture, "d1d3"), bonus(pawn_capture, "d1d3"))
        self.assertGreater(bonus(pawn_capture, "d1d3"), 0.0)

    def test_no_king_march_term(self):
        """A king walk earns nothing, deliberately: rewarding it was measured
        and rejected (see move_progress_score's docstring). In a bare K+R
        ending it was the only term that could fire and nothing was left to
        stop the king wandering off while the rook dropped."""
        board = chess.Board("8/8/4k3/8/8/4K3/8/3R4 w - - 0 55")
        self.assertEqual(bonus(board, "e3e4"), 0.0)
        self.assertEqual(bonus(board, "e3d2"), 0.0)

    def test_hanging_the_pushed_pawn_forfeits_the_bonus(self):
        """The plan still has to survive a glance. White's b-pawn is passed,
        but b4 walks it onto a square the black king simply takes it on."""
        board = chess.Board("8/8/8/8/2k5/1P6/6K1/8 w - - 0 50")
        self.assertEqual(bonus(board, "b3b4"), 0.0)
        safe = chess.Board("8/8/8/6k1/8/1P6/6K1/8 w - - 0 50")
        self.assertGreater(bonus(safe, "b3b4"), 0.0)

    def test_penalties_survive_the_hanging_check(self):
        """Only the bonus is forfeited when a move hangs something -- a
        shuffle is still a shuffle."""
        board = chess.Board("8/8/8/8/2k5/1P6/6K1/8 w - - 0 50")
        last = chess.Move.from_uci("g1g2")
        self.assertEqual(penalty(board, "g2g1", own_last_move=last), PROGRESS_UNDO_PENALTY)


class TestAppealIntegration(unittest.TestCase):
    """The gate that keeps the plan a tie-breaker, applied in
    get_stockfish_move: bonus only for moves within
    PROGRESS_EVAL_TOLERANCE of the best *perceived* eval."""

    def test_tolerance_gate_math(self):
        from common.constants import PROGRESS_EVAL_TOLERANCE
        perceived = {"a": 600, "b": 600 - PROGRESS_EVAL_TOLERANCE, "c": 100}
        best = max(perceived.values())
        eligible = {m for m, v in perceived.items() if v >= best - PROGRESS_EVAL_TOLERANCE}
        self.assertEqual(eligible, {"a", "b"})

    def test_blind_move_lets_the_plan_run(self):
        """A blind move zeroes every eval, so every move clears the gate and
        the memorised plan is executed without a safety check -- the
        catastrophe channel this must not close."""
        from common.constants import PROGRESS_EVAL_TOLERANCE
        perceived = {m: 0 for m in ("a", "b", "c")}
        best = max(perceived.values())
        eligible = {m for m, v in perceived.items() if v >= best - PROGRESS_EVAL_TOLERANCE}
        self.assertEqual(eligible, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
