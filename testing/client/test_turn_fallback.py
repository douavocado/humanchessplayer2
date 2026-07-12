"""Tests for the unreadable-turn fallback in clients/mp_original.py.

Regression for the 2026-07-12 game-4 flag: a stale premove highlight made
check_turn_from_last_moved return None on every scan, and the old fallback
adopted the first turn guess that patch_fens could link - a 3-ply
piece-shuffle hallucination (f8h6, e2e4, h6g7) instead of the real 2-ply
line (f8g7, e2e4) - handing the move to the opponent and flagging the game.
The fallback must rank both turn readings and prefer the shortest link.
"""
import sys
import unittest
from unittest.mock import MagicMock

# clients.mp_original instantiates Engine() and CustomCursor() at module
# level; stub the display/torch-bound modules so the import works headless.
for mod in ("engine", "common.custom_cursor", "pyautogui"):
    sys.modules.setdefault(mod, MagicMock())

import chess

from clients.mp_original import _link_candidates_for_unreadable_turn


class TestLinkCandidatesForUnreadableTurn(unittest.TestCase):

    def test_incident_prefers_real_two_ply_line(self):
        # Game 4, 2026-07-12 21:59:22: last adopted fen is after 5.d3, the
        # screen shows the position after 5...Bg7 6.e4 with the turn
        # unreadable. Both turn readings link; black-to-move (the truth)
        # must win on length.
        fen_before = "r1bqkbnr/ppp2p1p/2np2p1/4p3/2P5/2NP2P1/PP2PPBP/R1BQK1NR b KQkq - 0 5"
        scraped = "r1bqk1nr/ppp2pbp/2np2p1/4p3/2P1P3/2NP2P1/PP3PBP/R1BQK1NR w - - 0 1"
        candidates = _link_candidates_for_unreadable_turn(fen_before, scraped)
        self.assertEqual(len(candidates), 2)  # the 3-ply hallucination exists too
        n_plies, fen_after = candidates[0]
        self.assertEqual(n_plies, 2)
        self.assertEqual(chess.Board(fen_after).turn, chess.BLACK)

    def test_single_missed_ply(self):
        # Opponent just played e4 from the start position.
        board = chess.Board()
        board.push_uci("e2e4")
        scraped = board.board_fen() + " w - - 0 1"
        candidates = _link_candidates_for_unreadable_turn(chess.STARTING_FEN, scraped)
        self.assertEqual(candidates[0][0], 1)
        self.assertEqual(chess.Board(candidates[0][1]).turn, chess.BLACK)

    def test_two_missed_plies_white_perspective(self):
        # Mirror of the incident shape: we (white) moved and the opponent
        # answered before the next adopted scan.
        board = chess.Board()
        board.push_uci("d2d4")
        board.push_uci("d7d5")
        scraped = board.board_fen() + " w - - 0 1"
        candidates = _link_candidates_for_unreadable_turn(chess.STARTING_FEN, scraped)
        self.assertEqual(candidates[0][0], 2)
        self.assertEqual(chess.Board(candidates[0][1]).turn, chess.WHITE)

    def test_unchanged_board_yields_zero_ply_same_turn(self):
        # Same placement as the last fen: the 0-ply same-turn candidate must
        # win, so downstream hits the "fen has not changed" early return
        # exactly as before the fix.
        fen_before = "r1bqkbnr/ppp2p1p/2np2p1/4p3/2P5/2NP2P1/PP2PPBP/R1BQK1NR b KQkq - 0 5"
        scraped = chess.Board(fen_before).board_fen() + " w - - 0 1"
        candidates = _link_candidates_for_unreadable_turn(fen_before, scraped)
        self.assertEqual(candidates[0][0], 0)
        self.assertEqual(chess.Board(candidates[0][1]).turn, chess.BLACK)

    def test_unlinkable_board_yields_no_candidates(self):
        # Falls through to the confirmation re-capture / discard path.
        scraped = "r1bqk1nr/ppp2pbp/2np2p1/4p3/2P1P3/2NP2P1/PP3PBP/R1BQK1NR w - - 0 1"
        candidates = _link_candidates_for_unreadable_turn(chess.STARTING_FEN, scraped)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
