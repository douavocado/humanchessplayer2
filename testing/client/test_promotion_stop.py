"""Regression: promotion-threat detection (common/board_information).

From a live flagged-quality loss (2026-07-14): with a black pawn on d2
(FEN 8/p4kpp/4p3/5p2/1bP5/1P2P3/PB1p1PPP/5K2 w - - 0 31) the bot played
g3 and let d1=Q happen -- Ke2, the only non-losing move, never reached
the search root set because the NN under-ranked it and the weird-move
penalty buried it. The alteration layers now boost promotion-stopping
moves via these helpers.
"""
import unittest

import chess

from common.board_information import (
    opponent_can_promote,
    stops_opponent_promotion,
)


class TestPromotionStop(unittest.TestCase):
    def test_live_incident_position(self):
        board = chess.Board(
            "8/p4kpp/4p3/5p2/1bP5/1P2P3/PB1p1PPP/5K2 w - - 0 31")
        self.assertTrue(opponent_can_promote(board))
        # The king step that covers d1 addresses the threat (Ke1 is
        # illegal here: the d2 pawn itself attacks e1)
        self.assertTrue(stops_opponent_promotion(board, "f1e2"))
        # The move played live, and other quiet moves, do not
        for uci in ("g2g3", "a2a3", "f1g1", "b2e5"):
            self.assertFalse(stops_opponent_promotion(board, uci))

    def test_blockade_and_coverage_address_threat(self):
        board = chess.Board("8/8/8/8/8/1k6/3p4/R5K1 w - - 0 1")
        self.assertTrue(opponent_can_promote(board))
        # Occupying the promotion square blockades it
        self.assertTrue(stops_opponent_promotion(board, "a1d1"))
        # Staying on the back rank keeps d1 covered (queen is winnable)
        self.assertTrue(stops_opponent_promotion(board, "a1b1"))
        # Abandoning the back rank lets the queen live
        self.assertFalse(stops_opponent_promotion(board, "a1a2"))

    def test_defended_promotion_square_not_addressed(self):
        # Black bishop (a4-c2-d1) defends the promotion square: covering
        # it isn't enough, the promoted queen would survive Kxd1
        board = chess.Board("8/8/8/8/b2k4/8/3p1K2/8 w - - 0 1")
        self.assertTrue(opponent_can_promote(board))
        self.assertFalse(stops_opponent_promotion(board, "f2e2"))

    def test_no_promotion_no_trigger(self):
        self.assertFalse(opponent_can_promote(chess.Board()))
        # Pawn two steps away is not an immediate promotion threat
        board = chess.Board("8/8/8/8/8/1k1p4/8/3R2K1 w - - 0 1")
        self.assertFalse(opponent_can_promote(board))


if __name__ == "__main__":
    unittest.main()
