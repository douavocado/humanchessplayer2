"""Regression test: a new game must be detected one white move in.

A live chess.com game on 2026-07-25 was never picked up. Playing black, the
opponent had already answered with 1.e4 before the first scan - which is the
*normal* case as black, not an edge case - and ChessComSite.detect_new_game
required the placement to equal the starting position exactly, so it rejected
every scan while our clock ran down.

Lichess had always handled this (its comment: "as black the opponent may
already have moved before we scan"); the chess.com implementation was written
separately and lost it. The shared helper now lives on Site so the two cannot
diverge again, and this test pins the property for every registered site.
"""
import unittest

import chess

from sites import ChessComSite, LichessSite


def _placement_after(*sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.board_fen()


class StartLikeBoardFensTest(unittest.TestCase):
    def setUp(self):
        self.sites = [LichessSite(), ChessComSite()]

    def test_accepts_untouched_starting_position(self):
        for site in self.sites:
            with self.subTest(site=site.name):
                self.assertIn(chess.STARTING_BOARD_FEN, site.start_like_board_fens())

    def test_accepts_one_white_move_in(self):
        """The case that broke the live game: we are black and White has moved."""
        for site in self.sites:
            with self.subTest(site=site.name):
                start_like = site.start_like_board_fens()
                for opening in ("e4", "d4", "Nf3", "c4", "g3", "b3", "h4"):
                    self.assertIn(
                        _placement_after(opening), start_like,
                        f"{site.name} would not recognise a new game after 1.{opening}")

    def test_covers_every_legal_first_move(self):
        board = chess.Board()
        expected = 1 + board.legal_moves.count()   # start, plus one move in
        for site in self.sites:
            with self.subTest(site=site.name):
                self.assertEqual(len(site.start_like_board_fens()), expected)

    def test_rejects_positions_two_plies_in(self):
        """Must not be so loose that a game already under way looks new."""
        for site in self.sites:
            with self.subTest(site=site.name):
                start_like = site.start_like_board_fens()
                for line in (("e4", "e5"), ("d4", "Nf6"), ("Nf3", "d5")):
                    self.assertNotIn(_placement_after(*line), start_like)

    def test_cache_is_not_shared_between_sites(self):
        """A per-instance cache must not leak one site's set into another."""
        lichess, chess_com = LichessSite(), ChessComSite()
        first = lichess.start_like_board_fens()
        self.assertIsNot(first, ChessComSite().start_like_board_fens.__self__._start_like_fens)
        self.assertEqual(first, chess_com.start_like_board_fens())


if __name__ == "__main__":
    unittest.main()
